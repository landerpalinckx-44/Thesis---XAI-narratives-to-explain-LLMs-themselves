# video_shap.py
"""
VideoSHAP - SHAP-based interpretability for video analysis with VLMs.

This module provides tools for explaining Vision-Language Model outputs on videos
by computing Shapley values for tracked objects across frames.
"""

import os
import gc
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Suppress tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter
from tqdm.auto import tqdm
import imageio

# Suppress imageio/ffmpeg warnings about pix_fmt
warnings.filterwarnings("ignore", message=".*pix_fmt.*")

try:
    from .base import BaseSHAP, ModelBase, TextVectorizer
except ImportError:
    from base import BaseSHAP, ModelBase, TextVectorizer


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrackedObject:
    """Represents a tracked object across video frames"""
    object_id: int
    label: str
    frames: Dict[int, 'ObjectFrame']  # frame_idx -> ObjectFrame
    confidence: float  # max confidence across frames

    @property
    def frame_indices(self) -> List[int]:
        return sorted(self.frames.keys())

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def __str__(self) -> str:
        """Return just the label for clean Shapley value keys"""
        return self.label


@dataclass
class ObjectFrame:
    """Object data for a single frame"""
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float


@dataclass
class VideoTrackingResult:
    """Complete tracking results for a video"""
    video_frames: List[np.ndarray]  # List of RGB frames
    objects: Dict[int, TrackedObject]  # object_id -> TrackedObject
    fps: float
    resolution: Tuple[int, int]  # (width, height)

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def get_object_labels(self) -> List[str]:
        """Get list of all object labels"""
        return [obj.label for obj in self.objects.values()]


# =============================================================================
# SAM3 Video Segmentation Model Wrapper
# =============================================================================

class BaseVideoSegmentationModel(ABC):
    """Base class for video segmentation models"""

    @abstractmethod
    def track(self,
              video_path: str,
              text_prompts: List[str],
              target_fps: float = 8,
              resolution: Optional[Tuple[int, int]] = None,
              confidence_threshold: float = 0.3) -> VideoTrackingResult:
        """
        Track objects in video based on text prompts.

        Args:
            video_path: Path to input video
            text_prompts: List of object classes to detect (e.g., ["person", "car"])
            target_fps: Target FPS for processing (lower = faster)
            resolution: Optional resize resolution (width, height)
            confidence_threshold: Minimum detection confidence

        Returns:
            VideoTrackingResult with tracked objects
        """
        pass


class SAM3VideoSegmentationModel(BaseVideoSegmentationModel):
    """
    SAM3-based video segmentation and tracking.

    Uses SAM3's text-based detection and video propagation for
    high-quality object tracking across frames.

    Requires transformers with SAM3 support (install from source):
        pip install git+https://github.com/huggingface/transformers.git
    """

    def __init__(self,
                 model_name: str = "facebook/sam3",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 use_compile: bool = True,
                 max_cond_frames: int = 30,
                 max_non_cond_frames: int = 10,
                 verbose: bool = False):
        """
        Initialize SAM3 video tracker.

        Args:
            model_name: HuggingFace model name (e.g., "facebook/sam3")
            device: Device to run on ('cuda' or 'cpu')
            dtype: Torch dtype for model
            use_compile: Whether to torch.compile the model
            max_cond_frames: Max conditioning frames to keep in memory
            max_non_cond_frames: Max non-conditioning frames to keep
            verbose: Print progress messages
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_compile = use_compile
        self.max_cond_frames = max_cond_frames
        self.max_non_cond_frames = max_non_cond_frames
        self.verbose = verbose

        # Models (set during initialization)
        self.sam_model = None
        self.sam_processor = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of SAM3 model"""
        if self._initialized:
            return

        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError:
            raise ImportError(
                "SAM3 Video requires transformers with SAM3 support. "
                "Install from source: pip install git+https://github.com/huggingface/transformers.git"
            )

        torch.cuda.empty_cache()
        gc.collect()

        if self.verbose:
            print(f"Loading SAM3 Video model: {self.model_name}")

        self.sam_model = Sam3VideoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)

        if self.use_compile and self.device == "cuda":
            try:
                self.sam_model = torch.compile(
                    self.sam_model, mode="reduce-overhead", fullgraph=False
                )
                if self.verbose:
                    print("Model compiled successfully")
            except Exception as e:
                if self.verbose:
                    print(f"Compilation failed, using eager mode: {e}")

        self.sam_processor = Sam3VideoProcessor.from_pretrained(self.model_name)
        self._initialized = True

        if self.verbose:
            print("SAM3 Video initialized successfully")

    def _setup_memory_cleanup(self, session):
        """Apply memory optimization to session"""
        original_store_output = session.store_output

        def store_output_with_cleanup(obj_idx, frame_idx, output_key=None,
                                      output_value=None, is_conditioning_frame=True):
            original_store_output(obj_idx, frame_idx, output_key, output_value, is_conditioning_frame)

            if output_key is None:
                # Clean up old conditioning frames
                if "cond_frame_outputs" in session.output_dict_per_obj[obj_idx]:
                    cond_frames = list(session.output_dict_per_obj[obj_idx]["cond_frame_outputs"].keys())
                    if len(cond_frames) > self.max_cond_frames:
                        for old_frame in sorted(cond_frames)[:-self.max_cond_frames]:
                            frame_data = session.output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(old_frame, None)
                            if frame_data:
                                for v in frame_data.values():
                                    if isinstance(v, torch.Tensor):
                                        del v

                # Clean up old non-conditioning frames
                if "non_cond_frame_outputs" in session.output_dict_per_obj[obj_idx]:
                    non_cond_frames = list(session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].keys())
                    if len(non_cond_frames) > self.max_non_cond_frames:
                        for old_frame in sorted(non_cond_frames)[:-self.max_non_cond_frames]:
                            frame_data = session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(old_frame, None)
                            if frame_data:
                                for v in frame_data.values():
                                    if isinstance(v, torch.Tensor):
                                        del v

        session.store_output = store_output_with_cleanup

    def _load_video_frames(self,
                           video_path: str,
                           target_fps: float,
                           resolution: Optional[Tuple[int, int]]) -> Tuple[List[Image.Image], float, Tuple[int, int]]:
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame skip for target FPS
        frame_skip = max(1, int(original_fps / target_fps)) if target_fps else 1
        actual_fps = original_fps / frame_skip

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                if resolution:
                    pil_frame = pil_frame.resize(resolution, Image.LANCZOS)

                frames.append(pil_frame)

            frame_idx += 1

        cap.release()

        actual_resolution = resolution if resolution else (frame_width, frame_height)

        return frames, actual_fps, actual_resolution

    def _get_bbox_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Extract bounding box from binary mask"""
        if not mask.any():
            return None

        y_positions, x_positions = np.where(mask > 0.5)
        if len(y_positions) == 0:
            return None

        return (
            int(np.min(x_positions)),
            int(np.min(y_positions)),
            int(np.max(x_positions)),
            int(np.max(y_positions))
        )

    def track(self,
              video_path: str,
              text_prompts: List[str],
              target_fps: float = 8,
              resolution: Optional[Tuple[int, int]] = None,
              confidence_threshold: float = 0.3) -> VideoTrackingResult:
        """
        Track objects in video using SAM3.

        Args:
            video_path: Path to input video
            text_prompts: List of object classes to detect
            target_fps: Target FPS for processing
            resolution: Optional resize resolution (width, height)
            confidence_threshold: Minimum detection confidence

        Returns:
            VideoTrackingResult with all tracked objects
        """
        self._initialize()

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        if self.verbose:
            print(f"Loading video: {video_path}")
            print(f"Text prompts: {text_prompts}")

        # Load frames
        frames, actual_fps, actual_resolution = self._load_video_frames(
            video_path, target_fps, resolution
        )

        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")

        if self.verbose:
            print(f"Loaded {len(frames)} frames at {actual_fps:.1f} FPS")

        # Track with each prompt
        all_outputs_by_prompt = {}

        for prompt_idx, prompt in enumerate(text_prompts):
            if self.verbose:
                print(f"\nProcessing prompt {prompt_idx + 1}/{len(text_prompts)}: '{prompt}'")

            # Initialize session
            session = self.sam_processor.init_video_session(
                video=frames,
                inference_device=self.device,
                inference_state_device="cpu",
                processing_device=self.device,
                video_storage_device="cpu",
                max_vision_features_cache_size=1,
                dtype=self.dtype,
            )

            session = self.sam_processor.add_text_prompt(session, text=prompt)
            self._setup_memory_cleanup(session)

            # Process frames
            outputs_per_frame = {}

            with torch.inference_mode():
                with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
                    for frame_count, frame_outputs in enumerate(
                        self.sam_model.propagate_in_video_iterator(session), 1
                    ):
                        processed = self.sam_processor.postprocess_outputs(session, frame_outputs)
                        outputs_per_frame[frame_outputs.frame_idx] = {
                            'processed': processed,
                            'prompt': prompt
                        }

                        if frame_count % 16 == 0:
                            torch.cuda.empty_cache()

            all_outputs_by_prompt[prompt] = outputs_per_frame

            del session
            torch.cuda.empty_cache()
            gc.collect()

        # Convert frames to numpy
        video_frames_np = [np.array(f) for f in frames]

        # Merge outputs into TrackedObjects
        objects = self._merge_outputs(
            all_outputs_by_prompt,
            confidence_threshold,
            actual_resolution
        )

        if self.verbose:
            print(f"\nTracked {len(objects)} unique objects")

        return VideoTrackingResult(
            video_frames=video_frames_np,
            objects=objects,
            fps=actual_fps,
            resolution=actual_resolution
        )

    def _merge_outputs(self,
                       all_outputs_by_prompt: Dict,
                       confidence_threshold: float,
                       resolution: Tuple[int, int]) -> Dict[int, TrackedObject]:
        """Merge outputs from all prompts into TrackedObjects"""
        objects = {}
        next_object_id = 0

        # Collect all frame indices
        all_frame_indices = set()
        for outputs in all_outputs_by_prompt.values():
            all_frame_indices.update(outputs.keys())

        for frame_idx in sorted(all_frame_indices):
            for prompt, outputs in all_outputs_by_prompt.items():
                if frame_idx not in outputs:
                    continue

                processed = outputs[frame_idx]['processed']
                if processed is None:
                    continue

                n_objects = len(processed.get('object_ids', []))

                for i in range(n_objects):
                    score = processed['scores'][i].item() if 'scores' in processed else 1.0

                    if score < confidence_threshold:
                        continue

                    # Get mask
                    mask = processed['masks'][i]
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    mask = (mask > 0.5).astype(np.uint8)

                    # Get bbox
                    bbox = self._get_bbox_from_mask(mask)
                    if bbox is None:
                        continue

                    # Check mask area
                    area = np.sum(mask)
                    min_area = resolution[0] * resolution[1] * 0.001  # 0.1% of image
                    if area < min_area:
                        continue

                    # Get or create object (using prompt + SAM3 object_id as key)
                    sam3_obj_id = int(processed['object_ids'][i])
                    obj_key = f"{prompt}_{sam3_obj_id}"

                    # Find existing object or create new
                    found_obj_id = None
                    for obj_id, obj in objects.items():
                        if hasattr(obj, '_internal_key') and obj._internal_key == obj_key:
                            found_obj_id = obj_id
                            break

                    if found_obj_id is None:
                        found_obj_id = next_object_id
                        objects[found_obj_id] = TrackedObject(
                            object_id=found_obj_id,
                            label=prompt,
                            frames={},
                            confidence=score
                        )
                        objects[found_obj_id]._internal_key = obj_key
                        next_object_id += 1

                    # Add frame data
                    objects[found_obj_id].frames[frame_idx] = ObjectFrame(
                        mask=mask,
                        bbox=bbox,
                        score=score
                    )

                    # Update max confidence
                    if score > objects[found_obj_id].confidence:
                        objects[found_obj_id].confidence = score

        return objects


# =============================================================================
# Video VLM Models
# =============================================================================

class VideoModelBase(ModelBase):
    """Base class for video-capable VLMs"""

    @abstractmethod
    def generate_from_video(self,
                           prompt: str,
                           video_path: Optional[str] = None,
                           frames: Optional[List[np.ndarray]] = None) -> str:
        """
        Generate response from video input.

        Args:
            prompt: Text prompt/question about the video
            video_path: Path to video file (for native video support)
            frames: List of RGB frames (for frame-based approach)

        Returns:
            Model response text
        """
        pass


class GeminiVideoModel(VideoModelBase):
    """
    Google Gemini model with native video support.

    Uploads video files directly to Gemini for analysis.
    """

    def __init__(self,
                 model_name: str = "gemini-2.0-flash",
                 api_key: Optional[str] = None,
                 fps: Optional[float] = None,
                 temperature: Optional[float] = None):
        """
        Initialize Gemini video model.

        Args:
            model_name: Gemini model name
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            fps: Frame rate for video sampling (default: 1 FPS, supports 0.1-60)
            temperature: Sampling temperature (0.0-2.0, lower = more deterministic)
        """
        super().__init__(model_name, api_key=api_key)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.fps = fps
        self.temperature = temperature
        self._client = None

    def _initialize(self):
        if self._client is not None:
            return

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package required. "
                "Install with: pip install google-generativeai"
            )

        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, image_path: Optional[str] = None, video_path: Optional[str] = None) -> str:
        """Generate from text/image/video (inherited interface)"""
        self._initialize()

        import google.generativeai as genai

        # If video_path is provided, delegate to generate_from_video
        if video_path:
            return self.generate_from_video(prompt=prompt, video_path=video_path)

        if image_path:
            image = Image.open(image_path)
            response = self._client.generate_content([prompt, image])
        else:
            response = self._client.generate_content(prompt)

        return response.text

    def generate_from_video(self,
                           prompt: str,
                           video_path: Optional[str] = None,
                           frames: Optional[List[np.ndarray]] = None) -> str:
        """
        Generate from video using native video upload with optional FPS control.

        Args:
            prompt: Question/prompt about the video
            video_path: Path to video file
            frames: Fallback: list of frames as images

        Returns:
            Model response
        """
        self._initialize()

        import google.generativeai as genai

        if video_path:
            # Check file size for inline vs upload approach
            file_size = os.path.getsize(video_path)

            if file_size < 20 * 1024 * 1024 and self.fps:  # <20MB and FPS specified
                # Use inline upload with FPS control via new SDK
                try:
                    from google import genai as genai_new
                    from google.genai import types

                    client = genai_new.Client(api_key=self.api_key)
                    video_bytes = open(video_path, 'rb').read()

                    # Build config with temperature if specified
                    config = None
                    if self.temperature is not None:
                        config = types.GenerateContentConfig(temperature=self.temperature)

                    response = client.models.generate_content(
                        model=f'models/{self.model_name}',
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    inline_data=types.Blob(
                                        data=video_bytes,
                                        mime_type='video/mp4'
                                    ),
                                    video_metadata=types.VideoMetadata(fps=self.fps)
                                ),
                                types.Part(text=prompt)
                            ]
                        ),
                        config=config
                    )
                    return response.text
                except ImportError:
                    pass  # Fall back to file upload

            # File upload approach (no FPS control)
            video_file = genai.upload_file(video_path)

            # Wait for processing
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(f"Video processing failed: {video_file.state.name}")

            # Build generation config with temperature if specified
            generation_config = None
            if self.temperature is not None:
                generation_config = {"temperature": self.temperature}

            response = self._client.generate_content(
                [prompt, video_file],
                generation_config=generation_config
            )

            # Clean up
            genai.delete_file(video_file.name)

        elif frames:
            # Fallback: send frames as images
            images = [Image.fromarray(f) for f in frames]
            content = [prompt] + images
            generation_config = None
            if self.temperature is not None:
                generation_config = {"temperature": self.temperature}
            response = self._client.generate_content(content, generation_config=generation_config)
        else:
            raise ValueError("Either video_path or frames must be provided")

        return response.text


class OpenAIVideoModel(VideoModelBase):
    """
    OpenAI GPT-4o model using frame sequence approach.

    Sends multiple frames as images to simulate video understanding.
    """

    def __init__(self,
                 model_name: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 max_frames: int = 8):
        """
        Initialize OpenAI video model.

        Args:
            model_name: OpenAI model name (gpt-4o recommended)
            api_key: OpenAI API key
            max_frames: Maximum frames to send (cost/context limit)
        """
        super().__init__(model_name, api_key=api_key)
        self.max_frames = max_frames
        self._client = None

    def _initialize(self):
        if self._client is not None:
            return

        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, image_path: Optional[str] = None, video_path: Optional[str] = None) -> str:
        """Generate from text/image/video"""
        self._initialize()

        import base64

        # If video_path is provided, delegate to generate_from_video
        if video_path:
            return self.generate_from_video(prompt=prompt, video_path=video_path)

        messages = []

        if image_path:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.5
        )

        return response.choices[0].message.content

    def generate_from_video(self,
                           prompt: str,
                           video_path: Optional[str] = None,
                           frames: Optional[List[np.ndarray]] = None) -> str:
        """
        Generate from video using frame sequence.

        Args:
            prompt: Question about the video
            video_path: Path to video (will extract frames)
            frames: Pre-extracted frames

        Returns:
            Model response
        """
        self._initialize()

        import base64
        from io import BytesIO

        # Get frames
        if frames is None and video_path:
            frames = self._extract_frames(video_path)

        if not frames:
            raise ValueError("No frames available")

        # Select key frames
        key_indices = self._select_key_frames(len(frames), self.max_frames)
        selected_frames = [frames[i] for i in key_indices]

        # Build message with frame sequence
        content = [
            {"type": "text", "text": f"The following {len(selected_frames)} images are frames from a video, shown in chronological order.\n\n{prompt}"}
        ]

        for i, frame in enumerate(selected_frames):
            # Convert to base64
            pil_image = Image.fromarray(frame)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.5
        )

        return response.choices[0].message.content

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    def _select_key_frames(self, num_frames: int, max_frames: int) -> List[int]:
        """Select evenly spaced key frames"""
        if num_frames <= max_frames:
            return list(range(num_frames))

        step = num_frames / max_frames
        return [int(i * step) for i in range(max_frames)]


class QwenVideoModel(VideoModelBase):
    """
    Qwen3-VL model for local video understanding.

    Runs locally using HuggingFace transformers.
    """

    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda",
                 max_frames: int = 16,
                 torch_dtype: str = "bfloat16"):
        """
        Initialize Qwen video model.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            max_frames: Maximum frames to process
            torch_dtype: Torch data type
        """
        super().__init__(model_name)
        self.device = device
        self.max_frames = max_frames
        self.torch_dtype = getattr(torch, torch_dtype)

        self._model = None
        self._processor = None

    def _initialize(self):
        if self._model is not None:
            return

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Qwen VL requires transformers>=4.40. "
                "Install with: pip install transformers>=4.40"
            )

        print(f"Loading {self.model_name}...")

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        print("Qwen model loaded successfully")

    def generate(self, prompt: str, image_path: Optional[str] = None, video_path: Optional[str] = None) -> str:
        """Generate from text/image/video"""
        self._initialize()

        # If video_path is provided, delegate to generate_from_video
        if video_path:
            return self.generate_from_video(prompt=prompt, video_path=video_path)

        if image_path:
            image = Image.open(image_path)
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image_path:
            inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        else:
            inputs = self._processor(text=[text], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=512)

        return self._processor.decode(outputs[0], skip_special_tokens=True)

    def generate_from_video(self,
                           prompt: str,
                           video_path: Optional[str] = None,
                           frames: Optional[List[np.ndarray]] = None) -> str:
        """
        Generate from video using Qwen's video support.

        Args:
            prompt: Question about the video
            video_path: Path to video file
            frames: Pre-extracted frames

        Returns:
            Model response
        """
        self._initialize()

        # Get frames
        if frames is None and video_path:
            frames = self._extract_frames(video_path)

        if not frames:
            raise ValueError("No frames available")

        # Select key frames
        key_indices = self._select_key_frames(len(frames), self.max_frames)
        selected_frames = [Image.fromarray(frames[i]) for i in key_indices]

        # Build message with video frames
        content = [{"type": "video", "video": selected_frames}]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], videos=[selected_frames], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=512)

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self._processor.decode(generated_ids, skip_special_tokens=True)

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    def _select_key_frames(self, num_frames: int, max_frames: int) -> List[int]:
        """Select evenly spaced key frames"""
        if num_frames <= max_frames:
            return list(range(num_frames))

        step = num_frames / max_frames
        return [int(i * step) for i in range(max_frames)]


# =============================================================================
# Video Manipulators
# =============================================================================

class VideoManipulator(ABC):
    """Base class for video manipulation (object masking)"""

    @abstractmethod
    def mask_object(self,
                   frames: List[np.ndarray],
                   tracked_object: TrackedObject) -> List[np.ndarray]:
        """
        Mask/hide an object across all frames.

        Args:
            frames: List of RGB frames
            tracked_object: Object to mask

        Returns:
            Modified frames with object masked
        """
        pass

    def create_masked_video(self,
                           frames: List[np.ndarray],
                           objects_to_mask: List[TrackedObject]) -> List[np.ndarray]:
        """
        Create video with multiple objects masked.

        Args:
            frames: Original frames
            objects_to_mask: List of objects to mask

        Returns:
            Modified frames
        """
        result_frames = [f.copy() for f in frames]

        for obj in objects_to_mask:
            result_frames = self.mask_object(result_frames, obj)

        return result_frames


class VideoBlurManipulator(VideoManipulator):
    """Blur objects to hide them"""

    def __init__(self, blur_radius: int = 51):
        """
        Args:
            blur_radius: Gaussian blur kernel size (odd number)
        """
        self.blur_radius = blur_radius
        if self.blur_radius % 2 == 0:
            self.blur_radius += 1

    def mask_object(self,
                   frames: List[np.ndarray],
                   tracked_object: TrackedObject) -> List[np.ndarray]:
        """Blur object in all frames"""
        result = [f.copy() for f in frames]

        for frame_idx, frame in enumerate(result):
            if frame_idx not in tracked_object.frames:
                continue

            obj_frame = tracked_object.frames[frame_idx]
            mask = obj_frame.mask

            # Ensure mask matches frame size
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

            # Apply blur
            blurred = cv2.GaussianBlur(frame, (self.blur_radius, self.blur_radius), 0)

            # Composite: use blurred where mask is True
            mask_3ch = np.stack([mask, mask, mask], axis=-1) > 0
            result[frame_idx] = np.where(mask_3ch, blurred, frame)

        return result


class VideoBlackoutManipulator(VideoManipulator):
    """Black out objects to hide them, with optional bbox masking and overlap preservation"""

    def __init__(self,
                 fill_color: Tuple[int, int, int] = (0, 0, 0),
                 mask_type: str = "precise",
                 preserve_overlapping: bool = False,
                 expansion_pixels: int = 2):
        """
        Args:
            fill_color: RGB color to fill masked regions
            mask_type: Type of mask to use:
                - "precise": Exact segmentation mask (default)
                - "bbox": Full bounding box (use with preserve_overlapping for best results)
            preserve_overlapping: Whether to preserve pixels from objects not being masked.
                When True with mask_type='bbox', this enables bounding box masking while
                restoring pixels of other objects that intersect with the bounding box.
            expansion_pixels: Number of pixels to expand bbox by (only for bbox mode)
        """
        self.fill_color = fill_color
        self.mask_type = mask_type
        self.preserve_overlapping = preserve_overlapping
        self.expansion_pixels = expansion_pixels

    def mask_object(self,
                   frames: List[np.ndarray],
                   tracked_object: TrackedObject,
                   preserve_objects: Optional[List[TrackedObject]] = None) -> List[np.ndarray]:
        """
        Black out object in all frames while optionally preserving other objects.

        Args:
            frames: List of RGB frames
            tracked_object: Object to mask/hide
            preserve_objects: Optional list of objects whose pixels should be preserved
                (not blacked out even if they intersect with the masked region)

        Returns:
            Modified frames with object masked
        """
        result = [f.copy() for f in frames]

        for frame_idx, frame in enumerate(result):
            if frame_idx not in tracked_object.frames:
                continue

            obj_frame = tracked_object.frames[frame_idx]
            mask = obj_frame.mask

            # Ensure mask matches frame size
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

            # Get the masking region based on mask_type
            if self.mask_type == "bbox":
                # Use bounding box instead of precise segmentation
                mask_region = self._get_bbox_mask(mask, frame.shape[:2])
            else:
                # Use precise segmentation mask
                mask_region = mask

            # Create preserve mask if needed
            if self.preserve_overlapping and preserve_objects:
                preserve_mask = self._create_preserve_mask(
                    preserve_objects, frame_idx, frame.shape[:2]
                )
                # Apply blackout only to pixels not in preserve mask
                mask_bool = (mask_region > 0) & (~preserve_mask)
            else:
                mask_bool = mask_region > 0

            # Apply blackout
            result[frame_idx][mask_bool] = self.fill_color

        return result

    def _get_bbox_mask(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a bounding box mask from segmentation mask.

        Args:
            mask: Binary segmentation mask
            frame_shape: (height, width) of the frame

        Returns:
            Binary mask covering the bounding box region
        """
        # Compute bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return mask

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Expand bounding box if needed
        if self.expansion_pixels > 0:
            rmin = max(0, rmin - self.expansion_pixels)
            rmax = min(frame_shape[0] - 1, rmax + self.expansion_pixels)
            cmin = max(0, cmin - self.expansion_pixels)
            cmax = min(frame_shape[1] - 1, cmax + self.expansion_pixels)

        # Create a new mask from bounding box
        bbox_mask = np.zeros_like(mask)
        bbox_mask[rmin:rmax+1, cmin:cmax+1] = 1
        return bbox_mask

    def _create_preserve_mask(self,
                              preserve_objects: List[TrackedObject],
                              frame_idx: int,
                              frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a mask of pixels to preserve from other objects.

        This creates a union of all segmentation masks from objects that should
        be preserved, so their pixels won't be blacked out even if they intersect
        with the bounding box of the object being masked.

        Args:
            preserve_objects: List of TrackedObjects to preserve
            frame_idx: Current frame index
            frame_shape: (height, width) of the frame

        Returns:
            Boolean mask where True indicates pixels to preserve
        """
        preserve_mask = np.zeros(frame_shape, dtype=bool)

        for obj in preserve_objects:
            if frame_idx not in obj.frames:
                continue
            obj_mask = obj.frames[frame_idx].mask

            # Resize if needed
            if obj_mask.shape[:2] != frame_shape:
                obj_mask = cv2.resize(obj_mask, (frame_shape[1], frame_shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            preserve_mask |= (obj_mask > 0)

        return preserve_mask

    def create_masked_video(self,
                           frames: List[np.ndarray],
                           objects_to_mask: List[TrackedObject],
                           objects_to_preserve: Optional[List[TrackedObject]] = None) -> List[np.ndarray]:
        """
        Create video with multiple objects masked while preserving others.

        This override adds support for preserve_overlapping functionality,
        where objects_to_preserve will have their pixels restored even if
        they intersect with the bounding boxes of objects being masked.

        Args:
            frames: Original frames
            objects_to_mask: List of objects to mask/hide
            objects_to_preserve: Optional list of objects to preserve (their pixels
                won't be blacked out). Only used when preserve_overlapping=True.

        Returns:
            Modified frames with specified objects masked
        """
        result_frames = [f.copy() for f in frames]

        for obj in objects_to_mask:
            result_frames = self.mask_object(
                result_frames,
                obj,
                preserve_objects=objects_to_preserve if self.preserve_overlapping else None
            )

        return result_frames


class VideoInpaintManipulator(VideoManipulator):
    """Inpaint objects using OpenCV"""

    def __init__(self, inpaint_radius: int = 5):
        """
        Args:
            inpaint_radius: Radius of inpainting neighborhood
        """
        self.inpaint_radius = inpaint_radius

    def mask_object(self,
                   frames: List[np.ndarray],
                   tracked_object: TrackedObject) -> List[np.ndarray]:
        """Inpaint object in all frames"""
        result = [f.copy() for f in frames]

        for frame_idx, frame in enumerate(result):
            if frame_idx not in tracked_object.frames:
                continue

            obj_frame = tracked_object.frames[frame_idx]
            mask = obj_frame.mask

            # Ensure mask matches frame size
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

            # Dilate mask slightly for better inpainting
            kernel = np.ones((5, 5), np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)

            # Inpaint
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            inpainted = cv2.inpaint(frame_bgr, mask_dilated, self.inpaint_radius, cv2.INPAINT_TELEA)
            result[frame_idx] = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

        return result


# =============================================================================
# VideoSHAP Main Class
# =============================================================================

class VideoSHAP(BaseSHAP):
    """
    SHAP-based interpretability for video analysis with VLMs.

    Computes Shapley values for tracked objects to explain which objects
    in a video are most important for the model's response.

    Example with blur masking:
        ```python
        video_shap = VideoSHAP(
            model=GeminiVideoModel("gemini-2.0-flash"),
            segmentation_model=SAM3VideoSegmentationModel(),
            manipulator=VideoBlurManipulator(),
            vectorizer=OpenAIEmbeddings(api_key="...")
        )

        results_df, shapley_values = video_shap.analyze(
            video_path="crash.mp4",
            prompt="Which vehicle causes the accident?",
            text_prompts=["car", "truck", "motorcycle"],
            sampling_ratio=0.5
        )

        video_shap.visualize(output_path="explanation.mp4")
        ```

    Example with bounding box masking and segmentation reconstruction:
        ```python
        # Use bbox masking while preserving overlapping objects' pixels
        # This is the same approach as PixelSHAP's BlackoutSegmentationManipulator
        video_shap = VideoSHAP(
            model=GeminiVideoModel("gemini-2.0-flash"),
            segmentation_model=SAM3VideoSegmentationModel(),
            manipulator=VideoBlackoutManipulator(
                mask_type='bbox',              # Use bounding box for masking
                preserve_overlapping=True,     # Restore pixels of other objects
                expansion_pixels=2             # Slight bbox expansion
            ),
            vectorizer=OpenAIEmbeddings(api_key="...")
        )

        # When masking object A with bbox mode + preserve_overlapping:
        # - Object A's bounding box region is blacked out
        # - But pixels belonging to other objects (via their segmentation masks)
        #   are restored/preserved, even if they intersect with A's bbox
        ```
    """

    def __init__(self,
                 model: VideoModelBase,
                 segmentation_model: BaseVideoSegmentationModel,
                 manipulator: VideoManipulator,
                 vectorizer: TextVectorizer,
                 debug: bool = False,
                 temp_dir: str = 'temp_videos'):
        """
        Initialize VideoSHAP.

        Args:
            model: Video-capable VLM for generating responses
            segmentation_model: Model for tracking objects (SAM3)
            manipulator: Strategy for masking objects
            vectorizer: Text vectorizer for similarity calculation
            debug: Enable debug output
            temp_dir: Directory for storing manipulated videos (for debugging)
        """
        super().__init__(model, vectorizer, debug)
        self.segmentation_model = segmentation_model
        self.manipulator = manipulator
        self.temp_dir = temp_dir

        # Analysis state
        self.tracking_result: Optional[VideoTrackingResult] = None
        self.video_path: Optional[str] = None
        self.prompt: Optional[str] = None
        self.baseline_response: Optional[str] = None
        self._temp_dir_path: Optional[str] = None

    def analyze(self,
               video_path: str,
               prompt: str,
               text_prompts: List[str] = ["person", "vehicle", "object"],
               target_fps: float = 8,
               resolution: Optional[Tuple[int, int]] = None,
               sampling_ratio: float = 0.5,
               max_combinations: int = 100,
               confidence_threshold: float = 0.3,
               cleanup_temp_files: bool = True) -> Tuple[Any, Dict[str, float]]:
        """
        Analyze video to compute object importance using SHAP.

        Args:
            video_path: Path to input video
            prompt: Question/prompt about the video
            text_prompts: Object classes to detect and track
            target_fps: FPS for SAM3 tracking
            resolution: Optional resize resolution
            sampling_ratio: Ratio of combinations to sample (0-1)
            max_combinations: Maximum combinations to test
            confidence_threshold: Minimum detection confidence
            cleanup_temp_files: If True, delete temporary manipulated videos after analysis.
                               Set to False to keep videos for debugging.

        Returns:
            Tuple of (results_df, shapley_values dict)
        """
        self.video_path = video_path
        self.prompt = prompt

        # Create temp directory for masked videos
        os.makedirs(self.temp_dir, exist_ok=True)
        self._temp_dir_path = self.temp_dir

        try:
            # Step 1: Track objects with SAM3
            self._debug_print(f"Tracking objects in video...")
            self.tracking_result = self.segmentation_model.track(
                video_path=video_path,
                text_prompts=text_prompts,
                target_fps=target_fps,
                resolution=resolution,
                confidence_threshold=confidence_threshold
            )

            if self.tracking_result.num_objects == 0:
                raise ValueError("No objects detected in video")

            self._debug_print(f"Tracked {self.tracking_result.num_objects} objects")

            # Step 2: Generate baseline response
            self._debug_print(f"Generating baseline response...")
            self.baseline_response = self._calculate_baseline(video_path)
            self._debug_print(f"Baseline: {self.baseline_response[:200]}...")

            # Step 3: Get responses for combinations
            self._debug_print(f"Testing object combinations...")
            responses = self._get_result_per_combination(
                content=self.tracking_result,
                sampling_ratio=sampling_ratio,
                max_combinations=max_combinations
            )

            # Step 4: Calculate similarities and SHAP values
            self._debug_print(f"Computing SHAP values...")
            self.results_df = self._get_df_per_combination(responses, self.baseline_response)
            self.shapley_values = self._calculate_shapley_values(
                self.results_df, self.tracking_result
            )

            return self.results_df, self.shapley_values

        finally:
            # Cleanup temp files if requested
            if cleanup_temp_files and self._temp_dir_path and os.path.exists(self._temp_dir_path):
                for file in os.listdir(self._temp_dir_path):
                    if file.startswith('masked_'):
                        try:
                            os.remove(os.path.join(self._temp_dir_path, file))
                        except Exception:
                            pass

    # === Abstract method implementations ===

    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare args for baseline generation"""
        return {
            "prompt": self.prompt,
            "video_path": content if isinstance(content, str) else None,
            "frames": content.video_frames if isinstance(content, VideoTrackingResult) else None
        }

    def _calculate_baseline(self, content: Any, **kwargs) -> str:
        """Generate baseline response with all objects visible"""
        if isinstance(content, str):
            # Video path
            return self.model.generate_from_video(prompt=self.prompt, video_path=content)
        else:
            # VideoTrackingResult
            return self.model.generate_from_video(prompt=self.prompt, frames=content.video_frames)

    def _get_samples(self, content: Any) -> List[Any]:
        """Get tracked objects as samples"""
        if isinstance(content, VideoTrackingResult):
            return list(content.objects.values())
        return []

    def _prepare_combination_args(self, combination: List[Any], original_content: Any) -> Dict:
        """Prepare masked video for a combination of objects"""
        tracking_result = original_content

        # Determine which objects to mask (those NOT in combination)
        # Use object IDs for comparison since TrackedObject is not hashable
        combo_ids = {obj.object_id for obj in combination}
        objects_to_mask = [obj for obj in tracking_result.objects.values()
                          if obj.object_id not in combo_ids]

        # Objects to preserve are those in the combination (they should remain visible)
        # This is used when manipulator has preserve_overlapping=True
        objects_to_preserve = list(combination)

        # Create masked video
        # Pass objects_to_preserve for manipulators that support it (e.g., VideoBlackoutManipulator)
        if hasattr(self.manipulator, 'preserve_overlapping') and self.manipulator.preserve_overlapping:
            masked_frames = self.manipulator.create_masked_video(
                tracking_result.video_frames,
                objects_to_mask,
                objects_to_preserve=objects_to_preserve
            )
        else:
            masked_frames = self.manipulator.create_masked_video(
                tracking_result.video_frames,
                objects_to_mask
            )

        # Save to temp file with descriptive name
        visible_ids = ','.join([str(obj.object_id) for obj in combination])
        temp_path = os.path.join(self._temp_dir_path, f"masked_{visible_ids}.mp4")
        self._save_video(masked_frames, temp_path, tracking_result.fps)

        return {
            "prompt": self.prompt,
            "video_path": temp_path
        }

    def _get_combination_key(self, combination: List[Any], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        labels = "_".join([obj.label for obj in combination])
        return f"{labels}_{indexes}"

    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: float):
        """Save frames as video file"""
        imageio.mimwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264'
        )

    # === Visualization Methods ===

    def visualize(self,
                 output_path: Optional[str] = None,
                 show_colorbar: bool = True,
                 heatmap_opacity: float = 0.5,
                 thickness: int = 2,
                 roughness: int = 2,
                 background_opacity: float = 0.3) -> str:
        """
        Create visualization video with heatmap overlay (PixelSHAP style).

        Args:
            output_path: Path to save output video
            show_colorbar: Whether to show colorbar legend
            heatmap_opacity: Opacity of heatmap overlay (0-1)
            thickness: Border thickness for sketch effect
            roughness: Roughness for sketch effect
            background_opacity: Opacity of non-object areas (0=black, 1=original)

        Returns:
            Path to output video
        """
        if self.tracking_result is None or self.shapley_values is None:
            raise ValueError("Must run analyze() first")

        if output_path is None:
            output_path = self.video_path.replace('.mp4', '_videoshap.mp4')

        visualizer = VideoSHAPVisualizer()
        return visualizer.create_heatmap_video(
            frames=self.tracking_result.video_frames,
            objects=self.tracking_result.objects,
            shapley_values=self.shapley_values,
            fps=self.tracking_result.fps,
            output_path=output_path,
            show_colorbar=show_colorbar,
            heatmap_opacity=heatmap_opacity,
            baseline_response=self.baseline_response,
            prompt=self.prompt,
            thickness=thickness,
            roughness=roughness,
            background_opacity=background_opacity
        )

    def visualize_highlight_only(self,
                                output_path: Optional[str] = None,
                                top_k: int = 3,
                                dim_factor: float = 0.3) -> str:
        """
        Create video highlighting only important objects.

        Args:
            output_path: Path to save output video
            top_k: Number of top objects to highlight
            dim_factor: How much to dim unimportant objects (0-1)

        Returns:
            Path to output video
        """
        if self.tracking_result is None or self.shapley_values is None:
            raise ValueError("Must run analyze() first")

        if output_path is None:
            output_path = self.video_path.replace('.mp4', '_highlight.mp4')

        visualizer = VideoSHAPVisualizer()
        return visualizer.create_highlight_video(
            frames=self.tracking_result.video_frames,
            objects=self.tracking_result.objects,
            shapley_values=self.shapley_values,
            fps=self.tracking_result.fps,
            output_path=output_path,
            top_k=top_k,
            dim_factor=dim_factor
        )

    def create_side_by_side_gif(self,
                                output_path: str,
                                heatmap_opacity: float = 0.5,
                                background_opacity: float = 0.3,
                                thickness: int = 2,
                                roughness: int = 2) -> str:
        """
        Create a side-by-side GIF showing original video and visualization.

        This method ensures perfect frame synchronization by using the same
        frames for both sides.

        Args:
            output_path: Path to save output GIF
            heatmap_opacity: Opacity of heatmap overlay (0-1)
            background_opacity: Opacity of non-object areas
            thickness: Border thickness for sketch effect
            roughness: Roughness for sketch effect

        Returns:
            Path to output GIF
        """
        if self.tracking_result is None or self.shapley_values is None:
            raise ValueError("Must run analyze() first")

        from token_shap.visualization import create_side_by_side_visualization

        # Create visualization frames using VideoSHAPVisualizer
        visualizer = VideoSHAPVisualizer()
        viz_frames = visualizer.create_heatmap_frames(
            frames=self.tracking_result.video_frames,
            objects=self.tracking_result.objects,
            shapley_values=self.shapley_values,
            heatmap_opacity=heatmap_opacity,
            background_opacity=background_opacity,
            thickness=thickness,
            roughness=roughness,
        )

        # Get importance range for colorbar
        values = list(self.shapley_values.values())
        min_importance = min(values)
        max_importance = max(values)

        return create_side_by_side_visualization(
            original_video_path=None,
            visualization_video_path=None,
            output_path=output_path,
            prompt=self.prompt,
            model_output=self.baseline_response,
            fps=int(self.tracking_result.fps),
            min_importance=min_importance,
            max_importance=max_importance,
            original_frames=self.tracking_result.video_frames,
            visualization_frames=viz_frames,
        )

    def plot_importance_ranking(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot bar chart of object importance.

        Args:
            figsize: Figure size
        """
        if self.shapley_values is None:
            raise ValueError("Must run analyze() first")

        import matplotlib.pyplot as plt
        from matplotlib import cm

        # Sort by importance
        sorted_items = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)

        labels = [item[0].rsplit('_', 1)[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # Create bar chart
        fig, ax = plt.subplots(figsize=figsize)

        # Color by importance
        cmap = plt.colormaps['RdYlGn_r']
        colors = [cmap(v) for v in values]

        bars = ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()

        ax.set_xlabel('Shapley Value (Importance)')
        ax.set_title('VideoSHAP Object Importance Ranking')

        plt.tight_layout()
        plt.show()

        return fig


# =============================================================================
# Video SHAP Visualizer
# =============================================================================

class VideoSHAPVisualizer:
    """Visualization tools for VideoSHAP analysis"""

    def create_sketch_border(self, mask: np.ndarray, thickness: int = 2, roughness: int = 2) -> np.ndarray:
        """
        Creates a sketch-like binary border mask around the object (PixelSHAP style).

        Args:
            mask: Binary mask of the object
            thickness: Border thickness
            roughness: Roughness for sketch effect

        Returns:
            Binary border mask
        """
        borders = np.zeros_like(mask, dtype=bool)
        for i in range(roughness):
            kernel_size = 3 + 2 * (i % 2)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=thickness)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            border = dilated.astype(bool) & ~eroded.astype(bool)
            borders |= border
        return borders

    def create_heatmap_video(self,
                            frames: List[np.ndarray],
                            objects: Dict[int, TrackedObject],
                            shapley_values: Dict[str, float],
                            fps: float,
                            output_path: str,
                            show_colorbar: bool = True,
                            heatmap_opacity: float = 0.5,
                            baseline_response: Optional[str] = None,
                            prompt: Optional[str] = None,
                            thickness: int = 2,
                            roughness: int = 2,
                            background_opacity: float = 0.3) -> str:
        """
        Create video with heatmap overlay showing object importance (PixelSHAP style).

        Args:
            frames: Original video frames
            objects: Tracked objects
            shapley_values: SHAP values for each object
            fps: Output FPS
            output_path: Output path
            show_colorbar: Show colorbar
            heatmap_opacity: Overlay opacity for heatmap colors
            baseline_response: Model response to show
            prompt: Prompt to show
            thickness: Border thickness for sketch effect
            roughness: Roughness for sketch effect
            background_opacity: Opacity of non-object areas (0=black, 1=original)

        Returns:
            Path to output video
        """
        from matplotlib import cm

        # Map object IDs to importance values
        # IMPORTANT: Shapley keys use enumerate index (1-based position in objects list),
        # NOT the object_id. We must use the same ordering as _get_samples().
        obj_importance = {}
        objects_list = list(objects.values())  # Same order as _get_samples()
        for idx, obj in enumerate(objects_list, start=1):
            key = f"{obj.label}_{idx}"
            obj_importance[obj.object_id] = shapley_values.get(key, 0.0)

        # Normalize values for colormap
        values = list(obj_importance.values())
        if len(values) == 0:
            min_val, max_val = 0.0, 1.0
        else:
            min_val, max_val = min(values), max(values)

        import matplotlib.pyplot as plt
        cmap = plt.colormaps['RdYlGn_r']  # Red = important, Green = not

        output_frames = []

        for frame_idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            result = np.zeros_like(frame, dtype=np.float32)

            # Track all object masks for this frame
            all_objects_mask = np.zeros((h, w), dtype=bool)
            all_borders = np.zeros((h, w), dtype=bool)

            for obj_id, obj in objects.items():
                if frame_idx not in obj.frames:
                    continue

                obj_frame = obj.frames[frame_idx]
                mask = obj_frame.mask

                # Resize mask if needed
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

                mask_bool = mask > 0
                all_objects_mask |= mask_bool

                # Get color from importance
                importance = obj_importance[obj_id]
                norm_importance = (importance - min_val) / (max_val - min_val + 1e-8)
                color = np.array(cmap(norm_importance)[:3]) * 255

                # Apply heatmap color blended with original
                result[mask_bool] = (
                    frame[mask_bool].astype(np.float32) * (1 - heatmap_opacity) +
                    color * heatmap_opacity
                )

                # Create sketch border for this object
                border = self.create_sketch_border(mask_bool.astype(np.uint8), thickness, roughness)
                all_borders |= border

            # Apply background: blend with light gray (like PixelSHAP's alpha effect)
            # background_opacity=0.3 means 30% original + 70% light gray
            background_mask = ~all_objects_mask
            bg_blend_color = 220  # Light gray
            result[background_mask] = (
                frame[background_mask].astype(np.float32) * background_opacity +
                bg_blend_color * (1 - background_opacity)
            )

            # Apply black sketch borders on top
            result[all_borders] = [0, 0, 0]

            result = np.clip(result, 0, 255).astype(np.uint8)

            # Add colorbar to all frames
            if show_colorbar:
                result = self._add_colorbar(result, min_val, max_val, cmap)

            output_frames.append(result)

        # Save video
        imageio.mimwrite(
            output_path,
            output_frames,
            fps=fps,
            codec='libx264'
        )

        return output_path

    def create_heatmap_frames(self,
                              frames: List[np.ndarray],
                              objects: Dict[int, TrackedObject],
                              shapley_values: Dict[str, float],
                              heatmap_opacity: float = 0.5,
                              background_opacity: float = 0.3,
                              thickness: int = 2,
                              roughness: int = 2) -> List[np.ndarray]:
        """
        Create heatmap frames showing object importance (without saving).

        Args:
            frames: Original video frames
            objects: Tracked objects
            shapley_values: SHAP values for each object
            heatmap_opacity: Overlay opacity for heatmap colors
            background_opacity: Opacity of non-object areas
            thickness: Border thickness for sketch effect
            roughness: Roughness for sketch effect

        Returns:
            List of visualization frames
        """
        from matplotlib import cm
        import matplotlib.pyplot as plt

        # Map object IDs to importance values
        obj_importance = {}
        objects_list = list(objects.values())
        for idx, obj in enumerate(objects_list, start=1):
            key = f"{obj.label}_{idx}"
            obj_importance[obj.object_id] = shapley_values.get(key, 0.0)

        # Normalize values for colormap
        values = list(obj_importance.values())
        if len(values) == 0:
            min_val, max_val = 0.0, 1.0
        else:
            min_val, max_val = min(values), max(values)

        cmap = plt.colormaps['RdYlGn_r']

        output_frames = []

        for frame_idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            result = np.zeros_like(frame, dtype=np.float32)

            all_objects_mask = np.zeros((h, w), dtype=bool)
            all_borders = np.zeros((h, w), dtype=bool)

            for obj_id, obj in objects.items():
                if frame_idx not in obj.frames:
                    continue

                obj_frame = obj.frames[frame_idx]
                mask = obj_frame.mask

                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

                mask_bool = mask > 0
                all_objects_mask |= mask_bool

                importance = obj_importance[obj_id]
                norm_importance = (importance - min_val) / (max_val - min_val + 1e-8)
                color = np.array(cmap(norm_importance)[:3]) * 255

                result[mask_bool] = (
                    frame[mask_bool].astype(np.float32) * (1 - heatmap_opacity) +
                    color * heatmap_opacity
                )

                border = self.create_sketch_border(mask_bool.astype(np.uint8), thickness, roughness)
                all_borders |= border

            background_mask = ~all_objects_mask
            bg_blend_color = 220
            result[background_mask] = (
                frame[background_mask].astype(np.float32) * background_opacity +
                bg_blend_color * (1 - background_opacity)
            )

            result[all_borders] = [0, 0, 0]
            result = np.clip(result, 0, 255).astype(np.uint8)

            output_frames.append(result)

        return output_frames

    def create_highlight_video(self,
                              frames: List[np.ndarray],
                              objects: Dict[int, TrackedObject],
                              shapley_values: Dict[str, float],
                              fps: float,
                              output_path: str,
                              top_k: int = 3,
                              dim_factor: float = 0.3) -> str:
        """
        Create video highlighting only important objects.

        Args:
            frames: Original video frames
            objects: Tracked objects
            shapley_values: SHAP values
            fps: Output FPS
            output_path: Output path
            top_k: Number of top objects to highlight
            dim_factor: Dimming factor for unimportant regions

        Returns:
            Path to output video
        """
        # Get top-k important objects
        # IMPORTANT: Shapley keys use enumerate index (1-based position in objects list),
        # NOT the object_id. We must use the same ordering as _get_samples().
        obj_importance = {}
        objects_list = list(objects.values())  # Same order as _get_samples()
        for idx, obj in enumerate(objects_list, start=1):
            key = f"{obj.label}_{idx}"
            obj_importance[obj.object_id] = shapley_values.get(key, 0.0)

        sorted_objs = sorted(obj_importance.items(), key=lambda x: x[1], reverse=True)
        top_obj_ids = set([obj_id for obj_id, _ in sorted_objs[:top_k]])

        output_frames = []

        for frame_idx, frame in enumerate(frames):
            result = frame.copy().astype(np.float32)

            # Create mask of important objects
            important_mask = np.zeros(frame.shape[:2], dtype=bool)

            for obj_id in top_obj_ids:
                if obj_id not in objects:
                    continue
                obj = objects[obj_id]
                if frame_idx not in obj.frames:
                    continue

                mask = obj.frames[frame_idx].mask
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                important_mask |= (mask > 0)

            # Dim unimportant regions
            unimportant_mask = ~important_mask
            result[unimportant_mask] *= dim_factor

            result = np.clip(result, 0, 255).astype(np.uint8)
            output_frames.append(result)

        # Save video
        imageio.mimwrite(
            output_path,
            output_frames,
            fps=fps,
            codec='libx264'
        )

        return output_path

    def _add_colorbar(self, frame: np.ndarray, min_val: float, max_val: float, cmap) -> np.ndarray:
        """Add colorbar to frame"""
        h, w = frame.shape[:2]

        # Create colorbar
        bar_width = 20
        bar_height = h // 3
        bar_x = w - bar_width - 20
        bar_y = (h - bar_height) // 2

        for i in range(bar_height):
            norm_val = 1.0 - (i / bar_height)  # Top = high importance
            color = np.array(cmap(norm_val)[:3]) * 255
            frame[bar_y + i, bar_x:bar_x + bar_width] = color.astype(np.uint8)

        # Add labels
        cv2.putText(frame, f"{max_val:.2f}", (bar_x - 10, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"{min_val:.2f}", (bar_x - 10, bar_y + bar_height + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def _add_text(self, frame: np.ndarray, text: str, position: str = "top") -> np.ndarray:
        """Add text overlay to frame"""
        h, w = frame.shape[:2]

        # Create semi-transparent background
        overlay = frame.copy()

        if position == "top":
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
