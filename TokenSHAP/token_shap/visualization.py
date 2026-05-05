import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import colorsys
import cv2
from typing import Dict, List, Optional, Tuple, Union
import os
from matplotlib import cm
from pathlib import Path
import textwrap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb

def visualize_segmentation_results(image_path, boxes, labels, confidences, masks):
    """
    Displays an image with bounding boxes and segmentation masks

    Parameters:
    image_path (str): Path to the original image
    boxes (np.ndarray): Array of bounding boxes in the format [x1, y1, x2, y2]
    labels (list): List of labels for each object
    confidences (list): List of confidence scores for each object
    masks (np.ndarray): Array of binary masks with shape [num_objects, height, width]
    """
    # Load the original image
    image = cv2.imread(str(image_path))  # Convert to string in case it's a Path object
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Create display figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display 1: Image with bounding boxes
    ax1.imshow(image)    
    ax1.axis('off')  # Remove axes from the first display
    
    # Generate unique colors for each object based on HSV (different hue per object)
    num_objects = len(labels)
    colors = [hsv_to_rgb((i / num_objects, 0.8, 0.8)) for i in range(num_objects)]
    
    # Add bounding boxes
    for i, (box, label, confidence, color) in enumerate(zip(boxes, labels, confidences, colors)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # Add label and confidence
        label_text = f"{label}: {confidence:.2f}"
        ax1.text(x1, y1-5, label_text, color=color, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Display 2: Image with segmentation masks
    ax2.imshow(image)
    ax2.set_title('Segmentation Masks', fontsize=14)
    ax2.axis('off')  # Remove axes from the second display
    
    # Create an overlay of colored masks on the image
    color_masks = np.zeros_like(image)
    
    # Add each mask with a unique color
    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Ensure mask has the same dimensions as the image
        if mask.shape != (img_height, img_width):
            # Resize mask if needed
            mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask overlay
        color_mask = np.zeros_like(image)
        for c in range(3):
            color_mask[:, :, c] = mask * (color[c] * 255)
        
        # Blend the mask with the accumulated mask image
        alpha = 0.5  # Mask transparency
        idx = mask > 0
        color_masks[idx] = color_mask[idx] * alpha + color_masks[idx] * (1 - alpha)
    
    # Combine the original image with the color masks
    combined_img = image.copy()
    idx = np.sum(color_masks, axis=2) > 0
    combined_img[idx] = image[idx] * 0.5 + color_masks[idx] * 0.5
    ax2.imshow(combined_img)
    
    # Add legend
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax2.plot([], [], '-', color=color, label=f"{label}")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("segmentation_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

def process_segmentation_results(result):
    boxes = result[0]
    labels = result[1]
    confidences = result[2]
    masks = result[3]
    return boxes, labels, confidences, masks

class PixelSHAPVisualizer:
    """Visualization tools for PixelSHAP analysis"""
        
    def plot_importance_ranking(self,
                                shapley_values: Dict[str, float],
                                image: np.ndarray,
                                masks: np.ndarray,
                                figsize: Tuple[int, int] = None,
                                title: str = 'PixelSHAP Importance',
                                show_values: bool = True,
                                color_range: str = 'RdYlGn_r',
                                thumbnail_size: int = 40,
                                bar_height: float = 0.6,
                                frame_color: List[int] = [0, 0, 0],
                                max_rows: int = 10, 
                                bar_linewidth: float = 2.0):
        """
        Plot horizontal bar chart of object importance ranking with color gradient and object thumbnails
        
        Args:
            shapley_values: Dictionary where keys are labels and values are importance values
            image: Original image array (RGB)
            masks: Segmentation masks for each object
            figsize: Figure size (width, height)
            title: Chart title
            show_values: Whether to show the numeric values on bars
            color_range: Colormap to use for gradient (default: red-yellow-green reversed)
            thumbnail_size: Size of the thumbnail images and frames (in pixels)
            bar_height: Height of each bar (between 0 and 1)
            frame_color: RGB color for the thumbnail frames [R, G, B] values 0-255
            max_rows: Maximum number of rows to display (default: 10)
            bar_linewidth: Width of the bar outline (default: 2.0)
        """
        # Sort items by value in descending order (most important first)
        sorted_items = sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)
        
        # Limit number of items to max_rows
        if max_rows > 0 and len(sorted_items) > max_rows:
            sorted_items = sorted_items[:max_rows]
            
        num_items = len(sorted_items)
        
        # Adjust the figure size based on the number of items and thumbnail size
        if figsize is None:
            # Calculate appropriate figure height based on number of items
            height_per_item = max(0.7, thumbnail_size / 60)  # Adjust scaling factor
            fig_height = max(6, num_items * height_per_item + 2)  # Add 2 for title and colorbar
            figsize = (10, fig_height)  # Wider figure for better spacing
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        values = [value for _, value in sorted_items]
        
        # Normalize values for coloring
        min_val, max_val = min(values), max(values)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        
        # Get colormap
        cmap = cm.get_cmap(color_range)
        
        # Calculate the proper bar spacing based on thumbnail size
        # Ensure enough space for thumbnails
        bar_spacing = max(0.8, thumbnail_size / 40)
        
        # Create evenly spaced y positions with enough distance to avoid overlap
        # Reverse the order so the highest values appear at the top
        y_pos = np.arange(num_items-1, -1, -1) * (bar_height + bar_spacing)
        
        # Pre-process all thumbnails
        thumbnails = []
        
        # First pass: create all thumbnails to ensure consistent sizing
        for i, (obj_key, _) in enumerate(sorted_items):
            obj_idx = int(obj_key.split('_')[-1])
            
            if obj_idx >= len(masks):
                thumbnails.append(None)
                continue
                
            mask = masks[obj_idx]
            
            if not mask.any():  # Skip empty masks
                thumbnails.append(None)
                continue
                
            # Find the bounding box of the object
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                thumbnails.append(None)
                continue
                
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            
            # Extract the object
            obj_height = y_max - y_min + 1
            obj_width = x_max - x_min + 1
            
            # Create a transparent background image with exact dimensions
            object_img = np.zeros((obj_height, obj_width, 4), dtype=np.uint8)
            
            # Get the object mask within the bounding box
            local_mask = mask[y_min:y_max+1, x_min:x_max+1]
            
            # Set RGB values from original image
            for c in range(3):  # RGB channels
                channel_values = image[y_min:y_max+1, x_min:x_max+1, c]
                object_img[:, :, c][local_mask] = channel_values[local_mask]
            
            # Set alpha channel
            object_img[:, :, 3][local_mask] = 255
            
            # Create a fixed-size framed thumbnail
            thumbnail = self._create_framed_thumbnail(object_img, thumbnail_size, frame_color)
            thumbnails.append(thumbnail)
        
        # First, create white background bars for all rows (similar to the screenshot)
        max_value = max(values)
        background_bars = ax.barh(
            y_pos,
            [max_value] * num_items,  # All bars have the same length
            height=bar_height,
            color='white',  # White background
            edgecolor='gray',  # Light gray border
            linewidth=bar_linewidth  # Thicker outline as requested
        )
        
        # Then create colored bars with actual values on top of background bars
        bars = ax.barh(
            y_pos, 
            values, 
            height=bar_height,
            color=[cmap(norm(value)) for value in values],
            edgecolor='none'  # No border for the colored portions
        )
        
        # Add value labels on bars if requested
        if show_values:
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(
                    value + max_value * 0.02,  # Small offset from end of bar
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}',
                    va='center',
                    ha='left',
                    color='black',
                    fontweight='bold'
                )
        
        # Add thumbnails as images directly to the axis
        for i, (thumb, y) in enumerate(zip(thumbnails, y_pos)):
            if thumb is not None:
                # Create OffsetImage with the exact size we want
                im = OffsetImage(thumb, zoom=1.0)
                im.image.axes = ax
                
                # Reduce the offset to make thumbnails closer to the bars
                # Changed from 1.2 to 0.8 to reduce the space
                offset = thumbnail_size * 0.8
                
                # Position as AnnotationBbox with fixed offset to prevent overlap
                ab = AnnotationBbox(
                    im, 
                    (0, y),  # Anchor at y-axis at the correct y-position
                    xybox=(-offset, 0),  # Offset to the left by a reduced amount
                    xycoords=('axes fraction', 'data'),  # y-axis (x=0) and data y-coordinate
                    boxcoords=("offset points", "offset points"),  # Use offset in points
                    box_alignment=(1.0, 0.5),  # Right-align box to anchor point, vertically centered
                    pad=0.3,
                    frameon=False  # No frame for annotation box (we have our own frame)
                )
                ax.add_artist(ab)
        
        # Remove y-tick labels (replaced by thumbnails)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['' for _ in range(len(y_pos))])
        
        # Set y limits with some padding to ensure all bars and thumbnails are visible
        y_max = max(y_pos) + bar_height + bar_spacing
        ax.set_ylim(-bar_spacing, y_max)
        
        # Add some space at the left for thumbnails, but less than before
        # Calculate needed margin based on thumbnail size with smaller factor
        left_margin = 0.20 + (thumbnail_size / 500)  # Reduced from 0.25 to 0.20 and larger divisor
        plt.subplots_adjust(left=left_margin)
        
        # Add title
        plt.title(title)
        
        # Remove grid lines to match the screenshot
        plt.grid(False)
        
        # Add colorbar to show the mapping
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Create a new axis for the colorbar
        cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Importance Value')
        
        # Make it look nice
        plt.tight_layout()
        
        # Adjust layout to accommodate the colorbar at the bottom
        plt.subplots_adjust(bottom=0.15, left=left_margin)
        
        # Return the items in the original order (already sorted by importance)
        return sorted_items
        
    def _create_framed_thumbnail(self, img, size, frame_color=[0, 0, 0]):
        """
        Create a fixed-size thumbnail with a frame around it
        
        Args:
            img: Source image (RGBA)
            size: Size of output thumbnail (square)
            frame_color: RGB color for frame [R, G, B]
            
        Returns:
            Fixed-size framed thumbnail
        """
        # Parameters
        border_size = max(1, size // 20)     # Border thickness in pixels (5% of size)
        inner_padding = max(1, size // 10)   # Padding between object and frame (10% of size)
        
        # Create empty square canvas with alpha=0
        frame_size = size
        inner_size = frame_size - (2 * border_size)
        content_size = inner_size - (2 * inner_padding)
        
        # Create the frame (specified color with alpha=255)
        frame = np.zeros((frame_size, frame_size, 4), dtype=np.uint8)
        frame[:, :, :3] = frame_color  # RGB channels set to frame color
        frame[:, :, 3] = 255           # Alpha channel fully opaque
        
        # Create the inner transparent area
        inner = np.zeros((inner_size, inner_size, 4), dtype=np.uint8)
        
        # Handle empty or invalid images
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            # Just return the frame with empty center
            frame[border_size:border_size+inner_size, border_size:border_size+inner_size] = inner
            return frame
        
        # Resize the image to fit within the content area while maintaining aspect ratio
        h, w = img.shape[:2]
        
        # Handle single pixel or degenerate images
        if h <= 1 or w <= 1:
            # Just fill center with a color to indicate problem
            inner[:, :, :3] = [255, 0, 0]  # Red to indicate problem
            inner[:, :, 3] = 255
            frame[border_size:border_size+inner_size, border_size:border_size+inner_size] = inner
            return frame
        
        # Calculate scale to fit within content area
        scale = min(content_size / h, content_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Ensure minimum dimensions
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        
        # Ensure we don't upscale tiny objects too much (prevents pixelation)
        if scale > 3:
            # Limit extreme upscaling
            scale = min(scale, 3)
            new_h, new_w = int(h * scale), int(w * scale)
        
        try:
            # Resize using OpenCV with appropriate interpolation
            # Use INTER_AREA for downsampling and INTER_LINEAR for upsampling
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
            
            # Calculate position to center the object in the inner area
            y_offset = (inner_size - new_h) // 2
            x_offset = (inner_size - new_w) // 2
            
            # Make sure offsets are valid
            y_offset = max(0, min(y_offset, inner_size - new_h))
            x_offset = max(0, min(x_offset, inner_size - new_w))
            
            # Place the resized object in the inner area
            inner[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        except Exception as e:
            # In case of any resizing errors, create a solid color fill
            inner[:, :, :3] = [255, 0, 0]  # Red to indicate error
            inner[:, :, 3] = 255
        
        # Place the inner area inside the frame
        frame[border_size:border_size+inner_size, border_size:border_size+inner_size] = inner
        
        return frame

    def create_sketch_border(self, mask: np.ndarray, thickness: int = 1, roughness: int = 1) -> np.ndarray:
        """
        Creates a sketch-like binary border mask around the object
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
    
    def _place_labels_on_image(self, ax, label_info, used_positions, font_size=10):
        """
        Helper method to place labels on objects while avoiding overlaps
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes on which to draw the labels
        label_info : list
            List of (x, y, label, importance) tuples
        used_positions : list
            List to track used positions to avoid overlaps
        font_size : int
            Font size for labels
        """
        for label_x, label_y, label_text, importance in label_info:
            # Check for overlap
            overlap = False
            for used_x, used_y in used_positions:
                if abs(used_x - label_x) < 50 and abs(used_y - label_y) < 30:
                    overlap = True
                    break
            
            if not overlap:
                used_positions.append((label_x, label_y))
                ax.text(
                    label_x, label_y,
                    f"{label_text}: {importance:.2f}",
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.7),
                    ha='center',
                    va='center',
                    fontsize=font_size
                )
    
    def _add_importance_legend(self, ax, shapley_values, labels, border_colors):
        """
        Helper method to add a legend showing object importance ranking
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes on which to place the legend
        shapley_values : Dict[str, float]
            Dictionary of object importance values
        labels : List[str]
            List of object labels
        border_colors : List[tuple]
            List of RGB colors for each object border
        """
        # Sort items by importance
        sorted_items = []
        for i, (obj_key, importance) in enumerate(shapley_values.items()):
            obj_idx = int(obj_key.split('_')[-1])
            if obj_idx < len(labels):
                sorted_items.append((labels[obj_idx], importance, border_colors[obj_idx]))
        
        sorted_items = sorted(sorted_items, key=lambda x: x[1], reverse=True)
        
        # Create patches and labels for legend
        legend_patches = []
        for _, _, color in sorted_items:
            patch = plt.Rectangle((0, 0), 1, 1, fc='none', ec=color, linewidth=2)
            legend_patches.append(patch)
        
        legend_labels = [f"{label}: {value:.2f}" for label, value, _ in sorted_items]
        
        # Add legend to plot
        ax.legend(
            legend_patches,
            legend_labels,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            title="Object Importance"
        )
        
    def _format_model_output(self, model_output, max_width=100):
        """
        Format model output text with proper wrapping
        
        Parameters:
        -----------
        model_output : str
            The model output text to format
        max_width : int
            Maximum width for text wrapping
            
        Returns:
        --------
        str : Formatted and wrapped text
        """
        if not model_output:
            return ""
            
        wrapped_lines = []
        for line in model_output.split('\n'):
            wrapped_lines.extend(textwrap.wrap(line, width=max_width))
        
        return '\n'.join(wrapped_lines)
    
    def _add_model_output_box(self, fig, ax, model_output, prompt=None, width_reduction=0.2):
        """
        Add a model output text box with customizable width
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes on which to place the model output
        model_output : str
            The model output text to display
        prompt : Optional[str]
            Optional prompt text
        width_reduction : float
            How much to reduce the width (0.1 = 10% reduction)
        """
        if not model_output:
            return
            
        # Format the model output
        formatted_output = self._format_model_output(model_output)
        
        # Get the position of the main axes
        pos = ax.get_position()
        
        # Calculate reduced width
        reduced_width = pos.width * (1 - width_reduction)
        
        # Create a new axes for the text box - with reduced width
        text_height = 0.2  # Height of the text box
        bottom_margin = 0.05  # Margin from bottom of figure
        
        # Calculate position for the text box with centered reduced width
        text_ax = fig.add_axes([
            pos.x0 + (pos.width * width_reduction / 2),   # Center the reduced width box
            bottom_margin,                                # Position at bottom of figure with margin
            reduced_width,                                # Reduced width
            text_height                                   # Fixed height for text box
        ])
        
        # Hide the axes elements
        text_ax.axis('off')
        
        # Add a rectangle as background
        rect = plt.Rectangle(
            (0, 0), 1, 1,                  # Cover full area
            transform=text_ax.transAxes,   # Use axes coordinates
            facecolor='white',             # White background
            alpha=0.9,                     # Slightly transparent
            edgecolor='gray',              # Gray border
            linewidth=1,                   # Border width
            zorder=0                       # Place behind text
        )
        text_ax.add_patch(rect)
        
        # Add the header and model output text
        text_ax.text(
            0.5, 0.9,                      # Positioned at top center
            r"$\bf{Model\ Output:}$",      # Bold header
            transform=text_ax.transAxes,   # Use axes coordinates
            ha='center', va='top',         # Alignment
            fontsize=10,                   # Font size
            zorder=1                       # Place in front of background
        )
        
        text_ax.text(
            0.5, 0.75,                     # Positioned below header
            formatted_output,              # The formatted text
            transform=text_ax.transAxes,   # Use axes coordinates
            ha='center', va='top',         # Alignment
            fontsize=9,                    # Font size
            linespacing=1.3,               # Line spacing
            zorder=1                       # Place in front of background
        )
        
        # Adjust figure layout to accommodate the text box
        fig.subplots_adjust(bottom=bottom_margin + text_height + 0.05)
        
        return text_ax
    
    def plot_importance_transparent(self,
                                   image_path: Union[str, Path],
                                   masks: np.ndarray,
                                   shapley_values: Dict[str, float],
                                   labels: List[str],
                                   output_path: Optional[str] = None,
                                   show_labels: bool = False,
                                   heatmap_style: bool = True,
                                   show_colorbar: bool = False,
                                   overlay_original: bool = False,
                                   original_opacity: float = 0.3,
                                   heatmap_opacity: float = 0.7,
                                   brightness_exponent: float = 3.0,
                                   figsize: Tuple[int, int] = (8, 8),
                                   background_opacity: float = 0.2,
                                   thickness=1,
                                   roughness=2,
                                   prompt: Optional[str] = None,
                                   show_original_side_by_side: bool = False,
                                   label_font_size: int = 10,
                                   show_legend: bool = False,
                                   model_output: Optional[str] = None):
        """
        Plot image with object importance visualization and save a transparent version.
        
        Parameters:
        -----------
        image_path : Union[str, Path]
            Path to the original image.
        masks : np.ndarray
            Segmentation masks for objects.
        shapley_values : Dict[str, float]
            Dictionary of object importance values.
        labels : List[str]
            List of object labels.
        output_path : Optional[str], optional (default=None)
            Optional path to save visualization.
        show_labels : bool, optional (default=False)
            Whether to show object labels on the visualization.
        heatmap_style : bool, optional (default=True)
            Whether to use heatmap coloring.
        show_colorbar : bool, optional (default=False)
            Whether to show colorbar for heatmap.
        overlay_original : bool, optional (default=False)
            Whether to overlay original image with heatmap.
        original_opacity : float, optional (default=0.3)
            Opacity of the original image when overlaid.
        heatmap_opacity : float, optional (default=0.7)
            Opacity of the heatmap colors.
        brightness_exponent : float, optional (default=3.0)
            Exponent for brightness scaling.
        figsize : Tuple[int, int], optional (default=(8, 8))
            Figure size.
        background_opacity : float, optional (default=0.2)
            Controls the visibility of the original image in areas without objects.
            0.0 means completely transparent background, 1.0 means fully visible original image.
        thickness : int, optional (default=1)
            Thickness of object borders.
        roughness : int, optional (default=2)
            Roughness parameter for border creation.
        prompt : Optional[str], optional (default=None)
            Text to display below the title if provided.
        show_original_side_by_side : bool, optional (default=False)
            If True, displays the original image side by side with the visualization.
        label_font_size : int, optional (default=10)
            Font size for object labels.
        show_legend : bool, optional (default=False)
            Whether to show a legend with object importance ranking.
        model_output : Optional[str], optional (default=None)
            Text output from the model to display alongside the visualization.
        """
        # Load the original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        height, width = image.shape[:2]
        # Initialize with transparent background
        transparent_background = np.zeros((height, width, 4), dtype=np.uint8)
    
        # Normalize Shapley values
        importance_values = np.array(list(shapley_values.values()))
        min_val, max_val = importance_values.min(), importance_values.max()
        normalized_values = {k: (v - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                             for k, v in shapley_values.items()}
    
        # Create heat colormap
        cmap = cm.get_cmap('RdYlGn_r')
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
    
        # Create composite mask for all objects
        all_objects_mask = np.zeros((height, width), dtype=bool)
        
        # Generate unique colors for borders if showing legend
        border_colors = []
        if show_legend:
            for i in range(len(masks)):
                hue = i / len(masks)
                border_colors.append(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        
        # Track label positions to avoid overlap
        used_positions = []
        label_info = []  # To store label information for later placement
        
        # Process each object
        for i, (obj_key, importance) in enumerate(shapley_values.items()):
            obj_idx = int(obj_key.split('_')[-1])
            if obj_idx >= len(masks):
                continue
            
            mask = masks[obj_idx]
            # Ensure mask is boolean for all operations
            if mask.dtype != bool:
                mask = mask.astype(bool)
            all_objects_mask |= mask  # Add to composite mask
            normalized_importance = normalized_values[obj_key]
    
            if heatmap_style:
                color = cmap(norm(importance))
                color_rgb = np.array([int(c * 255) for c in color[:3]])
    
                if overlay_original:
                    obj_pixels = (
                        image[mask].astype(float) * original_opacity +
                        color_rgb * heatmap_opacity
                    )
                else:
                    obj_pixels = color_rgb
    
                transparent_background[mask, :3] = np.clip(obj_pixels, 0, 255).astype(np.uint8)
                transparent_background[mask, 3] = 255
            else:
                brightness = 0.5 + normalized_importance ** brightness_exponent
                brightness = min(brightness, 2.0)
                obj_pixels = image[mask].astype(float) * brightness
    
                if overlay_original:
                    obj_pixels = (
                        image[mask].astype(float) * original_opacity +
                        obj_pixels * (1 - original_opacity)
                    )
    
                transparent_background[mask, :3] = np.clip(obj_pixels, 0, 255).astype(np.uint8)
                transparent_background[mask, 3] = 255
            
            # Collect label information for later placement
            if show_labels and mask.any():
                y_positions, x_positions = np.where(mask)
                label_x = int(np.mean(x_positions))
                label_y = int(np.mean(y_positions))
                label_info.append((label_x, label_y, labels[obj_idx], importance))
                
            # Store border color for legend
            if show_legend:
                transparent_background[self.create_sketch_border(mask), :3] = [int(c * 255) for c in border_colors[i]]
                transparent_background[self.create_sketch_border(mask), 3] = 255
    
        # Apply background with controlled opacity
        if background_opacity > 0:
            # Get the inverted mask (areas with no objects)
            background_mask = ~all_objects_mask
            
            # Apply original image to background areas with specified opacity
            transparent_background[background_mask, :3] = image[background_mask]
            
            # Set alpha channel for background based on background_opacity
            transparent_background[background_mask, 3] = int(255 * background_opacity)
    
        # Create borders for all objects
        if not show_legend:  # only add black borders if not using colored borders for legend
            border = self.create_sketch_border(all_objects_mask, thickness=thickness, roughness=roughness)
            transparent_background[border] = [0, 0, 0, 255]  # Black border
    
        # Save visualization if path provided
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(transparent_background, cv2.COLOR_RGBA2BGRA))
    
        # Determine if we need to display model output
        has_model_output = model_output is not None
        
        # Side-by-side layout with original image
        if show_original_side_by_side:
            if show_colorbar and heatmap_style:
                # Use tight layout and small height ratio for colorbar
                fig = plt.figure(figsize=(figsize[0] * 2, figsize[1] * 1.1))
                
                # Create grid with very small height for colorbar and minimal vertical spacing
                gs = plt.GridSpec(2, 2, 
                                 height_ratios=[1, 0.06],  # Much smaller colorbar height
                                 width_ratios=[1, 1], 
                                 hspace=0.01,  # Minimal vertical spacing
                                 wspace=0.1)
                
                # Original image on left
                ax_orig = fig.add_subplot(gs[0, 0])
                ax_orig.imshow(image)
                ax_orig.axis('off')
                ax_orig.set_title("Original Image")
                
                # Visualization on right
                ax_vis = fig.add_subplot(gs[0, 1])
                ax_vis.imshow(transparent_background)
                ax_vis.axis('off')
                ax_vis.set_title("PixelSHAP Importance Visualization")
                
                # Colorbar under visualization only - thinner and closer to the image
                ax_cbar = fig.add_subplot(gs[1, 1])
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                ax_cbar.imshow(gradient, aspect='auto', cmap=cmap)
                ax_cbar.set_yticks([])
                ax_cbar.set_xticks([0, 128, 255])
                ax_cbar.set_xticklabels([f'{min_val:.2f}', f'{(min_val + max_val)/2:.2f}', f'{max_val:.2f}'])
                ax_cbar.set_xlabel('Importance Value', fontsize=8)  # Smaller font
                
                # Custom positioning for colorbar to make it closer to image and thinner
                plt.tight_layout()  # First apply tight layout
                
                # Then manually adjust colorbar position
                pos = ax_cbar.get_position()
                ax_cbar.set_position([
                    pos.x0 + pos.width * 0.2,    # Center it horizontally
                    pos.y0 + pos.height * 0.8,   # Move it up significantly closer to image
                    pos.width * 0.6,             # Make it 60% of original width
                    pos.height * 0.5             # Make it 50% of original height
                ])
                
                # Keep bottom-left cell empty but hide it
                fig.add_subplot(gs[1, 0]).set_visible(False)
                
                # Add prompt text in bold if provided
                if prompt:
                    ax_orig.text(0.5, -0.05, f"Prompt: {prompt}", 
                               transform=ax_orig.transAxes,
                               ha='center', va='top', 
                               fontsize=10, 
                               fontweight='bold')
                
                # Add model output text below original image if provided
                if has_model_output:
                    formatted_output = self._format_model_output(model_output)
                    # Add more space between prompt and model output
                    y_pos = -0.18 if prompt else -0.05
                    
                    # Display model output with bold header inside the same box
                    output_text = r"$\bf{Model\ Output:}$" + "\n\n" + formatted_output
                    ax_orig.text(0.5, y_pos, 
                               output_text,
                               transform=ax_orig.transAxes,
                               ha='center', va='top',
                               fontsize=9,
                               linespacing=1.2,
                               bbox=dict(
                                   boxstyle='round,pad=0.5',
                                   facecolor='white',
                                   alpha=0.8,
                                   edgecolor='gray'
                               ))
                
                # Place labels on objects if requested
                if show_labels:
                    self._place_labels_on_image(ax_vis, label_info, used_positions, label_font_size)
                
                # Add legend if requested
                if show_legend:
                    self._add_importance_legend(ax_vis, shapley_values, labels, border_colors)
            
            else:
                # Simple side-by-side layout without colorbar
                fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
                
                # Original image on left
                ax_orig = axes[0]
                ax_orig.imshow(image)
                ax_orig.axis('off')
                ax_orig.set_title("Original Image")
                
                # Visualization on right
                ax_vis = axes[1]
                ax_vis.imshow(transparent_background)
                ax_vis.axis('off')
                ax_vis.set_title("PixelSHAP Importance Visualization")
                
                # Add prompt text in bold if provided
                if prompt:
                    ax_orig.text(0.5, -0.05, f"Prompt: {prompt}", 
                               transform=ax_orig.transAxes,
                               ha='center', va='top', 
                               fontsize=10, 
                               fontweight='bold')
                
                # Add model output text with header and content in the same box
                if has_model_output:
                    formatted_output = self._format_model_output(model_output)
                    # Add more space between prompt and model output
                    y_pos = -0.18 if prompt else -0.05
                    
                    # Display model output with bold header inside the same box
                    output_text = r"$\bf{Model\ Output:}$" + "\n\n" + formatted_output
                    ax_orig.text(0.5, y_pos, 
                               output_text,
                               transform=ax_orig.transAxes,
                               ha='center', va='top',
                               fontsize=9,
                               linespacing=1.2,
                               bbox=dict(
                                   boxstyle='round,pad=0.5',
                                   facecolor='white',
                                   alpha=0.8,
                                   edgecolor='gray'
                               ))
                
                # Place labels on objects if requested
                if show_labels:
                    self._place_labels_on_image(ax_vis, label_info, used_positions, label_font_size)
                
                # Add legend if requested
                if show_legend:
                    self._add_importance_legend(ax_vis, shapley_values, labels, border_colors)
        
        else:
            # Single visualization without original image
            if show_colorbar and heatmap_style:
                gs = plt.GridSpec(2, 1, height_ratios=[6, 0.3], hspace=0.1)
                fig = plt.figure(figsize=figsize)
                
                # Visualization
                ax_img = fig.add_subplot(gs[0])
                ax_img.imshow(transparent_background)
                ax_img.axis('off')
                
                # Add title and prompt
                if prompt:
                    ax_img.set_title("PixelSHAP Importance Visualization")
                    ax_img.text(0.5, -0.05, f"Prompt: {prompt}", 
                              transform=ax_img.transAxes,
                              ha='center', va='top', 
                              fontsize=10, 
                              fontweight='bold')
                else:
                    ax_img.set_title("PixelSHAP Importance Visualization")
                
                # Colorbar
                ax_colorbar = fig.add_subplot(gs[1])
                gradient = np.linspace(0, 1, 256).reshape(1, -1)
                ax_colorbar.imshow(gradient, aspect='auto', cmap=cmap)
                ax_colorbar.set_yticks([])
                ax_colorbar.set_xticks([0, 128, 255])
                ax_colorbar.set_xticklabels([f'{min_val:.2f}', f'{(min_val + max_val)/2:.2f}', f'{max_val:.2f}'])
                ax_colorbar.set_xlabel('Importance Value')
                
                # Make colorbar thinner
                pos = ax_colorbar.get_position()
                ax_colorbar.set_position([pos.x0 + pos.width * 0.2, pos.y0, 
                                        pos.width * 0.6, pos.height * 0.5])
                
                # Place labels on objects if requested
                if show_labels:
                    self._place_labels_on_image(ax_img, label_info, used_positions, label_font_size)
                
                # Add legend if requested
                if show_legend:
                    self._add_importance_legend(ax_img, shapley_values, labels, border_colors)
                
                # Add model output with bold header and clear spacing in the same box
                if has_model_output:
                    formatted_output = self._format_model_output(model_output)
                    # Add more space between prompt and model output
                    y_pos = -0.18 if prompt else -0.05
                    
                    # Display model output with bold header inside the same box
                    output_text = r"$\bf{Model\ Output:}$" + "\n\n" + formatted_output
                    ax_img.text(0.5, y_pos, 
                              output_text,
                              transform=ax_img.transAxes,
                              ha='center', va='top',
                              fontsize=9,
                              linespacing=1.2,
                              bbox=dict(
                                  boxstyle='round,pad=0.5',
                                  facecolor='white',
                                  alpha=0.8,
                                  edgecolor='gray'
                              ))
            else:
                # Simple single visualization without colorbar
                fig = plt.figure(figsize=figsize)
                ax = plt.gca()
                ax.imshow(transparent_background)
                ax.axis('off')
                
                # Add title and prompt
                if prompt:
                    ax.set_title("PixelSHAP Importance Visualization")
                    ax.text(0.5, -0.05, f"Prompt: {prompt}", 
                          transform=ax.transAxes,
                          ha='center', va='top', 
                          fontsize=10, 
                          fontweight='bold')
                else:
                    ax.set_title("PixelSHAP Importance Visualization")
                
                # Place labels on objects if requested
                if show_labels:
                    self._place_labels_on_image(ax, label_info, used_positions, label_font_size)
                
                # Add legend if requested
                if show_legend:
                    self._add_importance_legend(ax, shapley_values, labels, border_colors)
                
                # Add model output with bold header and proper spacing in the same box
                if has_model_output:
                    formatted_output = self._format_model_output(model_output)
                    # Add more space between prompt and model output
                    y_pos = -0.15 if prompt else -0.05
                    
                    # Display model output with bold header inside the same box
                    output_text = r"$\bf{Model\ Output:}$" + "\n\n" + formatted_output
                    ax.text(0.5, y_pos, 
                          output_text,
                          transform=ax.transAxes,
                          ha='center', va='top',
                          fontsize=9,
                          linespacing=1.2,
                          bbox=dict(
                              boxstyle='round,pad=0.5',
                              facecolor='white',
                              alpha=0.8,
                              edgecolor='gray'
                          ))
    
        plt.tight_layout()
        
        # Adjust bottom margin if model output is displayed to ensure it's not cut off
        if has_model_output:
            plt.subplots_adjust(bottom=0.2)
            
        plt.show()
        
        return transparent_background  # Return the generated image for further processing if needed

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import textwrap


def create_side_by_side_visualization(
    original_video_path: str,
    visualization_video_path: str,
    output_path: str,
    prompt: str,
    model_output: str,
    fps: int = 8,
    min_importance: float = 0.0,
    max_importance: float = 1.0,
    original_frames: List[np.ndarray] = None,
    visualization_frames: List[np.ndarray] = None,
):
    """
    Create a side-by-side video/GIF with original and visualization.

    Layout:
    ┌─────────────────┬─────────────────┐
    │  Original Image │  Visualization  │
    ├─────────────────┼─────────────────┤
    │ Prompt:         │                 │
    │ Model Output:   │   [Colorbar]    │
    └─────────────────┴─────────────────┘

    Args:
        original_video_path: Path to original video (ignored if original_frames provided)
        visualization_video_path: Path to visualization video (ignored if visualization_frames provided)
        output_path: Output path (.gif or .mp4)
        prompt: Question/prompt text
        model_output: Model's answer text
        fps: Output frames per second
        min_importance: Minimum value for colorbar
        max_importance: Maximum value for colorbar
        original_frames: Optional list of original frames (numpy arrays) - use for perfect sync
        visualization_frames: Optional list of visualization frames (numpy arrays) - use for perfect sync

    Returns:
        Path to output file
    """
    # Use provided frames or read from video files
    if original_frames is not None:
        orig_frames_list = [np.array(f) if not isinstance(f, np.ndarray) else f for f in original_frames]
    else:
        original_reader = imageio.get_reader(original_video_path)
        orig_frames_list = list(original_reader)
        original_reader.close()

    if visualization_frames is not None:
        viz_frames_list = [np.array(f) if not isinstance(f, np.ndarray) else f for f in visualization_frames]
    else:
        viz_reader = imageio.get_reader(visualization_video_path)
        viz_frames_list = list(viz_reader)
        viz_reader.close()

    # CRITICAL: Match frame counts by index
    num_frames = min(len(orig_frames_list), len(viz_frames_list))

    # Get dimensions from first frame
    h, w = orig_frames_list[0].shape[:2]

    # Layout parameters
    padding = 16
    title_height = 28
    text_area_height = 160  # Increased for more text
    colorbar_height = 50

    # Total canvas size
    canvas_width = w * 2 + padding * 3
    canvas_height = title_height + h + padding + text_area_height

    # Create static elements ONCE (prevents flickering/color changes)
    colorbar_img = _create_colorbar_image(
        width=w,
        height=colorbar_height,
        min_val=min_importance,
        max_val=max_importance,
    )

    text_panel = _create_text_panel(
        width=w + padding,
        height=text_area_height,
        prompt=prompt,
        model_output=model_output,
        padding=12,
    )

    # Pre-render the static canvas template (everything except video frames)
    template = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    template_draw = ImageDraw.Draw(template)

    title_font = load_font(None, 13)

    # Draw titles once on template
    orig_title = "Original Image"
    bbox = title_font.getbbox(orig_title)
    orig_title_x = padding + (w - (bbox[2] - bbox[0])) // 2
    template_draw.text((orig_title_x, 8), orig_title, fill=(120, 120, 120), font=title_font)

    viz_title = "VideoSHAP Importance Visualization"
    bbox = title_font.getbbox(viz_title)
    viz_title_x = padding * 2 + w + (w - (bbox[2] - bbox[0])) // 2
    template_draw.text((viz_title_x, 8), viz_title, fill=(120, 120, 120), font=title_font)

    # Paste static elements on template
    content_y = title_height + h + padding // 2
    template.paste(text_panel, (padding, content_y))

    colorbar_x = padding * 2 + w
    colorbar_y = content_y + (text_area_height - colorbar_height) // 2
    template.paste(colorbar_img, (colorbar_x, colorbar_y))

    # Generate frames
    output_frames = []
    frame_y = title_height

    for i in range(num_frames):
        # Copy template for this frame
        canvas = template.copy()

        # Get synchronized frames (same index = same time)
        orig_frame = Image.fromarray(orig_frames_list[i]).convert("RGB")
        viz_frame = Image.fromarray(viz_frames_list[i]).convert("RGB")

        # Resize if needed (should be same size, but just in case)
        if orig_frame.size != (w, h):
            orig_frame = orig_frame.resize((w, h), Image.LANCZOS)
        if viz_frame.size != (w, h):
            viz_frame = viz_frame.resize((w, h), Image.LANCZOS)

        # Paste video frames
        canvas.paste(orig_frame, (padding, frame_y))
        canvas.paste(viz_frame, (padding * 2 + w, frame_y))

        output_frames.append(np.array(canvas))

    # Save output
    if output_path.endswith('.gif'):
        # Use higher quality GIF settings
        imageio.mimsave(
            output_path,
            output_frames,
            fps=fps,
            loop=0,
        )
    else:
        imageio.mimsave(output_path, output_frames, fps=fps)

    return output_path


def _create_colorbar_image(width: int, height: int, min_val: float, max_val: float):
    """Create a horizontal colorbar image with RdYlGn_r colormap."""
    bar_height = height - 22  # Space for labels

    # Create gradient using numpy for smooth colors
    gradient = np.linspace(0, 1, width)
    gradient = np.tile(gradient, (bar_height, 1))

    # Apply RdYlGn_r colormap
    cmap = cm.get_cmap('RdYlGn_r')
    colored = (cmap(gradient)[:, :, :3] * 255).astype(np.uint8)

    # Create PIL image with white background
    img = Image.new("RGB", (width, height), (255, 255, 255))

    # Paste the gradient
    gradient_img = Image.fromarray(colored)
    img.paste(gradient_img, (0, 0))

    # Draw border around colorbar
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width - 1, bar_height - 1)], outline=(180, 180, 180), width=1)

    # Add labels
    font = load_font(None, 11)
    label_y = bar_height + 4

    labels = [
        (4, f"{min_val:.2f}", "left"),
        (width // 2, f"{(min_val + max_val) / 2:.2f}", "center"),
        (width - 4, f"{max_val:.2f}", "right"),
    ]

    for x, label, align in labels:
        bbox = font.getbbox(label)
        label_width = bbox[2] - bbox[0]
        if align == "left":
            draw.text((x, label_y), label, fill=(80, 80, 80), font=font)
        elif align == "right":
            draw.text((x - label_width, label_y), label, fill=(80, 80, 80), font=font)
        else:
            draw.text((x - label_width // 2, label_y), label, fill=(80, 80, 80), font=font)

    return img


def _create_text_panel(width: int, height: int, prompt: str, model_output: str, padding: int = 12):
    """Create text panel with prompt and model output."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Fonts
    bold_font = load_font(None, 12)
    text_font = load_font(None, 11)

    y = padding

    # Prompt section
    draw.text((padding, y), "Prompt:", fill=(80, 80, 80), font=bold_font)
    y += 16

    # Wrap and draw prompt text
    max_chars = max(30, (width - padding * 2) // 7)
    wrapped_prompt = textwrap.wrap(prompt, width=max_chars)
    for line in wrapped_prompt[:2]:
        draw.text((padding, y), line, fill=(50, 50, 50), font=text_font)
        y += 14

    y += 10

    # Model Output box - calculate size based on remaining space
    box_x = padding
    box_y = y
    box_width = width - padding * 2
    box_height = height - y - padding

    # Draw box with subtle styling
    draw.rectangle(
        [(box_x, box_y), (box_x + box_width, box_y + box_height)],
        fill=(248, 249, 250),
        outline=(200, 200, 205),
        width=1,
    )

    # Header
    draw.text((box_x + 10, box_y + 8), "Model Output:", fill=(80, 80, 80), font=bold_font)

    # Model output text - more lines allowed
    inner_width = box_width - 20
    max_chars_output = max(25, inner_width // 7)
    wrapped_output = textwrap.wrap(model_output, width=max_chars_output)

    text_y = box_y + 26
    max_lines = (box_height - 36) // 14  # Calculate how many lines fit

    for i, line in enumerate(wrapped_output[:max_lines]):
        draw.text((box_x + 10, text_y), line, fill=(40, 40, 40), font=text_font)
        text_y += 14

    # Add ellipsis if text was truncated
    if len(wrapped_output) > max_lines:
        draw.text((box_x + 10, text_y - 14), wrapped_output[max_lines - 1].rstrip() + "...", fill=(40, 40, 40), font=text_font)

    return img


def mp4_to_gif_with_styled_title(
    mp4_path,
    gif_path,
    title_text,
    fps=4,
    banner_height=80,
    font_size=32,
    font_path=None,
    padding=20,
    answer_text=None,
    answer_banner_height=None,
    answer_font_size=None,
):
    """
    Convert MP4 to GIF with Q&A banners (PixelSHAP style).

    Args:
        mp4_path: Input video path
        gif_path: Output GIF path
        title_text: Question/prompt text for top banner
        fps: Output FPS
        banner_height: Height of top (question) banner
        font_size: Font size for question
        font_path: Custom font path
        padding: Horizontal padding for text
        answer_text: Model answer for bottom banner (optional)
        answer_banner_height: Height of bottom banner (defaults to banner_height * 1.5)
        answer_font_size: Font size for answer (defaults to font_size * 0.75)

    Returns:
        Path to output GIF
    """
    # Set defaults for answer banner
    if answer_banner_height is None:
        answer_banner_height = int(banner_height * 1.5)
    if answer_font_size is None:
        answer_font_size = int(font_size * 0.75)

    reader = imageio.get_reader(mp4_path)
    frames = []
    first_frame = True
    cached_top_banner = None
    cached_bottom_banner = None

    for frame in reader:
        base = Image.fromarray(frame).convert("RGB")
        w, h = base.size

        # Create banners only once (identical for all frames - no flickering!)
        if first_frame:
            # Top banner for question
            cached_top_banner = create_banner_with_fitted_text(
                width=w,
                height=banner_height,
                title_text=f"Q: {title_text}",
                max_font_size=font_size,
                padding=padding,
                font_path=font_path
            )

            # Bottom banner for answer (if provided)
            if answer_text:
                cached_bottom_banner = create_banner_with_fitted_text(
                    width=w,
                    height=answer_banner_height,
                    title_text=f"A: {answer_text}",
                    max_font_size=answer_font_size,
                    padding=padding,
                    font_path=font_path,
                    is_answer=True
                )
            first_frame = False

        # Calculate total canvas height
        total_height = banner_height + h
        if answer_text:
            total_height += answer_banner_height

        # Create canvas
        canvas = Image.new("RGB", (w, total_height), (255, 255, 255))

        # Paste top banner (question)
        canvas.paste(cached_top_banner, (0, 0))

        # Paste video frame
        canvas.paste(base, (0, banner_height))

        # Paste bottom banner (answer) if provided
        if answer_text and cached_bottom_banner:
            canvas.paste(cached_bottom_banner, (0, banner_height + h))

        frames.append(canvas)

    reader.close()

    # Convert to numpy arrays for imageio
    frame_arrays = [np.array(f) for f in frames]

    # Save with better quality (using pillow for quantization)
    imageio.mimsave(gif_path, frame_arrays, fps=fps, loop=0)

    return gif_path


def create_banner_with_fitted_text(width, height, title_text, max_font_size, padding, font_path=None, is_answer=False):
    """
    Create a banner with text that automatically fits the width.
    Returns a single banner image to be reused for all frames.

    Args:
        width: Banner width
        height: Banner height
        title_text: Text to display
        max_font_size: Maximum font size to try
        padding: Horizontal padding
        font_path: Custom font path
        is_answer: If True, style as answer banner (different background)
    """

    available_width = width - (padding * 2)

    # Find the best font size that fits
    font_size = max_font_size
    font = None
    best_lines = None

    while font_size >= 12:
        font = load_font(font_path, font_size)

        # Try different wrapping widths to find optimal fit
        for chars_per_line in range(100, 10, -5):
            lines = textwrap.wrap(title_text, width=chars_per_line)

            # Check if all lines fit within available width
            fits = True
            for line in lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                if line_width > available_width:
                    fits = False
                    break

            if fits:
                best_lines = lines
                break

        if best_lines:
            # Check if total height fits
            line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
            total_height = len(best_lines) * line_height + (len(best_lines) - 1) * 8

            if total_height <= height - 16:  # Leave vertical padding
                break

        font_size -= 2
        best_lines = None

    if not best_lines:
        best_lines = [title_text[:60] + "..."]  # Fallback
        font = load_font(font_path, 16)

    # Create the banner image with different styling for Q vs A
    banner = Image.new("RGB", (width, height), (250, 250, 252))
    draw = ImageDraw.Draw(banner)

    # Different gradient for question vs answer
    if is_answer:
        # Answer banner: light warm tint
        for y in range(height):
            ratio = y / height
            r = int(252 - ratio * 8)
            g = int(250 - ratio * 10)
            b = int(245 - ratio * 12)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        # Top border line (since answer is at bottom)
        draw.line([(0, 0), (width, 0)], fill=(220, 220, 225), width=1)
    else:
        # Question banner: neutral gradient
        for y in range(height):
            ratio = y / height
            gray = int(252 - ratio * 12)
            draw.line([(0, y), (width, y)], fill=(gray, gray, gray + 2))
        # Bottom border line
        draw.line([(0, height - 1), (width, height - 1)], fill=(220, 220, 225), width=1)

    # Calculate text positioning
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    line_spacing = 6
    total_text_height = len(best_lines) * line_height + (len(best_lines) - 1) * line_spacing
    y = (height - total_text_height) // 2

    # Draw each line centered
    for line in best_lines:
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        x = (width - line_width) // 2

        # Light shadow for depth
        draw.text((x + 1, y + 1), line, fill=(200, 200, 205), font=font)
        # Main text - dark gray
        draw.text((x, y), line, fill=(35, 35, 40), font=font)

        y += line_height + line_spacing

    return banner


def load_font(font_path, size):
    """Load font with fallbacks."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            pass
    
    # Try common font paths
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "Arial.ttf",
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    
    return ImageFont.load_default()