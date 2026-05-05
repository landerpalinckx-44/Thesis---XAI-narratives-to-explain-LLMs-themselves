# video_utils.py
"""
Video utilities using SAM3 for tracking.

This module is deprecated in favor of video_shap.py which provides
comprehensive VideoSHAP functionality with SAM3-based tracking.

For video tracking, use:
    from token_shap.video_shap import SAM3VideoSegmentationModel

For complete VideoSHAP analysis, use:
    from token_shap.video_shap import VideoSHAP
"""

import warnings

# Re-export from video_shap for backward compatibility
from .video_shap import (
    SAM3VideoSegmentationModel,
    VideoTrackingResult,
    TrackedObject,
    ObjectFrame,
)

# Compatibility alias
Sam3VideoTracker = SAM3VideoSegmentationModel
ChunkedSam3VideoTracker = SAM3VideoSegmentationModel

# Deprecated aliases - these will show warnings
def _deprecated_sam2_warning():
    warnings.warn(
        "SAM2 classes are deprecated. Use SAM3VideoSegmentationModel instead.",
        DeprecationWarning,
        stacklevel=3
    )

class Sam2VideoTracker(SAM3VideoSegmentationModel):
    """Deprecated: Use SAM3VideoSegmentationModel instead"""
    def __init__(self, *args, **kwargs):
        _deprecated_sam2_warning()
        super().__init__(*args, **kwargs)

class ChunkedSam2VideoTracker(SAM3VideoSegmentationModel):
    """Deprecated: Use SAM3VideoSegmentationModel instead"""
    def __init__(self, *args, **kwargs):
        _deprecated_sam2_warning()
        super().__init__(*args, **kwargs)


def create_video_tracker(*args, **kwargs):
    """Create a SAM3 video tracker (backward compatible function)"""
    return SAM3VideoSegmentationModel(*args, **kwargs)
