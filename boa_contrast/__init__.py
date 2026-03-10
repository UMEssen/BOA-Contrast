import importlib.metadata
import logging

from boa_contrast.commands import compute_segmentation, predict

__version__ = importlib.metadata.version("boa_contrast")
logging.basicConfig()
logging.captureWarnings(True)
# warnings.warn() in library code if the issue is avoidable and the client application
# should be modified to eliminate the warning

# logging.warning() if there is nothing the client application can do about the situation,
# but the event should still be noted

logger = logging.getLogger(__name__)
__all__ = [
    "predict",
    "compute_segmentation",
]
