"""Pure CPU primitives for reproducible observed-water mapping."""
from .core import aggregate, classify, confusion, smooth_power, tiled_mean

__all__ = ["aggregate", "classify", "confusion", "smooth_power", "tiled_mean"]
