# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
from .textimage2video import WanTI2V

# Optional dependency: `WanS2V` requires `decord` (and other S2V deps).
# Import it lazily so non-S2V tasks (e.g., T2V/I2V) don't require `decord`.
try:
    from .speech2video import WanS2V  # type: ignore
except ImportError as e:  # pragma: no cover
    _WAN_S2V_IMPORT_ERROR = e

    class WanS2V:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "WanS2V dependencies are not installed. Install S2V requirements and `decord` "
                "(e.g., `pip install -r requirements_s2v.txt` and then install `decord`), "
                "or use a non-S2V task (t2v/i2v/ti2v/animate)."
            ) from _WAN_S2V_IMPORT_ERROR

# Optional dependency: `WanAnimate` requires extra animate dependencies (e.g., `peft`).
try:
    from .animate import WanAnimate  # type: ignore
except ImportError as e:  # pragma: no cover
    _WAN_ANIMATE_IMPORT_ERROR = e

    class WanAnimate:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "WanAnimate dependencies are not installed. Install animate requirements "
                "(e.g., `pip install -r requirements_animate.txt`) or use a non-animate task "
                "(t2v/i2v/ti2v/s2v)."
            ) from _WAN_ANIMATE_IMPORT_ERROR