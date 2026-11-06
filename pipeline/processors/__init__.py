from .base import Processor
from .dwell import ZoneDwellProcessor
from .entry_exit import EntryExitProcessor
from .queue import QueueProcessor

__all__ = ["Processor", "ZoneDwellProcessor", "EntryExitProcessor", "QueueProcessor", "build_processor"]


def build_processor(cam_cfg: dict, emitter, **kw) -> Processor:
    kind = cam_cfg.get("processor") or cam_cfg.get("role")
    if kind in ("cam1", "cam2", "shelf", "dwell"):
        return ZoneDwellProcessor(cam_cfg, emitter, **{k: v for k, v in kw.items() if k in ZoneDwellProcessor.OPTIONS})
    if kind in ("entry_exit", "entry"):
        return EntryExitProcessor(cam_cfg, emitter, **{k: v for k, v in kw.items() if k in EntryExitProcessor.OPTIONS})
    if kind in ("queue", "billing"):
        return QueueProcessor(cam_cfg, emitter, **{k: v for k, v in kw.items() if k in QueueProcessor.OPTIONS})
    raise ValueError(f"unknown processor {kind!r}")
