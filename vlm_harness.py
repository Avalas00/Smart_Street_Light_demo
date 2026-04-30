import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from vlm_alert import build_vlm_client


@dataclass
class DetectionSummary:
    person_count: int
    level: int
    level_label: str
    brightness: int
    frame_width: int
    frame_height: int
    boxes: List[dict] = field(default_factory=list)


@dataclass
class HarnessConfig:
    enabled: bool
    min_interval_sec: float
    trigger_people: int
    trigger_level: int
    fallen_aspect_ratio: float
    min_box_height_ratio: float


class VLMHarness:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.client = build_vlm_client()
        self.lock = threading.Lock()
        self.running = False
        self.last_run = 0.0
        self.status = "enabled" if config.enabled else "disabled"

    def begin(self, summary: DetectionSummary) -> Optional[Dict]:
        if not self.config.enabled:
            self.status = "disabled"
            return None

        trigger = self._select_trigger(summary)
        if trigger is None:
            self.status = "idle"
            return None

        now = time.time()
        with self.lock:
            if self.running:
                self.status = "busy"
                return None
            if now - self.last_run < self.config.min_interval_sec:
                self.status = "cooldown"
                return None
            self.running = True
            self.last_run = now
            self.status = "running"
            return trigger

    def inspect(self, image_jpeg: bytes, summary: DetectionSummary, trigger: Dict) -> Dict:
        context = asdict(summary)
        context["trigger"] = trigger
        result = self.client.inspect(image_jpeg, context=context)
        result.setdefault("trigger", trigger)
        return result

    def finish(self) -> None:
        with self.lock:
            self.running = False
            self.status = "idle" if self.config.enabled else "disabled"

    def _select_trigger(self, summary: DetectionSummary) -> Optional[Dict]:
        if summary.person_count < self.config.trigger_people:
            return None

        fallen = self._fallen_candidate(summary)
        if fallen:
            return {"reason": "fallen_pose_candidate", "box": fallen}

        if summary.level >= self.config.trigger_level:
            return {"reason": "level_threshold", "level": summary.level}

        if summary.person_count >= 3:
            return {"reason": "crowd_check", "person_count": summary.person_count}

        return {"reason": "person_present", "person_count": summary.person_count}

    def _fallen_candidate(self, summary: DetectionSummary) -> Optional[Dict]:
        for box in summary.boxes:
            h = float(box.get("height_ratio", 0.0))
            w = float(box.get("width_ratio", 0.0))
            if h <= 0:
                continue
            if h < self.config.min_box_height_ratio:
                continue
            if w / h >= self.config.fallen_aspect_ratio:
                return box
        return None


def build_vlm_harness() -> VLMHarness:
    config = HarnessConfig(
        enabled=os.getenv("ENABLE_VLM", "0") == "1",
        min_interval_sec=float(os.getenv("VLM_MIN_INTERVAL_SEC", "20")),
        trigger_people=int(os.getenv("VLM_TRIGGER_PEOPLE", "1")),
        trigger_level=int(os.getenv("VLM_TRIGGER_LEVEL", "1")),
        fallen_aspect_ratio=float(os.getenv("VLM_FALLEN_ASPECT_RATIO", "1.25")),
        min_box_height_ratio=float(os.getenv("VLM_MIN_BOX_HEIGHT_RATIO", "0.08")),
    )
    return VLMHarness(config)
