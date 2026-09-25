from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Capture:
    rgb: Any
    actual_pose: dict
    metadata: dict


class WorldBackend(ABC):
    @abstractmethod
    def reset(self, arena_spec): ...
    @abstractmethod
    def build(self, objects, distractors): ...
    @abstractmethod
    def set_camera(self, pose): ...
    @abstractmethod
    def capture(self) -> Capture: ...
    @abstractmethod
    def close(self): ...
