from typing import List, Tuple
from enum import Enum

class ReplayMode(Enum):
    """No recording or replay"""
    IGNORE = 1
    """Record the training buffer"""
    RECORDING = 2
    """Replay a given buffer"""
    REPLAYING = 3

class Recording:
    def __init__(self, source: List) -> None:
        self.pos = 0
        self._history = source

    def __iter__(self) -> "Recording":
        self.pos = 0
        return self

    def __next__(self) -> Tuple:
        if self.pos <= len(self._history):
            x = self._history[self.pos]
            self.pos += 1
            return x
        else:
            raise StopIteration


class Recorder:
    def __init__(self) -> None:
        self._recording = []

    def append(self, new_obs, rewards, dones, infos, buffer_actions) -> None:
        self._recording.append((new_obs, rewards, dones, infos, buffer_actions))

    def freeze(self) -> Recording:
        return Recording(self._recording)
