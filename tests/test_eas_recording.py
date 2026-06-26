from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from rtl_weatherband.config import EasRecordingConfig
from rtl_weatherband.eas_recording import EasRecorderOutput


class FakeRecorderSettings:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class StopFailingRecorder:
    def __init__(self, settings: FakeRecorderSettings) -> None:
        self.settings = settings
        self.started = False

    def start(self) -> None:
        self.started = True

    def write(self, audio: bytes) -> None:
        pass

    def stop(self) -> None:
        raise RuntimeError("multimon-ng pipeline exited")


class EasRecordingTests(unittest.TestCase):
    def test_close_suppresses_recorder_stop_error(self) -> None:
        fake_easrecorder = types.SimpleNamespace(
            EASRecorder=StopFailingRecorder,
            RecorderSettings=FakeRecorderSettings,
        )
        with patch.dict(sys.modules, {"easrecorder": fake_easrecorder}):
            output = EasRecorderOutput(
                EasRecordingConfig(enabled=True, directory="/tmp/eas-alerts")
            )
            output.close()

        self.assertTrue(output.closed)


if __name__ == "__main__":
    unittest.main()
