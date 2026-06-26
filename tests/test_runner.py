from __future__ import annotations

import unittest
from unittest.mock import patch

from rtl_weatherband.config import (
    AppConfig,
    AudioConfig,
    CsdrServerConfig,
    IcecastConfig,
    StationConfig,
)
from rtl_weatherband.runner import (
    ReloadResult,
    _apply_icecast_reload,
    _connect_initial_icecast_outputs,
    _diff_icecast_destinations,
    _icecast_reload_plan,
    _retry_pending_icecast_reload,
    _should_queue_same_test_alert,
    run,
)
from rtl_weatherband.soundcard import SoundcardDependencyError


def destination(
    mount: str,
    bitrate: int = 32,
    password: str = "hackme",
    host: str = "127.0.0.1",
    port: int = 8000,
    username: str = "source",
) -> IcecastConfig:
    return IcecastConfig(
        host=host,
        port=port,
        mount=mount,
        username=username,
        password=password,
        format="mp3",
        sample_rate=16000,
        bitrate=bitrate,
    )


class FakeSink:
    pass


class FakeSource:
    def __init__(
        self,
        config: IcecastConfig,
        events: list[str] | None = None,
    ) -> None:
        self.config = config
        self.events = events
        self.closed = False

    def close(self) -> None:
        if self.events is not None:
            self.events.append(
                f"close {self.config.host}:{self.config.port}{self.config.mount}"
            )
        self.closed = True


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []
        self.icecast = ()

    def apply_icecast_outputs(
        self,
        icecast,
        remove_configs,
        additions,
        stop_unused_encoders=True,
    ) -> None:
        self.calls.append(
            (
                icecast,
                list(remove_configs),
                [config for config, _ in additions],
                stop_unused_encoders,
            )
        )
        self.icecast = icecast


class FatalStartupPipeline:
    def __init__(self, *args, **kwargs) -> None:
        self.stopped = False

    def start(self, encoded_outputs) -> None:
        raise SoundcardDependencyError("pactl missing")

    def stop(self) -> None:
        self.stopped = True


class StopRun(BaseException):
    pass


class CaptureStartupPipeline:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started_outputs = None
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self, encoded_outputs) -> None:
        self.started_outputs = list(encoded_outputs)
        raise StopRun()

    def stop(self) -> None:
        self.stopped = True


def app_config(*destinations: IcecastConfig) -> AppConfig:
    return AppConfig(
        csdr_server=CsdrServerConfig(host="127.0.0.1", port=4951),
        station=StationConfig(frequency=162.55),
        icecast=destinations,
        audio=AudioConfig(),
    )


class RunnerTests(unittest.TestCase):
    def test_soundcard_dependency_error_hard_fails_startup(self) -> None:
        config = app_config(destination("/one.mp3"))

        with patch(
            "rtl_weatherband.runner.IcecastSource",
        ) as source_class, patch(
            "rtl_weatherband.runner.StreamPipeline",
            FatalStartupPipeline,
        ):
            source = source_class.return_value
            source.connect.return_value = FakeSink()
            with self.assertRaises(SoundcardDependencyError):
                run(config)

    def test_initial_icecast_connect_keeps_successful_destinations(self) -> None:
        bad = destination("/bad.mp3")
        good = destination("/good.mp3")
        good_source = FakeSource(good)
        good_sink = FakeSink()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            side_effect=[
                ConnectionError("bad password"),
                (good_source, good_sink),
            ],
        ):
            sources, outputs, connected = _connect_initial_icecast_outputs(
                app_config(bad, good),
            )

        self.assertEqual(sources, [good_source])
        self.assertEqual(outputs, [good_sink])
        self.assertEqual(connected, (good,))

    def test_startup_icecast_failure_does_not_block_pipeline_start(self) -> None:
        bad = destination("/bad.mp3")
        good = destination("/good.mp3")
        good_source = FakeSource(good)
        good_sink = FakeSink()
        CaptureStartupPipeline.instances = []

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            side_effect=[
                ConnectionError("bad password"),
                (good_source, good_sink),
            ],
        ), patch(
            "rtl_weatherband.runner.StreamPipeline",
            CaptureStartupPipeline,
        ):
            with self.assertRaises(StopRun):
                run(app_config(bad, good))

        pipeline = CaptureStartupPipeline.instances[0]
        self.assertEqual(pipeline.args[1], (good,))
        self.assertEqual(pipeline.started_outputs, [good_sink])
        self.assertTrue(pipeline.stopped)

    def test_same_test_noop_reload_queues_alert(self) -> None:
        config = app_config(destination("/one.mp3"))

        self.assertTrue(
            _should_queue_same_test_alert(
                True,
                config,
                ReloadResult(config, had_errors=False),
            )
        )

    def test_same_test_changed_reload_does_not_queue_alert(self) -> None:
        current = app_config(destination("/one.mp3"))
        changed = app_config(destination("/two.mp3"))

        self.assertFalse(
            _should_queue_same_test_alert(
                True,
                current,
                ReloadResult(changed, had_errors=False),
            )
        )

    def test_same_test_invalid_reload_does_not_queue_alert(self) -> None:
        config = app_config(destination("/one.mp3"))

        self.assertFalse(
            _should_queue_same_test_alert(
                True,
                config,
                ReloadResult(config, had_errors=True),
            )
        )

    def test_normal_noop_reload_does_not_queue_alert(self) -> None:
        config = app_config(destination("/one.mp3"))

        self.assertFalse(
            _should_queue_same_test_alert(
                False,
                config,
                ReloadResult(config, had_errors=False),
            )
        )

    def test_icecast_diff_keeps_unchanged_destination(self) -> None:
        first = destination("/one.mp3")
        second = destination("/two.mp3")

        remove_configs, add_configs = _diff_icecast_destinations(
            (first, second),
            (first,),
        )

        self.assertEqual(remove_configs, [second])
        self.assertEqual(add_configs, [])

    def test_icecast_diff_replaces_changed_destination(self) -> None:
        old = destination("/one.mp3", bitrate=32)
        new = destination("/one.mp3", bitrate=40)

        remove_configs, add_configs = _diff_icecast_destinations((old,), (new,))

        self.assertEqual(remove_configs, [old])
        self.assertEqual(add_configs, [new])

    def test_icecast_reload_replaces_changed_mountpoint(self) -> None:
        old = destination("/old.mp3")
        new = destination("/new.mp3")
        events = []
        old_source = FakeSource(old, events)
        new_source = FakeSource(new, events)
        sources = [old_source]
        pipeline = FakePipeline()

        def connect(config, timeout):
            events.append(f"connect {config.host}:{config.port}{config.mount}")
            return new_source, FakeSink()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            side_effect=connect,
        ) as connect:
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (old,),
                (new,),
                timeout=1.0,
            )

        self.assertTrue(applied)
        connect.assert_called_once_with(new, 1.0)
        self.assertEqual(
            events,
            [
                "connect 127.0.0.1:8000/new.mp3",
                "close 127.0.0.1:8000/old.mp3",
            ],
        )
        self.assertEqual(pipeline.calls, [((new,), [old], [new], True)])
        self.assertEqual(sources, [new_source])
        self.assertTrue(old_source.closed)
        self.assertFalse(new_source.closed)

    def test_icecast_reload_does_not_reconnect_unchanged_destination(self) -> None:
        unchanged = destination("/one.mp3")
        old_changed = destination("/two.mp3")
        new_changed = destination("/three.mp3")
        unchanged_source = FakeSource(unchanged)
        old_changed_source = FakeSource(old_changed)
        new_changed_source = FakeSource(new_changed)
        sources = [unchanged_source, old_changed_source]
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            return_value=(new_changed_source, FakeSink()),
        ) as connect:
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (unchanged, old_changed),
                (unchanged, new_changed),
                timeout=1.0,
            )

        self.assertTrue(applied)
        connect.assert_called_once_with(new_changed, 1.0)
        self.assertEqual(
            pipeline.calls,
            [((unchanged, new_changed), [old_changed], [new_changed], True)],
        )
        self.assertEqual(sources, [unchanged_source, new_changed_source])
        self.assertFalse(unchanged_source.closed)
        self.assertTrue(old_changed_source.closed)
        self.assertFalse(new_changed_source.closed)

    def test_icecast_reload_plan_marks_exact_matches_unchanged_first(self) -> None:
        unchanged = destination("/one.mp3")
        old_changed = destination("/two.mp3")
        new_changed = destination("/three.mp3")

        plan = _icecast_reload_plan(
            (unchanged, old_changed),
            (unchanged, new_changed),
        )

        self.assertEqual(plan.unchanged, (unchanged,))
        self.assertEqual(plan.replacements, ((old_changed, new_changed),))
        self.assertEqual(plan.pure_removes, ())
        self.assertEqual(plan.pure_adds, ())

    def test_icecast_reload_replaces_changed_server_and_port(self) -> None:
        old = destination("/same.mp3", host="old.example", port=8000)
        new = destination("/same.mp3", host="new.example", port=9000)
        old_source = FakeSource(old)
        new_source = FakeSource(new)
        sources = [old_source]
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            return_value=(new_source, FakeSink()),
        ) as connect:
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (old,),
                (new,),
                timeout=1.0,
            )

        self.assertTrue(applied)
        connect.assert_called_once_with(new, 1.0)
        self.assertEqual(pipeline.calls, [((new,), [old], [new], True)])
        self.assertEqual(sources, [new_source])
        self.assertTrue(old_source.closed)

    def test_icecast_reload_replaces_same_mount_after_disconnect(self) -> None:
        old = destination("/same.mp3", password="old")
        new = destination("/same.mp3", password="new")
        old_source = FakeSource(old)
        new_source = FakeSource(new)
        sources = [old_source]
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            return_value=(new_source, FakeSink()),
        ) as connect:
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (old,),
                (new,),
                timeout=1.0,
            )

        self.assertTrue(applied)
        connect.assert_called_once_with(new, 1.0)
        self.assertEqual(
            pipeline.calls,
            [
                ((), [old], [], False),
                ((new,), [old], [new], True),
            ],
        )
        self.assertEqual(sources, [new_source])
        self.assertTrue(old_source.closed)
        self.assertFalse(new_source.closed)

    def test_icecast_reload_replaces_same_mount_username_after_disconnect(self) -> None:
        old = destination("/same.mp3", username="old")
        new = destination("/same.mp3", username="new")
        old_source = FakeSource(old)
        new_source = FakeSource(new)
        sources = [old_source]
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            return_value=(new_source, FakeSink()),
        ) as connect:
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (old,),
                (new,),
                timeout=1.0,
            )

        self.assertTrue(applied)
        connect.assert_called_once_with(new, 1.0)
        self.assertEqual(
            pipeline.calls,
            [
                ((), [old], [], False),
                ((new,), [old], [new], True),
            ],
        )
        self.assertTrue(old_source.closed)
        self.assertFalse(new_source.closed)

    def test_icecast_reload_failure_keeps_existing_outputs(self) -> None:
        old = destination("/old.mp3")
        new = destination("/new.mp3")
        old_source = FakeSource(old)
        sources = [old_source]
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            side_effect=ConnectionError("no route"),
        ):
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (old,),
                (new,),
                timeout=1.0,
            )

        self.assertFalse(applied)
        self.assertEqual(pipeline.calls, [])
        self.assertEqual(sources, [old_source])
        self.assertFalse(old_source.closed)

    def test_icecast_reload_can_disable_destination(self) -> None:
        old = destination("/one.mp3")
        old_source = FakeSource(old)
        sources = [old_source]
        pipeline = FakePipeline()

        applied = _apply_icecast_reload(
            pipeline,
            sources,
            (old,),
            (),
            timeout=1.0,
        )

        self.assertTrue(applied)
        self.assertEqual(pipeline.calls, [((), [old], [], True)])
        self.assertEqual(sources, [])
        self.assertTrue(old_source.closed)

    def test_icecast_reload_can_reenable_destination(self) -> None:
        new = destination("/one.mp3")
        new_source = FakeSource(new)
        sources = []
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            return_value=(new_source, FakeSink()),
        ) as connect:
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (),
                (new,),
                timeout=1.0,
            )

        self.assertTrue(applied)
        connect.assert_called_once_with(new, 1.0)
        self.assertEqual(pipeline.calls, [((new,), [], [new], True)])
        self.assertEqual(sources, [new_source])
        self.assertFalse(new_source.closed)

    def test_same_mount_reload_failure_restores_existing_output(self) -> None:
        old = destination("/same.mp3", password="old")
        new = destination("/same.mp3", password="new")
        old_source = FakeSource(old)
        restored_source = FakeSource(old)
        sources = [old_source]
        pipeline = FakePipeline()

        with patch(
            "rtl_weatherband.runner._connect_icecast",
            side_effect=[
                ConnectionError("bad password"),
                (restored_source, FakeSink()),
            ],
        ):
            applied = _apply_icecast_reload(
                pipeline,
                sources,
                (old,),
                (new,),
                timeout=1.0,
            )

        self.assertFalse(applied)
        self.assertEqual(
            pipeline.calls,
            [
                ((), [old], [], False),
                ((old,), [], [old], True),
            ],
        )
        self.assertEqual(sources, [restored_source])
        self.assertTrue(old_source.closed)
        self.assertFalse(restored_source.closed)

    def test_pending_icecast_reload_failure_keeps_current_outputs(self) -> None:
        old = app_config(destination("/old.mp3"))
        new = app_config(destination("/new.mp3"))
        pipeline = FakePipeline()
        sources = [FakeSource(old.icecast[0])]

        with patch(
            "rtl_weatherband.runner._apply_icecast_reload",
            return_value=False,
        ) as apply_reload:
            applied = _retry_pending_icecast_reload(pipeline, sources, old, new)

        self.assertFalse(applied)
        apply_reload.assert_called_once_with(
            pipeline,
            sources,
            old.icecast,
            new.icecast,
            old.csdr_server.timeout,
        )
        self.assertEqual(pipeline.calls, [])

    def test_pending_icecast_reload_success_reports_applied(self) -> None:
        old = app_config(destination("/old.mp3"))
        new = app_config(destination("/new.mp3"))
        pipeline = FakePipeline()
        sources = [FakeSource(old.icecast[0])]

        with patch(
            "rtl_weatherband.runner._apply_icecast_reload",
            return_value=True,
        ):
            applied = _retry_pending_icecast_reload(pipeline, sources, old, new)

        self.assertTrue(applied)


if __name__ == "__main__":
    unittest.main()
