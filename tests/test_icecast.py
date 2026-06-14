from __future__ import annotations

import unittest

from rtl_weatherband import __version__
from rtl_weatherband.config import IcecastConfig
from rtl_weatherband.icecast import IcecastError, IcecastSource


class IcecastTests(unittest.TestCase):
    def test_request_headers_use_package_version_in_user_agent(self) -> None:
        source = IcecastSource(
            IcecastConfig(
                host="127.0.0.1",
                port=8000,
                mount="/nwr.mp3",
                username="source",
                password="hackme",
                format="mp3",
                sample_rate=16000,
                bitrate=32,
            ),
            "audio/mpeg",
        )

        headers = source._request_headers().decode("utf-8")

        self.assertIn(f"User-Agent: rtl_weatherband/{__version__}", headers)

    def test_parses_success_status_code(self) -> None:
        self.assertEqual(
            IcecastSource._response_status_code("HTTP/1.1 200 OK"),
            200,
        )

    def test_rejects_invalid_status_line(self) -> None:
        with self.assertRaises(IcecastError):
            IcecastSource._response_status_code("not-http")


if __name__ == "__main__":
    unittest.main()
