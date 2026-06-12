from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, load_config
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rtl_weatherband",
        description="Stream NOAA Weather Radio from csdr_server to Icecast.",
    )
    parser.add_argument("config", help="path to a JSON5 configuration file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        run(config, args.config)
    except KeyboardInterrupt:
        return 130
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0
