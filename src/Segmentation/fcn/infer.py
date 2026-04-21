"""FCN inference."""
from __future__ import annotations

from src.Segmentation.common.infer import cli_load_and_run
from src.Segmentation.fcn.model import build_model


def main() -> None:
    cli_load_and_run(lambda: build_model())


if __name__ == "__main__":
    main()
