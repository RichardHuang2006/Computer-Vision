"""AlexNet inference CLI."""
from __future__ import annotations

from src.Classification.common.infer import cli_main

from .model import build_model


def main() -> None:
    cli_main(build_model, "alexnet")


if __name__ == "__main__":
    main()
