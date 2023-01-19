"""Basic example of using OTV."""

import json
import time
from pathlib import Path

from awive.algorithms.otv import Otv
from awive.config import Config
from awive.loader import Loader, make_loader

CONFIG_PATH = "examples/basic/config.json"
VIDEO_ID = "basic"


def run_otv(config_path: str, video_identifier: str) -> None:
    """Run OTV on a dataset."""
    config: Config = Config(
        **json.loads(
            Path(config_path).read_text()
        )[video_identifier]
    )

    loader: Loader = make_loader(config.dataset)
    otv: Otv = Otv(config.otv)
    start: float = time.time()
    otv.run(loader)
    end: float = time.time()
    print(f"OTV took {end - start} seconds")


if __name__ == "__main__":
    run_otv(CONFIG_PATH, VIDEO_ID)
