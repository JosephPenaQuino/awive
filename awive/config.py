"""Configuration."""

from pydantic import BaseModel


class Config(BaseModel):
    """Config class for awive."""

    image_dataset: str
    image_number_offset: int
    image_path_prefix: str
    image_path_digits: int
    video_path: str
