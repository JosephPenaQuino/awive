"""Configuration."""

from pydantic import BaseModel


class ConfigDataset(BaseModel):
    """Configuration dataset."""

    image_dataset: str
    image_number_offset: int
    image_path_prefix: str
    image_path_digits: int
    video_path: str
    width: int
    height: int


class ConfigOtv(BaseModel):
    """Configuration OTV."""


class ConfigStiv(BaseModel):
    """Configuration STIV."""


class Config(BaseModel):
    """Config class for awive."""

    dataset: ConfigDataset
    otv: ConfigOtv
    stiv: ConfigStiv
