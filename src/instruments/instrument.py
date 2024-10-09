import dataclasses
import os
import yaml
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from pathlib import Path


class InvalidSettingError(Exception):
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument:
    version: str
    constants: dict[str, Union[float, list[float], list[list[float]], dict[str, dict[str, Union[float, list[float]]]]]]
    settings: dict
    models: dict
    default_settings: str
    default_model: str

    @classmethod
    def from_file(cls, path: Path, version: Optional[str] = None):
        data = yaml.safe_load(path)

        if version is None:
            version = data['default_version']

        return cls(
            version,
            data[version]['constants'],
            data[version]['settings'],
            data[version]['models'],
            data['default_settings'],
            data['default_model'],
        )

    @classmethod
    def from_default(cls, instrument: str):
        return cls.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), instrument))

    def get_constant(self, name: str, setting: str):
        return self.settings[setting].get(name, self.constants[name])

    def get_resolution_function(self, model: str, setting: list[str], **_):
        raise NotImplementedError()

    @property
    def available_models(self) -> list[str]:
        return list(self.models.keys())
