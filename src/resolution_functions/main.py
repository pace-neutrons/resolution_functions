from typing import Callable, Optional, TYPE_CHECKING

from instruments import INSTRUMENTS

if TYPE_CHECKING:
    from instruments.instrument import Instrument


def get_instrument(name: str, version: Optional[str] = None) -> Instrument:
    instrument, alt_version = INSTRUMENTS[name.lower()]

    if version is None:
        version = alt_version

    return instrument.from_default(version)
