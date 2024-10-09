from typing import TYPE_CHECKING, Union

from .direct_instruments import PANTHER
from .indirect_instruments import TOSCA, Lagrange
from .instruments_2d import MAPS

if TYPE_CHECKING:
    from .instrument import Instrument


INSTRUMENTS: dict[str, tuple[type[Instrument], Union[None, str]]] = {
    'tosca': (TOSCA, None),
    'tfxa': (TOSCA, 'TFXA'),
    'lagrange': (Lagrange, None),
    'panther': (PANTHER, None),
    'maps': (MAPS, None),
}