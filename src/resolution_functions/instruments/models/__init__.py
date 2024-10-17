from __future__ import annotations

from typing import TYPE_CHECKING

from .polynomial import PolynomialModel1D, DiscontinuousPolynomialModel1D
from .tosca_book import ToscaBookModel

if TYPE_CHECKING:
    from .model_base import InstrumentModel


MODELS: dict[str, type[InstrumentModel]] = {
    'polynomial_1d': PolynomialModel1D,
    'discontinuous_polynomial': DiscontinuousPolynomialModel1D,
    'tosca_book': ToscaBookModel,
}
