from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import Float


class InstrumentModel:
    input: ClassVar[int]
    output: ClassVar[int]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class InstrumentModel1D(InstrumentModel):
    input = 1

    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs):
        raise NotImplementedError()


class InstrumentModel2D(InstrumentModel):
    input = 2

    def __call__(self,
                 frequencies: Float[np.ndarray, 'frequencies'],
                 q_scalars: Float[np.ndarray, 'frequencies'],
                 *args, **kwargs):
        raise NotImplementedError()
