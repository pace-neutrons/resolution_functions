Quick Start
===========

With the package :doc:`installed<installation>`, the first step is to create an
instance of a particular :term:`version` of a particular :term:`instrument`:

>>> from resolution_functions import Instrument
>>> tosca = Instrument('TOSCA')
>>> print(tosca)
Instrument(name='TOSCA', version='TOSCA', models=['AbINS', 'book', 'vision'])

To get the :term:`resolution function`, a couple choices might be necessary:

1. Choose the :term:`model` for our instrument.
2. Select one of the :term:`options<option>` for each :term:`configuration` of
   the chosen model.
3. Provide any other :term:`settings<setting>` that the model requires.

If we don't know what the possibilities are for the chosen instrument, the
information can be found either in the :doc:`documentation<instruments>` or
programmatically:

>>> tosca.available_models
['AbINS', 'book', 'vision']
>>> tosca.get_model_signature('book')
<Signature (model_name: Optional[str] = 'book', *, detector_bank: Literal['Backward', 'Forward'] = 'Backward', _) -> resolution_functions.models.tosca_book.ToscaBookModel>

With this, it is possible to make the choices and obtain the resolution function
via the
:py:meth:`~resolution_functions.instrument.Instrument.get_resolution_function`
method:

>>> book = tosca.get_resolution_function('book', detector_bank='Forward')
>>> print(book)
ToscaBookModel(citation=[''])

.. note::

    The settings *must* be passed in as keyword arguments.

The resolution function is a callable object that computes the resolution at the
given energy transfer (in meV) and/or momentum:

>>> book(100)
0.81802604002035
>>> import numpy as np
>>> book(np.array([100, 200, 300]))
array([0.81802604, 1.34222267, 1.88255039])