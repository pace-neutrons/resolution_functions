How To Programmatically Find Data
=================================

While all the information and data stored by ResINS is available in the
documentation *and* comes packaged together with the library via the YAML files
in ``resolution_functions/instrument_data``, there are many reasons why it is
desirable to obtain the data programmatically at runtime. ResINS implements
multiple methods for doing this:

.. contents::
    :backlinks: entry
    :depth: 2
    :local:

.. _how-to-instrument:

How To Find the Available Instruments
-------------------------------------

The :py:class:`~resolution_functions.instrument.Instrument` provides a method,
:py:meth:`~resolution_functions.instrument.Instrument.available_instruments`,
which lists all the available :term:`instruments<instrument>`:

>>> from resolution_functions import Instrument
>>> Instrument.available_instruments()
['ARCS', 'CNCS', 'HYSPEC', 'Lagrange', 'LET', 'MAPS', 'MARI', 'MERLIN', 'PANTHER', 'TFXA', 'TOSCA', 'VISION', 'SEQUOIA']


Advanced
^^^^^^^^

An advanced way of interacting with available instruments is to access the
``INSTRUMENT_MAP`` dictionary directly:

>>> from resolution_functions.instrument import INSTRUMENT_MAP
>>> INSTRUMENT_MAP
{'ARCS': ('arcs', None), 'CNCS': ('cncs', None), 'HYSPEC': ('hyspec', None), 'Lagrange': ('lagrange', None), 'LET': ('let', None), 'MAPS': ('maps', None), 'MARI': ('mari', None), 'MERLIN': ('merlin', None), 'PANTHER': ('panther', None), 'TFXA': ('tosca', 'TFXA'), 'TOSCA': ('tosca', None), 'VISION': ('vision', None), 'SEQUOIA': ('sequoia', None)}

which maps the user-input instrument name to the internal name and the
instrument :term:`version`. There should be no reason to access this object for
normal use.


.. _how-to-version:

How To Find the Versions Available for an Instrument
----------------------------------------------------

Once again, the :py:class:`~resolution_functions.instrument.Instrument` provides
a method,
:py:meth:`~resolution_functions.instrument.Instrument.available_versions`,
which lists all the :term:`versions<version>` available for an
:term:`instruments<instrument>`:

>>> from resolution_functions import Instrument
>>> available_versions, default_version = Instrument.available_versions('TOSCA')
>>> available_versions
['TFXA', 'TOSCA1', 'TOSCA']
>>> default_version
'TOSCA'

This method returns both the list of versions available for the instrument, and
the default version for that instrument. Though, if the instrument being
searched for is an alias for a particular version of an instrument, that version
is returned instead of the default version:

>>> available_versions, default_version = Instrument.available_versions('TFXA')
>>> available_versions
['TFXA', 'TOSCA1', 'TOSCA']
>>> default_version
'TFXA'

.. _how-to-model:

How To Find the Models Available for an Instrument
--------------------------------------------------

Given an :ref:`instrument<how-to-instrument>` and its
:ref:`version<how-to-version>`, it is possible to query the list of available
:term:`models<model>` by using the
:py:meth:`~resolution_functions.instrument.Instrument.available_models`
property:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.available_models
['AbINS', 'book', 'vision']

The default model can similarly be accessed via an attribute:

>>> tosca.default_model
'AbINS'

.. _how-to-config:

How To Find the Configurations that Must be Chosen for a Model
--------------------------------------------------------------

If the model has been chosen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the desired model is known, either from prior knowledge or by browsing the
:ref:`available models<how-to-model>`, the :term:`configurations<configuration>`
can be retrieved using the
:py:meth:`~resolution_functions.instrument.Instrument.possible_configurations_for_model`
method:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.possible_configurations_for_model('AbINS')
[]
>>> tosca.possible_configurations_for_model('book')
['detector_bank']

If the model has NOT been chosen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without the model information, the only way is to list all the
:term:`configurations<configuration>` for all the :term:`models<model>` via the
:py:meth:`~resolution_functions.instrument.Instrument.available_models_and_configurations`
property:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.available_models_and_configurations
{'AbINS': [], 'book': ['detector_bank'], 'vision': ['detector_bank']}

which shows a mapping of model names to lists of their configurations.


How To Find the Options Available for a Configuration
-----------------------------------------------------

If the model and setting are known
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To list the possible :term:`options<option>` for a given :term:`configuration`
(:ref:`how to find configurations<how-to-config>`) of a given :term:`model`
(:ref:`how to find models<how-to-model>`), the
:py:meth:`~resolution_functions.instrument.Instrument.possible_options_for_model_and_configuration`
method is provided:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.possible_options_for_model_and_configuration('book', 'detector_bank')
['Backward', 'Forward']

If only the model is known
^^^^^^^^^^^^^^^^^^^^^^^^^^

To list all the :term:`options<option>` for all :term:`configurations<configuration>` of a
given :term:`model` (:ref:`how to find models<how-to-model>`), the
:py:meth:`~resolution_functions.instrument.Instrument.possible_options_for_model`
method is provided:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.possible_options_for_model('book')
{'detector_bank': ['Backward', 'Forward']}

If the model is not known
^^^^^^^^^^^^^^^^^^^^^^^^^

If the model has not been chosen, all the :term:`options<option>` for all the
:term:`configurations<configuration>` of all :term:`models<model>` have to be enumerated,
which can be done using the
:py:meth:`~resolution_functions.instrument.Instrument.all_available_models_options`
property:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.all_available_models_options
{'AbINS': {}, 'book': {'detector_bank': ['Backward', 'Forward']}, 'vision': {'detector_bank': ['Backward', 'Forward']}}


How To Find the Default Option for a Configuration
--------------------------------------------------

Given the :term:`model` name (:ref:`how to find models<how-to-model>`) and the
:term:`configuration` (:ref:`how to find configurations<how-to-config>`),
the default option can be retrieved using the
:py:meth:`~resolution_functions.instrument.Instrument.default_option_for_configuration`
method:

>>> from resolution_functions import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.default_option_for_configuration('book', 'detector_bank')
'Backward'


How To Find the Default Values for a Setting
--------------------------------------------

The default values for all :term:`settings<setting>` associated with a
:term:`model` can be found using the ``default`` attribute of the model, which
can be retrieved using the
:py:meth:`~resolution_functions.instrument.Instrument.get_model_data` method:

>>> from resolution_functions import Instrument
>>> merlin = Instrument.from_default('MERLIN', 'MERLIN')
>>> model = merlin.get_model_data('PyChop_fit')
>>> type(model)
<class 'resolution_functions.models.pychop.PyChopModelDataFermi'>
>>> model.defaults
{'e_init': 400, 'chopper_frequency': 400}

.. warning::

    For some :term:`models<model>`, the :term:`configurations<configuration>`
    may affect the default values of the :term:`settings<setting>`.


How To Find the Restrictions on the Values of a Setting
-------------------------------------------------------

The restrictions on the values for all :term:`settings<setting>` associated with a
:term:`model` can be found using the ``restrictions`` attribute of the model, which
can be retrieved using the
:py:meth:`~resolution_functions.instrument.Instrument.get_model_data` method:

>>> from resolution_functions import Instrument
>>> merlin = Instrument.from_default('MERLIN', 'MERLIN')
>>> model = merlin.get_model_data('PyChop_fit')
>>> type(model)
<class 'resolution_functions.models.pychop.PyChopModelDataFermi'>
>>> model.restrictions
{'e_init': [0, 181], 'chopper_frequency': [50, 601, 50]}

.. warning::

    For some :term:`models<model>`, the :term:`configurations<configuration>`
    may affect the restrictions on the :term:`settings<setting>`:

>>> model = merlin.get_model_data('PyChop_fit', chopper_package='S')
>>> type(model)
<class 'resolution_functions.models.pychop.PyChopModelDataFermi'>
>>> model.restrictions
{'e_init': [7, 2000], 'chopper_frequency': [50, 601, 50]}
