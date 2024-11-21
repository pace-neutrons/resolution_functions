"""
This module is the main part of the library, containing the :py:cls:`Instrument` object, which is
used as a starting point for all functionality, as it represents a physical INS instrument and all
its data. To get started, you will need to know which instrument and which version of that
instrument you want to use (for more information about versions, please see :py:cls:`Instrument`),
all of which can be found in this documentation of programmatically:

```
>>> from resolution_functions import Instrument
>>> Instrument.available_instruments()
['ARCS', 'CNCS', 'HYSPEC', 'Lagrange', 'LET', 'MAPS', 'MARI', 'MERLIN', 'PANTHER', 'TFXA', 'TOSCA', 'VISION', 'SEQUOIA']
>>> Instrument.available_versions('TOSCA')
['TFXA', 'TOSCA1', 'TOSCA']
```

With this information, it is possible to instantiate the relevant instrument:

```
>>> tfxa = Instrument.from_default('TOSCA', 'TFXA')
>>> tfxa.name
'TOSCA'
>>> tfxa.version
'TFXA'
>>> tfxa.default_model
'book'
```

However, even if we don't know the exact version of an instrument to use, a reasonable default is
provided (usually the most recent version of that instrument):

```
>>> tosca = Instrument.from_default('TOSCA')
>>> tosca.name
'TOSCA'
>>> tosca.version
'TOSCA'
>>> tosca.default_model
'AbINS'
```

A lower level API is also provided for advanced use cases (see Advanced Useage).

Once we have the instrument, the next step is to get a model of that instrument. Models describe the
widening (i.e. resolution) of a given instrument at different energy transfers - there are many
ways of modeling this behaviour, which is why most instruments have multiple models. For more
information see :py:mod:`models`. The important part here is that different models include different
levels of detail and so may require different information. If we know all the information for our
model (or wish to use the defaults), we can skip to getting the model, but there are programmatic
ways to query all the various information.

All the things you will need to consider can be obtained in progressive levels by using the
properties:

```
>>> # Reveal all the models that we can use with the TOSCA instrument
>>> tosca.available_models
['AbINS', 'book', 'vision']
>>> # Reveal all the models, and which parameters we will need to pass in to each one
>>> tosca.available_models_and_settings
{'AbINS': [], 'book': ['detector_bank'], 'vision': ['detector_bank']}
>>> # Reveal all the models, which parameters we will need to pass in to each one, and what the options are for each of the parameters
>>> tosca.all_available_models_options
{'AbINS': {}, 'book': {'detector_bank': ['Backward', 'Forward']}, 'vision': {'detector_bank': ['Backward', 'Forward']}}
```

It is also possible to make more precise queries:

```
>>> tosca.possible_settings_for_model('book')
['detector_bank']
>>> tosca.possible_options_for_model('book')
{'detector_bank': ['Backward', 'Forward']}
>>> tosca.possible_options_for_model_and_setting('book', 'detector_bank')
['Backward', 'Forward']
>>> tosca.default_option_for_setting('book', 'detector_bank')
'Backward'
```

However, the most comphrehensive way to prepare for getting a model, once we knowh which model we
want, is to get its signature:

```
>>> maps = Instrument.from_default('MAPS', 'MAPS')
>>> signature = maps.get_model_signature('PyChop_fit')
>>> signature
<Signature (model_name: Optional[str] = 'PyChop_fit', *, chopper_package: Literal['A', 'B', 'S'] = 'A', e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500, chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400, fitting_order: 'Optional[int]' = 4, _)>
```

This function returns a `inspect.Signature` object from the standard library, which can be further
queried and contains all the information required for getting the model.

```
>>> # For the MAPS instrument and its PyChop_fit model:
>>> # The first parameter, called model name, is a string with the name of the model. The default is 'PyChop_fit'
>>> signature.parameters['model_name']
<Parameter "model_name: Optional[str] = 'PyChop_fit'">
>>> # The parameter chopper_package is one of the Fermi chopper options on the MAPS instrument.
>>> # It can be one of 'A', 'B', or 'S'. The default is 'A'
>>> signature.parameters['chopper_package']
<Parameter "chopper_package: Literal['A', 'B', 'S'] = 'A'">
>>> # The parameter e_init (initial energy) is a float with a default value of 500 meV and the
>>> # restriction that it must be within 0 < e < 2000
>>> signature.parameters['e_init']
<Parameter "e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500">
>>> # The parameter chopper_frequency specifies the frequency of the Fermi chopper.
>>> # The default is 400 Hz with the restriction that it must be one of the ints between 50 and 600 with a period of 50.
>>> signature.parameters['chopper_frequency']
<Parameter "chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400">
>>> # The fitting_order parameter is an int with a default of 4, specifying the order of the polynomial used in fitting.
>>> signature.parameters['fitting_order']
>>> # We can also check whether each of the parameters is positional or keyword argument
>>> [parameter.kind for parameter in signature.parameters.values()]
[<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>, <_ParameterKind.KEYWORD_ONLY: 3>, <_ParameterKind.KEYWORD_ONLY: 3>, <_ParameterKind.KEYWORD_ONLY: 3>, <_ParameterKind.KEYWORD_ONLY: 3>, <_ParameterKind.KEYWORD_ONLY: 3>]
```


"""
from __future__ import annotations

from collections import ChainMap
import dataclasses
import os
import yaml
from typing import Optional, Union, TYPE_CHECKING

from .models import MODELS

if TYPE_CHECKING:
    from .models.model_base import ModelData, InstrumentModel
    from inspect import Signature

INSTRUMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instrument_data')

INSTRUMENT_MAP: dict[str, tuple[str, Union[None, str]]] = {
    'ARCS': ('arcs', None),
    'CNCS': ('cncs', None),
    'HYSPEC': ('hyspec', None),
    'Lagrange': ('lagrange', None),
    'MAPS': ('maps', None),
    'MARI': ('mari', None),
    'MERLIN': ('merlin', None),
    'PANTHER': ('panther', None),
    'TFXA': ('tosca', 'TFXA'),
    'TOSCA': ('tosca', None),
    'VISION': ('vision', None),
    'SEQUOIA': ('sequoia', None),
}


class InvalidInstrumentError(Exception):
    """An Exception representing an invalid user input for the instrument name."""
    pass


class InvalidModelError(Exception):
    """
    An Exception representing an invalid user input for the model of an instrument.

    This class does not support an arbitrary message; instead the message is constructed in here
    from the provided information.

    Parameters
    ----------
    provided_name
        The invalid name for the model that the user provided.
    instrument
        The instance of the `Instrument` object in which the `provided_name` was used.
    """
    def __init__(self, provided_name: str, instrument: Instrument):
        message = f'"{provided_name}" is not a valid model for the {instrument.name} instrument ' \
                  f'version {instrument.version}. This instrument only supports the following ' \
                  f'models: {instrument.available_models}.'

        super().__init__(message)


class InvalidSettingError(Exception):
    """
    An Exception representing an invalid user input for the setting of a model of an instrument.

    This class does not support an arbitrary message; instead the message is constructed in here
    from the provided information.

    Parameters
    ----------
    provided_name
        The invalid name for the setting that the user provided.
    model_name
        The name of the model for which the `provided_name` was provided.
    instrument
        The instance of the `Instrument` object in which the `provided_name` was used.
    """
    def __init__(self, provided_name: str, model_name: str, instrument: Instrument):
        message = f'"{provided_name}" is not a valid setting for the {model_name} model of the ' \
                  f'{instrument.version} version of the {instrument.name} instrument. This ' \
                  f'instrument only supports the following models: {instrument.available_models}.'

        super().__init__(message)


class InvalidVersionError(Exception):
    """An Exception representing an invalid user input for the version of an instrument."""
    pass


@dataclasses.dataclass(init=True, repr=True, frozen=True, slots=True)
class Instrument:
    """
    Instrument is a representation of a physical INS instrument, containing all its associated data.

    To be precise, it holds all information about one version of an instrument (for more about
    instrument versions, see ...), which makes it the centrepiece of this library; the data is
    necessary for computing the resolution functions.

    However, this information is static and curated by the library, which is why Instrument is a
    frozen data class. It should never be instantiated directly; instead the `from_default`
    constructor should be used. Similarly, it should not be inspected directly; a variety of methods
    and properties are provided for querying relevant information.

    Regardless, the most important function of Instrument is to construct a resolution function,
    which can be done by using the `get_resolution_function` method.

    Parameters
    ----------
    name
        The name of the INS instrument.
    version
        The name of a particular version of that INS instrument.
    _models
        A dictionary detailing all the models and their data available for this version of this
        instrument.
    default_model
        The default model for this version of this instrument.

    Attributes
    ----------
    name
        The name of the INS instrument represented by this instance.
    version
        The version of the INS instrument represented by this instance.
    default_model
        The name of the model for this version of this INS instrument that is used by default.
    instrument_versions
    available_models
    available_models_and_settings
    all_available_models_options
    """
    name: str
    version: str
    _models: dict[str, dict[str, Union[str, Union[dict[str, Union[float, int, str, list[float], dict]],
    dict[str, dict[str, Union[float, int, str, list[float]]]]]]]]
    default_model: str

    def __str__(self):
        return f'Instrument(name={self.name}, version={self.version})'

    @classmethod
    def available_instruments(cls) -> list[str]:
        """
        Lists all INS instruments currently available.

        Returns
        -------
        instrument_list
            A list of names of INS instruments supported by this library.
        """
        return list(INSTRUMENT_MAP.keys())

    @classmethod
    def _available_versions(cls, path: str) -> tuple[list[str], str]:
        """
        Lists the names of all versions of the INS instrument contained in the file found at `path`.

        Parameters
        ----------
        path
            The path to the file that will be inspected for versions.

        Returns
        -------
        available_versions
            A list of the version names found in the provided file.
        default_version
            The default version of this instrument, as specified in the file.

        Warnings
        --------
        This method performs an I/O (read) operation.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return list(data['version'].keys()), data['default_version']

    @classmethod
    def available_versions(cls, instrument_name: str) -> tuple[list[str], str]:
        """
        Lists the names of all versions available for an INS instrument, as well as the default version.

        Parameters
        ----------
        instrument_name
            The name of the INS instrument whose versions to retrieve.

        Returns
        -------
        available_versions
            A list of the version names available for the instrument.
        default_version
            The version of the instrument that is used by default.

        Warnings
        --------
        This method performs an I/O (read) operation.

        See Also
        --------
        available_instruments : Lists the available instruments
        """
        path, implied_version = cls._get_file(instrument_name)

        versions, default_version = list(cls._available_versions(path))

        if implied_version is None:
            return versions, default_version
        else:
            return versions, implied_version

    @classmethod
    def from_file(cls, path: str, version: Optional[str] = None) -> Instrument:
        """
        Instantiates an `Instrument` from the data loaded from the file found at `path`.

        Please note that while this method is a part of the public API, it is marked as being for
        advanced use only. For most use cases, use the `from_default` method.

        This method assumes that the data in the file follows the standard (please see ... for more
        details). No validation is performed, so if there are any issues, either unhandled
        exceptions will be raised, or the errors will be silently propagated. For adding new
        instruments, versions, or models, please open an issue on our GitHub. For other purposes,
        please see ... .

        While the data file may contain multiple versions, only the one specified by `version` will
        be saved in memory. If the `version` parameter is not provided, the default version is read
        from the file.

        Parameters
        ----------
        path
            The path to the file containing the instrument data.
        version
            The version of the instrument to load. If not provided, ``the default_version``
            specified in the file will be used.

        Returns
        -------
        instrument
            An instance of the `Instrument` class containing the data found at `path`

        Raises
        ------
        InvalidVersionError
            If the file does not contain the provided `version`.
        KeyError
            If certain parts of the file are egregiously incorrect.
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if version is None:
            version = data['default_version']

        version_data = data['version']
        try:
            version_data = version_data[version]
        except KeyError:
            versions = list(version_data.keys())
            raise InvalidVersionError(f'"{version}" is not a valid version name. Only the following'
                                      f' versions are supported for this instrument: {versions}')

        return cls(
            data['name'],
            version,
            version_data['models'],
            version_data['default_model'],
        )

    @classmethod
    def from_default(cls, name: str, version: Optional[str] = None) -> Instrument:
        """
        Instantiates an `Instrument` class with the data of the `name` INS instrument and its `version`.

        This is the primary, recommended way of instantiating the Instrument class. It loads the
        instrument data as curated in this library for a particular version of an INS instrument.

        Parameters
        ----------
        name
            The name of the INS instrument to instantiate.
        version
            The version of the `name` INS instrument to instantiate. If not provided, the default
            version of that instrument is instantiated.

        Returns
        -------
        instrument
            An instance of the `Instrument` class containing the data corresponding to `name` and
            `version`.

        Raises
        ------
        InvalidInstrumentError
            If the specified instrument `name` does not exist.
        InvalidVersionError
            If the specified `version` is not available for the specified instrument `name`.

        See Also
        --------
        available_instruments : Lists the available instruments
        available_versions : Lists the available versions of an instrument and its default version.
        """
        path, implied_version = cls._get_file(name)

        if version is None:
            version = implied_version

        return cls.from_file(path, version)

    @staticmethod
    def _get_file(instrument_name: str) -> tuple[str, Union[str, None]]:
        """
        Private method for obtaining the path to the default file corresponding to `instrument_name`.

        This method is mostly a wrapper around the ``INSTRUMENT_MAP`` dictionary. Therefore, it also
        provides the functionality that `instrument_name` does not have to be a unique "instrument",
        but can instead be an alias for a combination of an ``instrument`` and ``version``. For
        example, TFXA is in this library considered a version of the TOSCA instrument, but it can
        be passed in to this method as ``instrument_name='TFXA'`, which will be correctly
        interpreted as ``name='TOSCA'`` and ``version='TFXA'``. This is the purpose of the
        ``implied_version`` output parameter.

        Parameters
        ----------
        instrument_name
            The name of the instrument whose file to retrieve.

        Returns
        -------
        path
            The path to the file corresponding to `instrument_name`
        implied_version
            The version implied by `instrument_name`

        Raises
        ------
        InvalidInstrumentError
            If the specified instrument `name` does not exist.
        """
        try:
            file_name, implied_version = INSTRUMENT_MAP[instrument_name]
        except KeyError:
            raise InvalidInstrumentError(
                f'"{instrument_name}" is not a valid instrument name. Only the following instruments are '
                f'supported: {list(INSTRUMENT_MAP.keys())}')

        return os.path.join(INSTRUMENT_DATA_PATH, file_name + '.yaml'), implied_version

    def get_model_data(self, model_name: Optional[str] = None, **kwargs) -> ModelData:
        """
        Retrieves the physical parameters associated with the specified `model_name`.

        This method can be used for inspecting the parameters of a particular model, but cannot be
        used to modify them. It returns a subclass of the `ModelData` class corresponding to the
        particular model. Another use for this method is to inspect the default values for the
        model's parameters, as well as any restrictions that they might have, via the
        `ModelData.restrictions` and `ModelData.defaults` attributes.

        Parameters
        ----------
        model_name
            The name of the model whose parameters to retrieve. If not provided, the parameters of
            the default_model will be retrieved.
        kwargs
            Keyword arguments can be passed in to choose an option for each of settings specific to
            the `model_name`. If not provided, default values are used.

        Returns
        -------
        model_data
            The data associated with `model_name`.

        See Also
        --------
        default_model : The default model for this instrument.
        available_models : List of models available for this instrument
        possible_settings_for_model : List of settings that can be chosen for this model.
        """
        out, _ = self._get_model_data(model_name, **kwargs)
        return out

    def _get_model_data(self, model_name: Optional[str] = None, **kwargs) -> tuple[ModelData, str]:
        """
        The specific implementation for `get_model_data`.

        Parameters
        ----------
        model_name
            The name of the model whose parameters to retrieve. If not provided, the parameters of
            the default_model will be retrieved.
        kwargs
            Keyword arguments can be passed in to choose an option for each of settings specific to
            the `model_name`. If not provided, default values are used.

        Returns
        -------
        model_data
            The data associated with `model_name`.
        model_name
            The name of the returned model. This will be the same as `model_name` if it was
            provided, otherwise it is the default model name.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not available for this version of this instrument.
        """
        if model_name is None:
            model_name = self.default_model

        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        available_settings = model['settings']

        settings = []
        for setting_name, options in available_settings.items():
            kwarg = kwargs.pop(setting_name, None)
            if kwarg is None:
                kwarg = options['default_setting']

            settings.append(options[kwarg])

        model_class = MODELS[model['function']]
        return model_class.data_class(function=model['function'],
                                      citation=model['citation'],
                                      **ChainMap(*settings, model['parameters'])), model_name

    def get_resolution_function(self, model_name: Optional[str] = None, **kwargs) -> InstrumentModel:
        """
        Generates a resolution function, as modelled by the `model_name` model, for a set of parameters.

        This method is the main use case of the `Instrument` class. It generates a callable object
        that, when called, returns the resolution of the instrument at an energy value(s).

        However, while a simple, common interface is provided, different models (and sometimes the
        same model for different instruments!) require different settings to be chosen and different
        parameters to be provided. All of these have to be passed in as keyword arguments (though
        sensible defaults are provided). These keyword arguments correspond to physical user choices
        made when running an INS experiment on the corresponding instrument. For example, direct
        instruments have a tunable incident energy, so their models usually require an ``e_init``
        parameter. For more information about a model, please see its corresponding documentation,
        or for programmatic querying, please see "How to programmatically query model".

        Parameters
        ----------
        model_name
            The name of the model to instantiate. If not provided, the `default_model` is used.
        kwargs
            Keyword arguments specifying the various settings and parameters of the `model_name` model

        Returns
        -------
        model
            An instance of the requested `model_name` model.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not available for this version of this instrument.

        Warnings
        --------
        Other custom, model-specific errors may be raised, e.g. the PyChop model can raise the
        ``NoTransmissionError``. Default Python errors are likely to be bugs.

        *Importantly*, if there are any mistakes in the model-specific parameters passed in as
        keyword arguments, they will be silently ignored and the default values for the missing
        parameters will be used.

        See Also
        --------
        available_models : List of models available for this version of this instrument
        get_model_signature : Constructs a call signature for calling this method for a particular model.

        Examples
        --------
        >>> from resolution_functions import Instrument
        >>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
        >>> print(tosca.get_resolution_function())
        PolynomialModel1D(citation="")

        If a model is not provided, the default model will be used - this differs between versions
        and instruments.

        >>> print(tosca.get_resolution_function('vision'))
        VisionPaperModel(citation="https://doi.org/10.1016/j.nima.2009.03.204")
        """
        model_data, model_name = self._get_model_data(model_name, **kwargs)
        model_class = MODELS[model_data.function]

        return model_class(model_data, **kwargs)

    def get_model_signature(self, model_name: Optional[str] = None) -> Signature:
        """
        Constructs a call signature for the `get_resolution_function` method with a specific model.

        This method provides a programmatic way of inspecting the call signature of the
        `get_resolution_function` method required when calling it for the `model_name` model. This
        is useful because its default signature uses the ``**kwargs`` construct to provide a
        unified interface, but in fact different models require different sets of parameters that
        have to be passed in through the keyword arguments.

        There are other methods and properties that can be used to inspect some of the options, but
        this method retrieves all the information and returns it as an `inspect.Signature` object
        that can be used to examine the signature in detail. The only other comprehensive source
        of this information is the documentation for the relevant model.

        Parameters
        ----------
        model_name
            The name of the model whose signature to construct. If not provided, the signature of
            the `default_model` is constructed.

        Returns
        -------
        signature
            The call signature of the `get_resolution_method` for the `model_name` model.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not available for this version of this instrument.

        See Also
        --------
        available_models : List of models available for this version of this instrument.
        get_model_data : Allows for checking the default values of and restrictions on model parameters.
        possible_options_for_model : Lists the settings and their options for a model.

        Examples
        --------
        >>> maps = Instrument.from_default('MAPS')
        >>> sig = maps.get_model_signature()
        >>> sig
        <Signature (model_name: Optional[str] = 'PyChop_fit', *, chopper_package: Literal['A', 'B', 'S'] = 'A', e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500, chopper_frequency: Annotated[ForwardRef('Optional[int]'), 'restriction=[50, 601, 50]'] = 400, fitting_order: 'Optional[int]' = 4, _)>
        >>> sig.parameters['e_init']
        <Parameter "e_init: Annotated[ForwardRef('Optional[float]'), 'restriction=[0, 2000]'] = 500">
        >>> sig.parameters['e_init'].kind
        <_ParameterKind.KEYWORD_ONLY: 3>

        The `inspect.Signature` object provides easy inspection of a call signature.
        """
        from inspect import signature, Signature, Parameter
        from typing import Annotated, Literal

        model_data, model_name = self._get_model_data(model_name)
        model_class = MODELS[model_data.function]

        signature = signature(model_class)

        params = {
            'model_name': Parameter('model_name',
                                    Parameter.POSITIONAL_OR_KEYWORD,
                                    default=model_name,
                                    annotation=Optional[str])
        }

        for setting_name, options in self._models[model_name]['settings'].items():
            option_names = self._get_options(options)
            params[setting_name] = Parameter(setting_name,
                                             Parameter.KEYWORD_ONLY,
                                             default=options['default_setting'],
                                             annotation=Literal[tuple(option_names)])

        for key, value in signature.parameters.items():
            if key == 'model_data':
                continue

            args = {}
            try:
                args['default'] = model_data.defaults[key]
            except KeyError:
                pass

            try:
                args['annotation'] = Annotated[value.annotation, f'restriction={model_data.restrictions[key]}']
            except KeyError:
                pass

            params[key] = value.replace(**args, kind=Parameter.KEYWORD_ONLY)

        return Signature(parameters=list(params.values()))

    @property
    def available_models(self) -> list[str]:
        """
        A list of all models available for this version of this instrument.

        Returns
        -------
        available_models
            A list of all available models.
        """
        return list(self._models.keys())

    @property
    def available_models_and_settings(self) -> dict[str, list[str]]:
        """
        A dictionary mapping each available model to the user settings available for that model.

        All models available for this version of this instrument, and all their settings are listed.

        Returns
        -------
        models_and_settings
            All models and all their settings.
        """
        return {model_name: list(model['settings'].keys()) for model_name, model in self._models.items()}

    @property
    def all_available_models_options(self) -> dict[str, dict[str, list[str]]]:
        """
        A dictionary mapping each available model, to the user settings and its options.

        All models available for this version of this instrument, all the settings of each of the
        models, and all the options for each of the settings, are listed.

        Returns
        -------
        everything
            All models, all their settings, and all their options.
        """
        return {model_name: {setting: self._get_options(value) for setting, value in list(model['settings'].items())}
                for model_name, model in self._models.items()}

    def possible_settings_for_model(self, model_name: str) -> list[str]:
        """
        Returns all the settings that the `model_name` model supports.

        Parameters
        ----------
        model_name
            The name of the model whose settings to retrieve.

        Returns
        -------
        settings
            A list of settings available for the `model_name` model.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        return list(model['settings'].keys())

    def possible_options_for_model(self, model_name: str) -> dict[str, list[str]]:
        """
        Returns a dictionary mapping all the settings of the `model_name` model to their options.

        Parameters
        ----------
        model_name
            The name of the model whose settings to retrieve.

        Returns
        -------
        settings_and_options
            All the settings available for the `model_name` model and all their possible options.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        return {setting: self._get_options(value) for setting, value in model['settings'].items()}

    def possible_options_for_model_and_setting(self, model_name: str, setting: str) -> list[str]:
        """
        Lists all the options that can be chosen for a given `setting` of the `model_name` model.

        Parameters
        ----------
        model_name
            The name of the model to which the `setting` belongs.
        setting
            The name of the setting whose to retrieve.

        Returns
        -------
        options
            A list of options available for the `setting` and `model_name`.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        InvalidSettingError
            If the provided `setting` is not supported for the `model_name` model of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        settings = model['settings']

        try:
            settings = settings[setting]
        except KeyError:
            raise InvalidSettingError(setting, model_name, self)

        return self._get_options(settings)

    @staticmethod
    def _get_options(setting: dict[str, Union[str, dict]]) -> list[str]:
        """
        Retrieves all the possible options from ``self._models[model_name]['settings'][setting]``.

        Private method that takes the subset of the raw data in ``_models`` that corresponds to one
        setting of one model, and lists the options, ignoring the ``default_setting`` parameter.

        Parameters
        ----------
        setting
            A dictionary corresponding to one setting of one model, containing all the options.

        Returns
        -------
        options
            A list of options as found in the provided `setting` dictionary.
        """
        return [value for value in setting.keys() if value != 'default_setting']

    def default_option_for_setting(self, model_name: str, setting: str) -> str:
        """
        Returns the default option for the `setting` setting of the `model_name` model of this instrument.

        Parameters
        ----------
        model_name
            The name of the model whose `setting` to look up.
        setting
            The name of the setting whose default option to retrieve.

        Returns
        -------
        default_option
            The default option for the `setting` setting.

        Raises
        ------
        InvalidModelError
            If the provided `model_name` is not supported for this version of this instrument.
        InvalidSettingError
            If the provided `setting` is not supported for the `model_name` model of this instrument.
        """
        try:
            model = self._models[model_name]
        except KeyError:
            raise InvalidModelError(model_name, self)

        settings = model['settings']

        try:
            settings = settings[setting]
        except KeyError:
            raise InvalidSettingError(setting, model_name, self)

        return settings['default_setting']
