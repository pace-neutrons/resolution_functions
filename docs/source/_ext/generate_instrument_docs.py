import inspect
import os
import yaml

from resolution_functions.instrument import Instrument, INSTRUMENT_MAP
from resolution_functions.models import MODELS


OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'include',
    'auto-instruments'
)
MAX = int(2 ** 31)


def main():
    """Creates a page for each instrument based on its yaml data."""
    instruments = [instr for instr, (_, version) in INSTRUMENT_MAP.items() if version is None]

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    for instrument_name in instruments:
        file, _ = Instrument._get_file(instrument_name)

        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        out_path = os.path.join(OUT_DIR, f'{data["name"].lower()}-auto.rst')

        rst, data = generate_rst(data)

        rst += generate_data_section(data)

        with open(out_path, 'w') as f:
            f.write(rst)


def generate_rst(data):
    """
    Generates the automatic .rst page for an instrument, based on the yaml data.

    Parameters
    ----------
    data
        The data read out of the instrument's yaml file.

    Returns
    -------
    out
        The nicely rst formatted page.
    """
    out = 'Versions\n********\n\n'

    default_version = data['default_version']
    for version_name in data['version'].keys():
        if version_name == default_version:
            out += f'* :ref:`{version_name}-version` (default)\n'
        else:
            out += f'* :ref:`{version_name}-version`\n'

    out += '\n'

    for version_name, version_data in data['version'].items():
        link = f'{version_name}-data'
        out += f'.. _{version_name}-version:\n\n' \
               f'{version_name}\n' \
               f'{len(version_name) * "="}\n\n' \
               f'For details on the parameters associated with the {version_name} version, please' \
               f' see the :iref:ref:`{version_name} data<{link}>`.\n\n' \
                'Models\n------\n\n'

        version_data['sphinx_link'] = link

        default_model = version_data['default_model']
        for model_name in version_data['models'].keys():
            if model_name == default_model:
                out += f'* :ref:`{version_name}-{model_name}-model` (default)\n'
            else:
                out += f'* :ref:`{version_name}-{model_name}-model`\n'

        out += '\n'

        for model_name, model_data in version_data['models'].items():
            cls = MODELS[model_data["function"]]
            module = inspect.getmodule(cls)
            link = f'{version_name}-{model_name}-data'

            out += f'.. _{version_name}-{model_name}-model:\n\n' \
                   f'{model_name}\n' \
                   f'{len(model_name) * "^"}\n\n' \
                   f'This section contains the data for the {version_name} instrument associated ' \
                   f'with the {model_name} model. For more information about how the model works ' \
                   f'and its implementation, please see ' \
                   f':py:class:`{module.__name__}.{cls.__name__}`. For more information on the ' \
                   f'model parameters, please see :iref:ref:`{model_name} model data<{link}>`.\n\n'

            model_data['sphinx_link'] = link

            if not model_data['configurations']:
                out += 'Configurations: NONE\n\n'
                continue
            else:
                out += 'Configurations:\n\n'

            for config_name, config_data in model_data['configurations'].items():
                config_link = f'{version_name}-{model_name}-{config_name}-data'
                out += f'* :iref:ref:`{config_name}<{config_link}>`\n\n'

                default_option = config_data['default_option']
                for option_name in config_data.keys():
                    link = f'{version_name}-{model_name}-{config_name}-{option_name}-data'

                    if option_name == 'default_option':
                        continue
                    else:
                        out += f'  * :iref:ref:`{option_name}<{link}>`'
                        config_data[option_name]['sphinx_link'] = link

                        if option_name == default_option:
                            out +=  ' (default)\n'
                        else:
                            out += '\n'

                config_data['sphinx_link'] = config_link

                out += '\n'

    return out, data


def generate_data_section(data_with_links: dict, target_role: str = ':iref:target:'):
    """
    Generates the data section of an instrument's documentation, effectively reproducing
    `yaml.dump` but with sphinx roles injected to serve as targets for links.

    Parameters
    ----------
    data_with_links
        The yaml data. but with the link names inserted.
    target_role
        The sphinx role to use to mark the target links.

    Returns
    -------
    out
        A string containing the nicely formatted data.
    """
    out = '\nData\n****\n\n' \
          '.. parsed-code-block:: yaml\n\n'

    for key, value in data_with_links.items():
        if key == 'version':
            continue

        out += f'    {key}: {format_value(value)}\n'

    out += '    version:\n'

    for version_name, version_data in data_with_links['version'].items():
        link = version_data['sphinx_link']
        out += f'        {target_role}`{version_name}<{link}>`:\n' \
               f'            default_model: "{version_data["default_model"]}"\n' \
                '            models:\n'

        for model_name, model_data in version_data['models'].items():
            link = model_data['sphinx_link']
            out += ' ' * 16 + f'{target_role}`{model_name}<{link}>`:\n'

            for key, value in model_data.items():
                if key in ['parameters', 'configurations', 'sphinx_link']:
                    continue

                out += ' ' * 20 + f'{key}: {format_value(value)}\n'

            out += ' ' * 20 + 'parameters:\n'

            out += add_parameters(model_data['parameters'], 24)

            out += ' ' * 20 + f'configurations:'

            if not model_data['configurations']:
                out += ' {}\n'
                continue

            out += '\n'

            for config_name, config_data in model_data['configurations'].items():
                link = config_data['sphinx_link']
                out += ' ' * 24 + f'{target_role}`{config_name}<{link}>`:\n'
                out += ' ' * 28 + f'default_option: "{config_data["default_option"]}"\n'

                for option_name, option_data in config_data.items():
                    if option_name in ['sphinx_link', 'default_option']:
                        continue

                    link = option_data['sphinx_link']
                    out += ' ' * 28 + f'{target_role}`{option_name}<{link}>`:\n'

                    for key, value in option_data.items():
                        if key == 'sphinx_link':
                            continue

                        out += ' ' * 32 + f'{key}: {format_value(value)}\n'

    return out


def format_value(value):
    """Puts strings into quotes"""
    return f'"{value}"' if isinstance(value, str) else value


def add_parameters(parameters: dict, depth: int = 4) -> str:
    """
    Recursively adds all parameters, in a yaml style.

    Parameters
    ----------
    parameters
        The data representing a model parameters.
    depth
        The number of spaces to put at the first level of indentation.

    Returns
    -------
    out
        The string containing all the nicely formatted parameters.
    """
    out = ''
    for parameter_name, parameter_data in parameters.items():
        if isinstance(parameter_data, dict):
            out += ' ' * depth + f'{parameter_name}:\n'
            out += add_parameters(parameter_data, depth + 4)
        else:
            out += ' ' * depth + f'{parameter_name}: {format_value(parameter_data)}\n'

    return out


if __name__ == '__main__':
    main()
