import pint

ureg = pint.UnitRegistry()


def convert_wavenumber_to_meV_polynomial(coefficients: list[float]) -> list[float]:
    new_coefficients = []
    for i, coeff in enumerate(coefficients):
        new_coefficients.append(coeff * (1 * ureg.reciprocal_centimeter).to('meV', 'spectroscopy').magnitude ** (1 - i))

    return new_coefficients
