import numpy as np


def calculate_tosca_book_delta_theta_b(theta_b: float, eta_g: float, ws: float, wd: float, dg: float) -> float:
    temp = (1 + (1 / np.tan(np.deg2rad(theta_b))) ** 2) / dg
    alpha_2 = ws * 0.5 * temp
    alpha_3 = wd * 0.5 * temp

    numerator =  np.sqrt(alpha_2 ** 2 * alpha_3 ** 2 + alpha_2 ** 2 * eta_g ** 2 + alpha_3 ** 2 * eta_g ** 2)
    denominator = np.sqrt(alpha_2 ** 2 + alpha_3 ** 2 + 4 * eta_g ** 2)

    return numerator / denominator
