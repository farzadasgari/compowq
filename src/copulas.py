"""
src/copulas.py

A small library of bivariate copula families used to link SAPEI and STI in
src.compute_scdhi.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, multivariate_normal, norm

EPS = 1e-6


def _clip(u, v):
    return np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)


# Frank
def _frank_density(u, v, theta):
    if abs(theta) < 1e-6:
        return np.ones_like(u)
    num = theta * (1 - np.exp(-theta)) * np.exp(-theta * (u + v))
    denom = ((1 - np.exp(-theta)) - (1 - np.exp(-theta * u))
             * (1 - np.exp(-theta * v))) ** 2
    return num / denom


def frank_cdf(u, v, params):
    theta = params["theta"]
    u, v = _clip(u, v)
    if abs(theta) < 1e-6:
        return u * v
    return -1 / theta * np.log(
        1 + (np.exp(-theta * u) - 1) *
        (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1)
    )


def frank_loglik(u, v, params):
    u, v = _clip(u, v)
    d = np.clip(_frank_density(u, v, params["theta"]), 1e-300, None)
    return float(np.sum(np.log(d)))


def frank_fit(u, v):
    u, v = _clip(u, v)

    def neg_ll(theta):
        return -np.sum(np.log(np.clip(_frank_density(u, v, theta), 1e-300, None)))

    result = minimize_scalar(neg_ll, bounds=(-30, 30), method="bounded")
    return {"theta": float(result.x)}


FRANK = {"fit": frank_fit, "loglik": frank_loglik,
         "cdf": frank_cdf, "n_params": 1}


# Gaussian
def gaussian_cdf(u, v, params):
    rho = params["rho"]
    u, v = _clip(u, v)
    x, y = norm.ppf(u), norm.ppf(v)
    mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    pts = np.column_stack([x, y])
    return np.array([mvn.cdf(p) for p in pts])


def gaussian_loglik(u, v, params):
    rho = params["rho"]
    u, v = _clip(u, v)
    x, y = norm.ppf(u), norm.ppf(v)
    density = (1 / np.sqrt(1 - rho**2)) * np.exp(
        -(rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
    )
    return float(np.sum(np.log(np.clip(density, 1e-300, None))))


def gaussian_fit(u, v):
    u, v = _clip(u, v)
    x, y = norm.ppf(u), norm.ppf(v)
    rho = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.clip(rho, -0.999, 0.999))
    return {"rho": rho}


GAUSSIAN = {"fit": gaussian_fit, "loglik": gaussian_loglik,
            "cdf": gaussian_cdf, "n_params": 1}


# Clayton / Gumbel base
def _clayton_cdf_base(u, v, theta):
    return np.maximum(u ** (-theta) + v ** (-theta) - 1, 1e-12) ** (-1 / theta)


def _clayton_density_base(u, v, theta):
    return (
        (theta + 1)
        * (u * v) ** (-theta - 1)
        * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 2)
    )


def _clayton_tau_to_theta(tau):
    return 2 * tau / max(1 - tau, 1e-6)


def _gumbel_cdf_base(u, v, theta):
    A, B = -np.log(u), -np.log(v)
    w = A**theta + B**theta
    return np.exp(-(w ** (1 / theta)))


def _gumbel_density_base(u, v, theta):
    A, B = -np.log(u), -np.log(v)
    w = A**theta + B**theta
    C = np.exp(-(w ** (1 / theta)))
    return (
        C / (u * v) * (A * B) ** (theta - 1) *
        w ** (1 / theta - 2) * (w ** (1 / theta) + theta - 1)
    )


def _gumbel_tau_to_theta(tau):
    return 1 / max(1 - tau, 1e-6)


def _make_rotated_family(base_cdf, base_density, tau_to_theta_start):
    def cdf(u, v, params):
        u, v = _clip(u, v)
        theta = params["theta"]
        if params["rotated"]:
            return v - base_cdf(1 - u, v, theta)
        return base_cdf(u, v, theta)

    def loglik(u, v, params):
        u, v = _clip(u, v)
        theta = params["theta"]
        uu = 1 - u if params["rotated"] else u
        d = np.clip(base_density(uu, v, theta), 1e-300, None)
        return float(np.sum(np.log(d)))

    def fit(u, v):
        u, v = _clip(u, v)
        tau = kendalltau(u, v)[0]
        rotated = tau < 0
        uu = 1 - u if rotated else u
        theta0 = max(tau_to_theta_start(abs(tau)), 1e-3)

        def neg_ll(theta):
            return -np.sum(np.log(np.clip(base_density(uu, v, max(theta, 1e-6)), 1e-300, None)))

        result = minimize_scalar(neg_ll, bounds=(1e-6, 50), method="bounded")
        return {"theta": float(result.x), "rotated": bool(rotated), "_theta0": theta0}

    return {"fit": fit, "loglik": loglik, "cdf": cdf, "n_params": 1}


CLAYTON = _make_rotated_family(
    _clayton_cdf_base, _clayton_density_base, _clayton_tau_to_theta)
GUMBEL = _make_rotated_family(
    _gumbel_cdf_base, _gumbel_density_base, _gumbel_tau_to_theta)

FAMILIES = {"frank": FRANK, "gaussian": GAUSSIAN,
            "clayton": CLAYTON, "gumbel": GUMBEL}


def compare_families(u: np.ndarray, v: np.ndarray, families: dict = None) -> list:
    families = families or FAMILIES
    results = []
    for name, fam in families.items():
        try:
            params = fam["fit"](u, v)
            ll = fam["loglik"](u, v, params)
            aic = 2 * fam["n_params"] - 2 * ll
            results.append({"family": name, "params": params,
                           "loglik": ll, "aic": aic})
        except Exception as exc:
            results.append({"family": name, "params": None,
                           "loglik": None, "aic": np.inf, "error": str(exc)})
    return sorted(results, key=lambda r: r["aic"])
