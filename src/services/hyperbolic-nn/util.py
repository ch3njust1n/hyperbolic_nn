import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from numpy import linalg as LA

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0


def project_hyp_vecs(x: torch.Tensor, c: float) -> torch.Tensor:
    """Project vectors into the Poincare ball.

    Args:
        x: Tensor with shape [batch, dim].
        c: Positive curvature scalar.

    Returns:
        Tensor with shape [batch, dim] and norm less than the ball radius.
    """
    max_norm = (1.0 - PROJ_EPS) / np.sqrt(c)
    norm_x = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(EPS)
    scale = (max_norm / norm_x).clamp_max(1.0)
    return x * scale


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Compute stable inverse hyperbolic tangent for positive scalars."""
    return torch.atanh(torch.clamp(x, max=1.0 - EPS))


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Compute stable hyperbolic tangent for arbitrary tensors."""
    return torch.tanh(torch.clamp(x, min=-MAX_TANH_ARG, max=MAX_TANH_ARG))


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return row-wise dot products with shape [batch, 1]."""
    return torch.sum(x * y, dim=1, keepdim=True)


def norm(x: torch.Tensor) -> torch.Tensor:
    """Return row-wise L2 norms with shape [batch, 1]."""
    return torch.linalg.norm(x, dim=1, keepdim=True)


def mob_add(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Compute Mobius addition for tensors with shape [batch, dim]."""
    adjusted_v = v + EPS
    dot_u_v = 2.0 * c * dot(u, adjusted_v)
    norm_u_sq = c * dot(u, u)
    norm_v_sq = c * dot(adjusted_v, adjusted_v)
    denominator = 1.0 + dot_u_v + norm_v_sq * norm_u_sq
    result = (1.0 + dot_u_v + norm_v_sq) / denominator * u + (
        1.0 - norm_u_sq
    ) / denominator * adjusted_v
    return project_hyp_vecs(result, c)


def mob_add_np(u: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
    """Compute NumPy Mobius addition for vectors with shape [dim]."""
    numerator = (1.0 + 2.0 * c * np.dot(u, v) + c * LA.norm(v) ** 2) * u + (
        1.0 - c * LA.norm(u) ** 2
    ) * v
    denominator = (
        1.0 + 2.0 * c * np.dot(u, v) + c**2 * LA.norm(v) ** 2 * LA.norm(u) ** 2
    )
    return numerator / denominator


def poinc_dist_sq(u: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Return squared Poincare distance for tensors with shape [batch, dim]."""
    sqrt_c = np.sqrt(c)
    mobius_difference = mob_add(-u, v, c) + EPS
    dist_poincare = 2.0 / sqrt_c * atanh(sqrt_c * norm(mobius_difference))
    return dist_poincare**2


def poinc_dist_sq_np(u: np.ndarray, v: np.ndarray, c: float) -> float:
    """Return NumPy squared Poincare distance for vectors with shape [dim]."""
    sqrt_c = np.sqrt(c)
    atanh_x = sqrt_c * LA.norm(mob_add_np(-u, v, c))
    dist_poincare = 2.0 / sqrt_c * np.arctanh(atanh_x)
    return float(dist_poincare**2)


def euclid_dist_sq(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return squared Euclidean distance for tensors with shape [batch, dim]."""
    return torch.sum(torch.square(u - v), dim=1, keepdim=True)


def euclid_dist_np(u: np.ndarray, v: np.ndarray) -> float:
    """Return NumPy Euclidean distance for vectors with shape [dim]."""
    return float(LA.norm(u - v))


def mob_scalar_mul(r: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Compute Mobius scalar multiplication for tensors with shape [batch, dim]."""
    adjusted_v = v + EPS
    norm_v = norm(adjusted_v)
    numerator = tanh(r * atanh(np.sqrt(c) * norm_v))
    result = numerator / (np.sqrt(c) * norm_v) * adjusted_v
    return project_hyp_vecs(result, c)


def mob_scalar_mul_np(r: float, v: np.ndarray, c: float) -> np.ndarray:
    """Compute NumPy Mobius scalar multiplication for vectors with shape [dim]."""
    norm_v = LA.norm(v)
    numerator = np.tanh(r * np.arctanh(np.sqrt(c) * norm_v)) * v
    return numerator / (np.sqrt(c) * norm_v)


def lambda_x(x: torch.Tensor, c: float) -> torch.Tensor:
    """Return conformal factor lambda_x with shape [batch, 1]."""
    return 2.0 / (1.0 - c * dot(x, x))


def lambda_x_np(x: np.ndarray, c: float) -> float:
    """Return NumPy conformal factor for a vector with shape [dim]."""
    return float(2.0 / (1.0 - c * LA.norm(x) ** 2))


def unit_speed_geo_np(x: np.ndarray, v: np.ndarray, t: float, c: float) -> np.ndarray:
    """Return a NumPy point on a unit-speed geodesic for vectors with shape [dim]."""
    second_term = np.tanh(np.sqrt(c) * t / 2.0) / (np.sqrt(c) * LA.norm(v)) * v
    return mob_add_np(x, second_term, c)


def exp_map_x(x: torch.Tensor, v: torch.Tensor, c: float) -> torch.Tensor:
    """Apply exponential map at x to tangent vector v, both shape [batch, dim]."""
    adjusted_v = v + EPS
    norm_v = norm(adjusted_v)
    second_term = (
        tanh(np.sqrt(c) * lambda_x(x, c) * norm_v / 2.0)
        / (np.sqrt(c) * norm_v)
        * adjusted_v
    )
    return mob_add(x, second_term, c)


def exp_map_x_np(x: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
    """Apply NumPy exponential map at x to tangent vector v, both shape [dim]."""
    second_term = (
        np.tanh(np.sqrt(c) * lambda_x_np(x, c) * LA.norm(v) / 2.0)
        / (np.sqrt(c) * LA.norm(v))
        * v
    )
    return mob_add_np(x, second_term, c)


def log_map_x(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Apply logarithmic map from x to y, both shape [batch, dim]."""
    diff = mob_add(-x, y, c) + EPS
    norm_diff = norm(diff)
    lam = lambda_x(x, c)
    return ((2.0 / np.sqrt(c)) / lam) * atanh(np.sqrt(c) * norm_diff) / norm_diff * diff


def log_map_x_np(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    """Apply NumPy logarithmic map from x to y, both shape [dim]."""
    diff = mob_add_np(-x, y, c)
    lam = lambda_x_np(x, c)
    return (
        2.0
        / (np.sqrt(c) * lam)
        * np.arctanh(np.sqrt(c) * LA.norm(diff))
        / LA.norm(diff)
        * diff
    )


def exp_map_zero(v: torch.Tensor, c: float) -> torch.Tensor:
    """Apply exponential map at the origin to tangent vectors shape [batch, dim]."""
    adjusted_v = v + EPS
    norm_v = norm(adjusted_v)
    result = tanh(np.sqrt(c) * norm_v) / (np.sqrt(c) * norm_v) * adjusted_v
    return project_hyp_vecs(result, c)


def log_map_zero(y: torch.Tensor, c: float) -> torch.Tensor:
    """Apply logarithmic map at the origin to points shape [batch, dim]."""
    diff = y + EPS
    norm_diff = norm(diff)
    return 1.0 / np.sqrt(c) * atanh(np.sqrt(c) * norm_diff) / norm_diff * diff


def mob_mat_mul(M: torch.Tensor, x: torch.Tensor, c: float) -> torch.Tensor:
    """Compute Mobius matrix multiplication for x [batch, in_dim] and M [in_dim, out_dim]."""
    adjusted_x = x + EPS
    matrix_product = torch.matmul(adjusted_x, M) + EPS
    product_norm = norm(matrix_product)
    x_norm = norm(adjusted_x)
    result = (
        1.0
        / np.sqrt(c)
        * tanh(product_norm / x_norm * atanh(np.sqrt(c) * x_norm))
        / product_norm
        * matrix_product
    )
    return project_hyp_vecs(result, c)


def mob_mat_mul_np(M: np.ndarray, x: np.ndarray, c: float) -> np.ndarray:
    """Compute NumPy Mobius matrix multiplication for x [in_dim] and M [out_dim, in_dim]."""
    matrix_product = M.dot(x)
    product_norm = LA.norm(matrix_product)
    x_norm = LA.norm(x)
    return (
        1.0
        / np.sqrt(c)
        * np.tanh(product_norm / x_norm * np.arctanh(np.sqrt(c) * x_norm))
        / product_norm
        * matrix_product
    )


def mob_pointwise_prod(x: torch.Tensor, u: torch.Tensor, c: float) -> torch.Tensor:
    """Compute diag(u) Mobius multiplication for x and u with shape [batch, dim]."""
    adjusted_x = x + EPS
    matrix_product = adjusted_x * u + EPS
    product_norm = norm(matrix_product)
    x_norm = norm(adjusted_x)
    result = (
        1.0
        / np.sqrt(c)
        * tanh(product_norm / x_norm * atanh(np.sqrt(c) * x_norm))
        / product_norm
        * matrix_product
    )
    return project_hyp_vecs(result, c)


def riemannian_gradient_c(u: torch.Tensor, c: float) -> torch.Tensor:
    """Return Poincare-ball gradient rescaling factor with shape [batch, 1]."""
    return ((1.0 - c * dot(u, u)) ** 2) / 4.0


def eucl_non_lin(eucl_h: torch.Tensor, non_lin: str) -> torch.Tensor:
    """Apply a Euclidean nonlinearity to tensors with shape [batch, dim]."""
    if non_lin == "id":
        return eucl_h
    if non_lin == "relu":
        return F.relu(eucl_h)
    if non_lin == "tanh":
        return torch.tanh(eucl_h)
    if non_lin == "sigmoid":
        return torch.sigmoid(eucl_h)
    raise ValueError("Invalid nonlinearity: " + non_lin)


def hyp_non_lin(
    hyp_h: torch.Tensor, non_lin: str, hyp_output: bool, c: float
) -> torch.Tensor:
    """Apply a nonlinearity through log_0/exp_0 for hyperbolic tensors [batch, dim]."""
    if non_lin == "id":
        if hyp_output:
            return hyp_h
        return log_map_zero(hyp_h, c)

    eucl_h = eucl_non_lin(log_map_zero(hyp_h, c), non_lin)
    if hyp_output:
        return exp_map_zero(eucl_h, c)
    return eucl_h


def setup_logger(
    name_logfile: str, logs_dir: str, also_stdout: bool = False
) -> logging.Logger:
    """Create a file logger and optionally mirror logs to stdout."""
    os.makedirs(logs_dir, exist_ok=True)
    sanitized_name = name_logfile.replace(";", "#").replace(":", "_")
    logger = logging.getLogger(sanitized_name)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(os.path.join(logs_dir, sanitized_name), mode="w")
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    if also_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
