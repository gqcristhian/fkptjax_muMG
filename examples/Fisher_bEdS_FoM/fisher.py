"""Fisher-matrix assembly, staged marginalization, correlations, and the
2-parameter ellipse geometry used for plotting.

    F_ij = dP/dtheta_i^T C^-1 dP/dtheta_j + F_prior

``F_prior`` is diagonal, from ``config.PARAMS[name]["prior_sigma"]`` (see that
module's docstring for the placeholder-prior caveat).
"""

import numpy as np

import config


def fisher_matrix(derivs, cov_inv, params):
    """F_ij (no priors) for a given ordered parameter list, from a
    {param: dP/dtheta array} dict and the (already-inverted) data covariance.
    """
    D = np.stack([derivs[p] for p in params], axis=0)  # (nparam, ndata)
    return D @ cov_inv @ D.T


def prior_matrix(params):
    return np.diag([1.0 / config.PARAMS[p]["prior_sigma"] ** 2 for p in params])


def add_priors(F, params):
    return F + prior_matrix(params)


def sigma_and_corr(Finv, params, target=config.MG_PARAM_NAME):
    """From an already-inverted Fisher matrix: (sigma_target, {param: rho})."""
    i = params.index(target)
    sigma_target = np.sqrt(Finv[i, i])
    corr = {}
    for j, p in enumerate(params):
        if p == target:
            continue
        sigma_p = np.sqrt(Finv[j, j])
        corr[p] = float(Finv[i, j] / (sigma_target * sigma_p))
    return float(sigma_target), corr


def staged_results(derivs_by_param, cov_inv, stages=config.STAGES, target=config.MG_PARAM_NAME):
    """Run every staged-marginalization case (a)-(e). Returns
    {stage_name: dict(params, F, Finv, sigma_mu0, corr)}.
    """
    out = {}
    for stage_name, groups in stages.items():
        params = config.params_in_groups(groups)
        F = fisher_matrix(derivs_by_param, cov_inv, params)
        F = add_priors(F, params)
        Finv = np.linalg.inv(F)
        sigma_mu0, corr = sigma_and_corr(Finv, params, target=target)
        out[stage_name] = dict(params=params, F=F, Finv=Finv, sigma_mu0=sigma_mu0, corr=corr)
    return out


def ellipse_sub(Finv, params, p1, p2):
    """2x2 marginalized covariance sub-block for (p1, p2) from a full Finv,
    plus its eigen-decomposition (semi-axes/angle) for plotting.
    """
    i, j = params.index(p1), params.index(p2)
    sub = np.array([[Finv[i, i], Finv[i, j]], [Finv[j, i], Finv[j, j]]])
    sigma1, sigma2 = np.sqrt(sub[0, 0]), np.sqrt(sub[1, 1])
    rho = sub[0, 1] / (sigma1 * sigma2)
    eigvals, eigvecs = np.linalg.eigh(sub)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return dict(sigma1=float(sigma1), sigma2=float(sigma2), rho=float(rho),
                semi_major=float(np.sqrt(eigvals[0])), semi_minor=float(np.sqrt(eigvals[1])),
                angle_deg=angle, sub=sub)
