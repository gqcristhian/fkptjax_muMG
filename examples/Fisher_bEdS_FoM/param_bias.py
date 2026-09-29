"""Parameter-bias test: take the bEdS-full prediction at fiducial parameters
as synthetic truth, and ask how much EdS / bEdS-fkPT would mis-recover the
fiducial parameters if fit to that truth (standard linear Fisher-bias
formula, e.g. Taylor, Kitching & Huff 2007; Amara & Refregier 2008):

    DeltaP = P_true(theta_fid) - P_approx(theta_fid)
    Delta_theta_i = (F_approx^-1)_ij [dP_approx/dtheta_j]^T C^-1 DeltaP
"""

import numpy as np

import config
import derivatives
import fisher as fisher_mod


def parameter_bias(theory_approx, cov_inv, stage="(e) full nuisance model"):
    """Fisher-bias vector for one approximate theory ("eds" or "approx"),
    evaluated with the parameter set active at the given staged-marginalization
    stage (default: the full nuisance model).
    """
    params = config.params_in_groups(config.STAGES[stage])

    P_true = derivatives.fiducial_poles("full")
    P_approx = derivatives.fiducial_poles(theory_approx)
    DeltaP = derivatives.stacked_poles(P_true) - derivatives.stacked_poles(P_approx)

    derivs = derivatives.all_derivatives(theory_approx, params=params)
    F = fisher_mod.fisher_matrix(derivs, cov_inv, params)
    F = fisher_mod.add_priors(F, params)
    Finv = np.linalg.inv(F)

    D = np.stack([derivs[p] for p in params], axis=0)
    rhs = D @ cov_inv @ DeltaP
    delta_theta = Finv @ rhs

    i_mu0 = params.index(config.MG_PARAM_NAME)
    sigma_mu0 = float(np.sqrt(Finv[i_mu0, i_mu0]))

    return dict(
        params=params,
        delta_theta={p: float(v) for p, v in zip(params, delta_theta)},
        sigma_mu0=sigma_mu0,
        delta_mu0_over_sigma=float(delta_theta[i_mu0] / sigma_mu0),
        DeltaP=DeltaP,
        Finv=Finv,
    )
