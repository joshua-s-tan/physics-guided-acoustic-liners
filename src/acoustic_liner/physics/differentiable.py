# -*- coding: utf-8 -*-
r"""cVAE v2.7 --- DIFFERENTIABLE re-implementation of the reduced-order oracle forward path.

Per sec.5 of the historical cVAE v2.7 pre-registration (not distributed). This re-implements, in PyTorch (float64), the FORWARD path of
`oracle.label_v2` for a FIXED design so that gradients of `dOASPL_worst` and `total_installed_mass` w.r.t. the
decoder vector (and hence, through the frozen decoder, w.r.t. the latent z) are available by autograd.

It mirrors the numpy oracle exactly (verified against `oracle.label_v2`: tests/unit/test_v27_oracle_torch.py):
  z_surface -> diffuse_alpha (fixed Gauss-Legendre angle quadrature) -> interp to the 8 band centers ->
  robust_delta_oaspl (worst-case MIN over the FIXED alpha_bare x kappa grid) ; and the corrected shared-wall mass.

It does NOT re-implement the coverage root-finders (`min_coverage_for_target` / `_golden_max` / bisection):
`f_cov` is a decoder output, so those are OUTSIDE the gradient path (prereg sec.5).

Input contract: a DVEC-layout decoder vector (dataset.DVEC_LAYOUT, 19-D, PHYSICAL values --- the exact inverse
`dataset.design_from_decoder_vector`), plus the condition (spl_dB[8], A_wall, kappa_nominal). NO box clipping is
applied (the `design_from_decoder_vector` path does not clip), so there are no clip kinks from the design box.
"""
from __future__ import annotations

import numpy as np
import torch

from . import resonator as RM  # RHO, C
from . import liner as OR  # Z_AIR, DESIGN_BAND_HZ, MAX_DEPTH_M, MIN_NECK_R_M
from . import mass as MM  # RHO_WALL, InstallBOM defaults
from ..design import schema as DS  # N_GROUPS, Q_FIXED
from . import oracle as O2  # ALPHA_BARE_RANGE/NOMINAL, DELTA_KAPPA_DEFAULT, _grid_with_node

# --- constants (pulled from the frozen sources; never re-typed by hand) ------------------------------ #
_RHO = float(RM.RHO)
_C = float(RM.C)
_Z_AIR = float(OR.Z_AIR)
_RHO_WALL = float(MM.RHO_WALL)
_Q = float(DS.Q_FIXED)
_NG = int(DS.N_GROUPS)
_BANDS = np.array(OR.DESIGN_BAND_HZ, dtype=float)                       # [8]
_INSTALL_AREAL = float(MM.InstallBOM().install_areal_mass())            # 0.30 (adhesive+fastener+frame+margin)
_END_CORR = 2.0 * 0.85                                                  # both_sides Rayleigh end correction: 1.7*r
_N_ANGLE = 48
_N_A = 9
_N_K = 5

# fine frequency grid (identical to oracle.liner_diffuse_alpha_band)
_FINE = np.linspace(OR.DESIGN_BAND_HZ[0] * 0.9, OR.DESIGN_BAND_HZ[-1] * 1.05, 900)     # [900]

# Gauss-Legendre angle nodes on [0, pi/2] (identical to oracle._gl_angle_nodes)
_gx, _gw = np.polynomial.legendre.leggauss(_N_ANGLE)
_THETA = (np.pi / 4.0) * (_gx + 1.0)
_WTHETA = (np.pi / 4.0) * _gw


def _interp_indices(xp: np.ndarray, x: np.ndarray):
    """Linear-interpolation weights for np.interp(x, xp, .) with xp ascending and every x inside [xp0, xp-1].
    Returns (i0, i1, w1) so that y(x) = yp[i0]*(1-w1) + yp[i1]*w1."""
    i1 = np.searchsorted(xp, x, side="left")
    i1 = np.clip(i1, 1, len(xp) - 1)
    i0 = i1 - 1
    w1 = (x - xp[i0]) / (xp[i1] - xp[i0])
    return i0, i1, w1


_I0, _I1, _W1 = _interp_indices(_FINE, _BANDS)


class TorchOracle:
    """Holds the fixed (dtype, device) tensors and evaluates the differentiable oracle."""

    def __init__(self, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        self.fine = t(_FINE)                                   # [900]
        self.omega_fine = t(2.0 * np.pi * _FINE)               # [900]
        self.cos_theta = t(np.cos(_THETA))                     # [48]
        self.sin2theta = t(np.sin(2.0 * _THETA))               # [48]
        self.wtheta = t(_WTHETA)                               # [48]
        self.i0 = torch.as_tensor(_I0, dtype=torch.long, device=device)
        self.i1 = torch.as_tensor(_I1, dtype=torch.long, device=device)
        self.w1 = t(_W1)                                       # [8]

    # ------------------------------------------------------------------ #
    def _z_surface(self, D, f0, r, L, frac):
        """Complex specific surface impedance z_s(f) over the fine grid. All args are tensors; f0/r/L/frac
        are [NG]. Returns complex [900]."""
        cdt = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        S = np.pi * r ** 2                                     # [NG]
        L_eff = L + _END_CORR * r                              # [NG]
        k = 2.0 * np.pi * f0 / _C                              # [NG]
        V_cav = S / (L_eff * k ** 2)                           # [NG]
        M_a = _RHO * L_eff / S                                 # [NG]
        C_a = V_cav / (_RHO * _C ** 2)                         # [NG]
        omega0 = 1.0 / torch.sqrt(M_a * C_a)                   # [NG]
        R_r = omega0 * M_a / _Q                                # [NG]
        density = frac * D / torch.clamp(V_cav, min=1e-9)      # [NG]
        omega = self.omega_fine.unsqueeze(0)                   # [1,900]
        Ma = M_a.unsqueeze(1); Ca = C_a.unsqueeze(1); Rr = R_r.unsqueeze(1)   # [NG,1]
        reactance = omega * Ma - 1.0 / (omega * Ca)            # [NG,900]
        Z_r = torch.complex(Rr.expand_as(reactance), reactance)               # [NG,900] complex
        beta = (density.unsqueeze(1).to(cdt) / Z_r).sum(dim=0)                # [900] complex
        z_s = 1.0 / beta                                       # [900] complex
        return z_s, V_cav

    def _diffuse_alpha_band(self, z_s):
        """Diffuse (random-incidence) absorption at the 8 band centers. z_s complex [900] -> real [8]."""
        zeta = z_s / _Z_AIR                                    # [900] complex
        zc = zeta.unsqueeze(1) * self.cos_theta.unsqueeze(0).to(zeta.dtype)   # [900,48] complex
        Rc = (zc - 1.0) / (zc + 1.0)
        a = 1.0 - (Rc.abs() ** 2)                              # [900,48] real
        raw = (a * self.sin2theta.unsqueeze(0)) @ self.wtheta  # [900] real (Paris quadrature)
        raw = torch.clamp(raw, 0.0, 1.0)                       # roundoff clip (identity for passive designs)
        a_band = raw[self.i0] * (1.0 - self.w1) + raw[self.i1] * self.w1      # [8] linear interp
        return a_band

    def _delta_oaspl(self, nr_band, w_energy):
        """Spectrum-weighted OASPL reduction [dB]. nr_band [8]; w_energy [8] precomputed energy weights."""
        return -10.0 * torch.log10(torch.sum(w_energy * 10.0 ** (-nr_band / 10.0)))

    def evaluate(self, dvec, spl_dB, A_wall, kappa_nominal,
                 delta_kappa=None, return_parts=False):
        """dvec: tensor [19] (DVEC layout, physical). spl_dB: array/tensor [8]. A_wall,kappa scalars.
        Returns (dOASPL_worst, m_total) as scalar tensors (differentiable in dvec)."""
        if delta_kappa is None:
            delta_kappa = float(O2.DELTA_KAPPA_DEFAULT)
        dt, dev = self.dtype, self.device
        D = dvec[0]; face_t = dvec[1]; wall_t = dvec[2]
        f0 = torch.stack([dvec[3 + 3 * g] for g in range(_NG)])
        r = torch.stack([dvec[4 + 3 * g] for g in range(_NG)])
        L = torch.stack([dvec[5 + 3 * g] for g in range(_NG)])
        u = torch.stack([dvec[3 + 3 * _NG + j] for j in range(_NG - 1)])      # [NG-1] area log-ratios
        z_logits = torch.cat([u, torch.zeros(1, dtype=dt, device=dev)])       # softmax([u1,u2,u3,0])
        frac = torch.softmax(z_logits, dim=0)                                 # [NG]
        f_cov = dvec[3 + 3 * _NG + (_NG - 1)]                                 # dvec[18]

        z_s, V_cav = self._z_surface(D, f0, r, L, frac)
        a_band = self._diffuse_alpha_band(z_s)                                # [8]

        # --- robust worst-case dOASPL over the FIXED (alpha_bare, kappa) grid (oracle.robust_delta_oaspl) ---
        A_wall_t = torch.as_tensor(float(A_wall), dtype=dt, device=dev)
        spl = torch.as_tensor(np.asarray(spl_dB, dtype=float), dtype=dt, device=dev)
        e = 10.0 ** ((spl - spl.max()) / 10.0)
        w_energy = e / e.sum()                                                # [8] energy weights (const in dvec)
        kn = float(kappa_nominal)
        k_lo = max(0.0, kn - delta_kappa); k_hi = kn + delta_kappa
        a_grid = O2._grid_with_node(np.linspace(O2.ALPHA_BARE_RANGE[0], O2.ALPHA_BARE_RANGE[1], _N_A),
                                    O2.ALPHA_BARE_NOMINAL)
        k_grid = (np.array([kn]) if k_hi <= k_lo
                  else O2._grid_with_node(np.linspace(k_lo, k_hi, _N_K), kn))
        S_wall = A_wall_t
        S_treated = f_cov * S_wall
        ds = []
        for ab in a_grid:
            for kk in k_grid:
                A_other = float(kk) * float(A_wall)
                A0 = float(ab) * S_wall + A_other
                A1 = a_band * S_treated + float(ab) * (S_wall - S_treated) + A_other
                nr = 10.0 * torch.log10(A1 / A0)                             # [8]
                ds.append(self._delta_oaspl(nr, w_energy))
        d_all = torch.stack(ds)
        dOASPL_worst = torch.min(d_all)
        # gap between the worst and 2nd-worst grid value: near-zero => the robust MIN is (near-)tied, i.e. the
        # design sits on a worst-case-switch kink where the objective is non-differentiable (prereg sec.5).
        if d_all.numel() >= 2:
            two_lowest = torch.topk(d_all, 2, largest=False).values
            robust_gap = float((two_lowest[1] - two_lowest[0]).detach())
        else:
            robust_gap = float("inf")

        # --- corrected shared-wall total installed mass (physics.mass) ---
        m_areal = _RHO_WALL * face_t                                          # facesheet (backing_t=0)
        n_local = D / torch.clamp(V_cav, min=1e-9)                            # cells/m^2 within each patch
        partition_areal = _RHO_WALL * wall_t * D * 2.0 * torch.sqrt(n_local)  # shared-wall coeff 2.0 [NG]
        m_areal = m_areal + torch.sum(frac * partition_areal)
        A_t = f_cov * A_wall_t
        m_total = m_areal * A_t + _INSTALL_AREAL * A_t                        # m_liner + m_install (m_fixed=0)

        if return_parts:
            return dOASPL_worst, m_total, dict(a_band=a_band, frac=frac, V_cav=V_cav, robust_gap=robust_gap)
        return dOASPL_worst, m_total


def feasible(dvec) -> bool:
    """Non-differentiable feasibility gate mirroring oracle.label_v2 (depth/neck/area). For the polish to
    reject infeasible decoded designs; NOT part of the gradient path."""
    x = dvec.detach().cpu().numpy() if torch.is_tensor(dvec) else np.asarray(dvec, float)
    D = float(x[0])
    r = [float(x[4 + 3 * g]) for g in range(_NG)]
    u = [float(x[3 + 3 * _NG + j]) for j in range(_NG - 1)]
    z = np.array(u + [0.0]); e = np.exp(z - z.max()); fr = e / e.sum()
    feas_depth = D <= OR.MAX_DEPTH_M + 1e-9
    feas_neck = all(rr >= OR.MIN_NECK_R_M for rr in r)
    feas_area = abs(float(fr.sum()) - 1.0) < 0.02
    return bool(feas_depth and feas_neck and feas_area)
