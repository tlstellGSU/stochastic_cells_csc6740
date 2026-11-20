#!/usr/bin/env python3
"""Stochastic cell walls: from public flicker videos to a fitted Ornstein-Uhlenbeck wall model, and
what that wall's motion does to transport inside the cell.

CSC 6740 class project, single-file version of the brownian_cells repository pipeline.
Tommy Stell, Georgia State University, 2026-11-19.

QUESTIONS
  Q1 (inverse)  What can 1 kHz bright-field videos of red blood cells tell us about the stochastic law
                of the cell wall? Is a passive membrane enough, or is a colored (active) forcing needed,
                and which physical parameters can the videos actually determine?
  Q2 (forward)  With the fitted law, does the wall's motion change how long a molecule takes to travel
                from a membrane channel to the cell centre while it decays on the way, with the cytosol
                carried by the wall? (Frozen-wall models assume it does not.)

PIPELINE (one section of this file per data-mining stage; names follow the IEEE report)
  1 Data            Caselli et al. 2024 deposit: 11 videos (6 free, 5 optically trapped red blood
                    cells), 300 x 300 px at 47.2 nm/px, 1 kHz, 1 ms exposure, 5000 frames each.
  2 Preprocessing   Level-crossing contour tracker on 2048 rays; validation on synthetic frames
                    rendered from each video's own radial profile and pixel noise (static: noise
                    floor; moving: gain); tracking-noise bound from the 1 ms MSD.
  3 Analysis        Features: Fourier contour modes h_q(t), q = 2..6, and their periodograms.
                    Exploration: is the motion Gaussian and does it have memory? Model: every 3D
                    shape mode is an OU process (passive P) or an OU driven by a colored OU forcing
                    (active A); the camera sees the equatorial projection, exposure-averaged, plus
                    white noise. Fits: Whittle likelihood, natural parameters.
  4 Evaluation      95% profile-likelihood intervals (identifiability); Delta AIC = AIC(P) - AIC(A)
                    calibrated by a parametric bootstrap under three passive generators; a
                    goodness-of-fit statistic on the binned periodogram.
  5 Interpretation  Report with the answers to Q1 and Q2, and the transport simulation: paired
                    molecules in a moving and a frozen copy of a fitted wall (first passage to
                    r < 0.3 R, first-order decay applied exactly afterwards).

USAGE (from any directory; the data path defaults to ./data next to this file, see README.md)
  python3 stochastic_cells.py --quick              # whole pipeline, reduced sizes (two videos, 1000 frames)
  python3 stochastic_cells.py                      # full run (several hours on 24 cores)
  python3 stochastic_cells.py --stages fit profile # selected stages; each stage caches its output
  Stages, in order: track explore fit profile select null transport report.

OUTPUTS (--out, default ./output next to this file)
  tracks/<state>_<cell>.npz, explore.json, fits.json, profiles.json, selection.json, null.json,
  transport.json, report.txt, figures/fig1_data.png ... fig6_transport.png

DEPENDENCIES  Python 3.12, numpy 1.26, scipy 1.11, matplotlib 3.6, Pillow 10 (versions verified).

NOT REPRODUCED HERE (see the repository): the grid advection-diffusion scheme behind the share-at-a-
fixed-time metric Phi (exp27), neural posterior estimation (exp23/26), kick-injection tests (exp8), the
Dryad contour data (exp40-42). Fits here follow one protocol (binned Whittle, l_max 40, then unbinned
l_max 80 for Delta AIC), so numbers can differ slightly from the repository's exp9/exp14/exp15/exp28.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one BLAS thread per worker process

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.linalg import expm, solve_continuous_lyapunov  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import gammaln  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_OUT = HERE / "output"

FRAME_DT = 1e-3  # s; frame period = exposure time (1 kHz, 1 ms)
NM_PER_PX = 47.2  # Caselli et al. 2024 analysis code
TEMPERATURE = 310.15  # K, held fixed in every fit
K_B = 1.380649e-23  # J/K
FIT_MODES = (2, 3, 4, 5, 6)  # contour modes fitted and used for the transport walls
BIN = 8  # periodogram ordinates per bin in the binned Whittle likelihood
L_MAX_FIT = 40  # highest 3D mode in the fitted spectra
L_MAX_SELECT = 80  # ... in the unbinned likelihood used for Delta AIC
L_MAX_WALL = 30  # ... in the simulated transport walls
SPREAD_MAX = 2.0  # bound on the log-normal rate spread s of generator G2

FREE_CELLS = ("RBC1", "RBC2", "RBC3", "RBC10", "RBC11", "RBC13")
TRAPPED_CELLS = ("RBC1", "RBC2", "RBC4", "RBC10", "RBC13")
NULL_CELLS = (("RBC2", "free"), ("RBC10", "free"), ("RBC3", "free"), ("RBC4", "trapped"))

# Natural parameters (Section 3): tension = sigma R^2 / kT and confinement = gamma R^4 / kT replace the
# reduced sigma_bar and gamma_bar, which removes a likelihood ridge kappa -> 0, sigma_bar -> infinity.
PARAMS = {
    "P": ["kappa_kT", "tension", "eta", "confinement", "e2"],
    "A": ["kappa_kT", "tension", "eta", "confinement", "a", "tau_a", "e2"],
    "G1": ["kappa_kT", "tension", "eta", "confinement", "e2"],
    "G2": ["kappa_kT", "tension", "eta", "confinement", "e2", "s"],
}
UNITS = {
    "kappa_kT": "k_BT",
    "tension": "",
    "eta": "Pa s",
    "confinement": "",
    "a": "x thermal",
    "tau_a": "s",
    "e2": "m^2",
    "s": "",
}


# =====================================================================================================
# 1. DATA
# =====================================================================================================
def all_videos():
    """Every (cell, state) video in the deposit. Only RBC1, 2, 10, 13 were recorded both free and trapped."""
    return [(c, "free") for c in FREE_CELLS] + [(c, "trapped") for c in TRAPPED_CELLS]


def video_folder(data_root, cell, state):
    if state == "free":
        return Path(data_root) / "Free_cells" / cell
    return Path(data_root) / "Trapped Cells" / f"Trapped_{cell}"


def frame_paths(folder):
    return sorted(glob.glob(str(Path(folder) / "*.tif")))


def load_frames(folder, start=0, count=None):
    """Grayscale float32 stack (frames, height, width); the deposit's TIFFs are RGB copies of one channel."""
    paths = frame_paths(folder)[start : None if count is None else start + count]
    return np.stack([np.asarray(Image.open(p), np.float32).mean(axis=2) for p in paths])


def video_label(cell, state):
    return f"{state} {cell}"


# =====================================================================================================
# 2. PREPROCESSING: contour tracking and its validation
# =====================================================================================================
def fit_circle(x, y):
    """Algebraic least-squares (Kasa) circle through points (x, y): returns (cx, cy, r)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    (a0, a1, a2), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy = a0 / 2, a1 / 2
    return cx, cy, np.sqrt(a2 + cx**2 + cy**2)


def hough_centre(mean_img, r_range=(45, 100), smooth=2.0, top=0.05):
    """Cell centre by gradient-direction Hough voting (robust to dust specks), and the radius of the
    strongest outward dark-to-bright step of the angle-median radial profile."""
    gy = ndimage.gaussian_filter(mean_img, smooth, order=(1, 0))
    gx = ndimage.gaussian_filter(mean_img, smooth, order=(0, 1))
    g = np.hypot(gx, gy)
    yy, xx = np.nonzero(g > np.quantile(g, 1 - top))
    ux, uy = gx[yy, xx] / g[yy, xx], gy[yy, xx] / g[yy, xx]
    height, width = mean_img.shape
    votes = np.zeros((height, width))
    for r in np.arange(*r_range, 1.0):
        for sign in (1, -1):
            cx = np.rint(xx - sign * r * ux).astype(int)
            cy = np.rint(yy - sign * r * uy).astype(int)
            ok = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
            np.add.at(votes, (cy[ok], cx[ok]), 1)
    cy, cx = np.unravel_index(np.argmax(ndimage.gaussian_filter(votes, 2)), votes.shape)
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = np.arange(*r_range, 0.25)
    profile = np.median(polar_samples(mean_img, cx, cy, theta, r, order=1), axis=0)
    r0 = r[np.argmax(np.gradient(ndimage.gaussian_filter1d(profile, 2)))]
    return float(cx), float(cy), float(r0)


def polar_samples(img, cx, cy, theta, r, order=3):
    """Image sampled on rays theta (rows) at radii r (columns) about (cx, cy)."""
    return ndimage.map_coordinates(img, [cy + np.outer(np.sin(theta), r), cx + np.outer(np.cos(theta), r)], order=order)


def subpixel_peak(g, k):
    """Parabolic refinement of the maximum of each row of g at integer column k."""
    rows = np.arange(g.shape[0])
    k = np.clip(k, 1, g.shape[1] - 2)
    gm, g0, gp = g[rows, k - 1], g[rows, k], g[rows, k + 1]
    curvature = gm - 2 * g0 + gp
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(curvature) > 1e-12, 0.5 * (gm - gp) / curvature, 0.0)
    return k + np.clip(shift, -1, 1)


class LevelCrossingTracker:
    """Membrane contour as a per-ray crossing of a fixed intensity level in a smoothed image.

    fit(ref): on the time-mean image, find the strongest radial intensity step near the rim (its sign
    gives the polarity, so dark- and bright-interior cells both work), its radius on every ray, and the
    reference intensity there (the level). track(frames): on each ray, the level crossing nearest that
    ray's mean edge within +-half_window px, linearly interpolated, smoothed along the contour
    (sigma_contour samples). Each frame is searched about the previous frame's centroid and the radii are
    returned about the frame's own contour centroid, which removes translation (modes q = +-1).
    A level crossing of a strongly smoothed image is far less noise-sensitive than a gradient peak
    (noise floor ~11 nm against ~60 nm on static synthetic frames).
    """

    def __init__(
        self,
        n_rays=2048,
        sigma_img=3.0,
        half_window=3.0,
        dr=0.05,
        search=(35.0, 115.0),
        sigma_contour=3.0,
        fit_halfwidth=20.0,
    ):
        self.n_rays, self.sigma_img, self.half_window, self.dr = n_rays, sigma_img, half_window, dr
        self.search, self.sigma_contour, self.fit_halfwidth = search, sigma_contour, fit_halfwidth
        self.theta = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)

    def fit(self, ref, n_iter=3):
        """Locate the mean edge on ref, re-centring on the edge contour until the centre moves < 0.5 px."""
        self.cx, self.cy, _ = hough_centre(ref)
        for _ in range(n_iter):
            self._locate_mean_edge(ref)
            x = self.cx + self.mean_edge * np.cos(self.theta)
            y = self.cy + self.mean_edge * np.sin(self.theta)
            cx, cy, _ = fit_circle(x, y)
            if np.hypot(cx - self.cx, cy - self.cy) < 0.5:
                break
            self.cx, self.cy = cx, cy
        smoothed = ndimage.gaussian_filter(ref, self.sigma_img)
        self.level = ndimage.map_coordinates(
            smoothed,
            [self.cy + np.sin(self.theta) * self.mean_edge, self.cx + np.cos(self.theta) * self.mean_edge],
            order=3,
        )
        return self

    def _locate_mean_edge(self, ref):
        r = np.arange(*self.search, 0.25)
        rays = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        g = np.gradient(
            np.median(polar_samples(ndimage.gaussian_filter(ref, 2.0), self.cx, self.cy, rays, r, order=1), axis=0),
            0.25,
        )
        k = np.argmax(np.abs(g))  # strongest rim step, either sign
        self.sign, r_rim = float(np.sign(g[k])), float(r[k])
        # per ray: strongest step of the same sign within +-fit_halfwidth of the rim (cells are not round),
        # median-filtered along the contour against stray fringes, then refined within +-half_window
        rr = np.arange(max(r_rim - self.fit_halfwidth, 5.0), r_rim + self.fit_halfwidth, self.dr)
        smoothed = ndimage.gaussian_filter(ref, max(self.sigma_img, 2.0))
        gr = self.sign * np.gradient(polar_samples(smoothed, self.cx, self.cy, self.theta, rr), self.dr, axis=1)
        coarse = ndimage.median_filter(rr[np.argmax(gr, axis=1)], size=61, mode="wrap")
        offsets = np.arange(-self.half_window, self.half_window, self.dr)
        radii = coarse[:, None] + offsets[None, :]
        img = ndimage.gaussian_filter(ref, self.sigma_img)
        g2 = self.sign * np.gradient(
            ndimage.map_coordinates(
                img,
                [self.cy + np.sin(self.theta)[:, None] * radii, self.cx + np.cos(self.theta)[:, None] * radii],
                order=3,
            ),
            self.dr,
            axis=1,
        )
        k2 = subpixel_peak(g2, np.argmax(g2, axis=1))
        self.mean_edge = ndimage.median_filter(coarse + offsets[0] + k2 * self.dr, size=31, mode="wrap")

    def edge_radii(self, img, cx, cy):
        """Edge radius on every ray about (cx, cy) [px], and 1.0 where a crossing was found (else filled)."""
        offsets = np.arange(-self.half_window, self.half_window + self.dr / 2, self.dr)
        radii = self.mean_edge[:, None] + offsets[None, :]
        smoothed = ndimage.gaussian_filter(img, self.sigma_img)
        profile = ndimage.map_coordinates(
            smoothed, [cy + np.sin(self.theta)[:, None] * radii, cx + np.cos(self.theta)[:, None] * radii], order=3
        )
        f = self.sign * (profile - self.level[:, None])  # increases through 0 at the edge
        crossing = (f[:, :-1] < 0) & (f[:, 1:] >= 0)
        distance = np.where(crossing, np.abs(offsets[:-1] + self.dr / 2)[None, :], np.inf)
        k = np.argmin(distance, axis=1)
        rows = np.arange(len(k))
        found = np.isfinite(distance[rows, k])
        f0, f1 = f[rows, k], f[rows, k + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(f1 != f0, -f0 / (f1 - f0), 0.5)
        r = np.where(found, self.mean_edge + offsets[k] + frac * self.dr, np.nan)
        if not found.all():  # fill misses periodically
            r = (
                np.interp(self.theta, self.theta[found], r[found], period=2 * np.pi)
                if found.any()
                else self.mean_edge.copy()
            )
        if self.sigma_contour > 0:
            r = ndimage.gaussian_filter1d(r, self.sigma_contour, mode="wrap")
        return r, found.astype(float)

    def track(self, frames):
        """Radii (frames, n_rays) [px] about each frame's contour centroid, centroids, found flags."""
        radii = np.empty((len(frames), self.n_rays))
        centres = np.empty((len(frames), 2))
        found = np.empty((len(frames), self.n_rays))
        x0 = self.cx + self.mean_edge * np.cos(self.theta)
        y0 = self.cy + self.mean_edge * np.sin(self.theta)
        offset = np.array([self.cx - x0.mean(), self.cy - y0.mean()])  # search centre - contour centroid
        sx, sy = self.cx, self.cy
        for t, img in enumerate(frames):
            r, found[t] = self.edge_radii(img, sx, sy)
            x, y = sx + r * np.cos(self.theta), sy + r * np.sin(self.theta)
            cx, cy = x.mean(), y.mean()
            radii[t], centres[t] = np.hypot(x - cx, y - cy), (cx, cy)
            sx, sy = cx + offset[0], cy + offset[1]
        return radii, centres, found


def radial_template(mean_img, cx, cy, r_max=145.0, dr=0.05):
    """Angle-median radial intensity profile of the mean image about (cx, cy): (r [px], intensity)."""
    r = np.arange(0.0, r_max, dr)
    rays = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    samples = ndimage.map_coordinates(
        mean_img, [cy + np.outer(np.sin(rays), r), cx + np.outer(np.cos(rays), r)], order=1, mode="nearest"
    )
    return r, np.median(samples, axis=0)


def pixel_noise(frames, corner=40):
    """Per-pixel temporal noise [grey levels] from frame differences in the empty top-right corner."""
    background = frames[:, :corner, -corner:]
    return float(np.std(np.diff(background, axis=0)) / np.sqrt(2))


class SyntheticCell:
    """Synthetic bright-field frames with a known boundary: the cell's own radial profile I(r), drawn as
    I(rho - h(phi)) so the whole rim pattern moves with the membrane, plus the measured pixel noise,
    spatially correlated over 0.6 px as in the camera."""

    def __init__(self, mean_img, cx, cy, noise_sd, rng):
        self.r, self.intensity = radial_template(mean_img, cx, cy)
        yy, xx = np.mgrid[0 : mean_img.shape[0], 0 : mean_img.shape[1]].astype(float)
        self.rho, self.phi = np.hypot(xx - cx, yy - cy), np.arctan2(yy - cy, xx - cx)
        self.noise_sd, self.rng = noise_sd, rng

    def frame(self, theta_grid=None, h_grid=None):
        h = 0.0 if h_grid is None else np.interp(self.phi, theta_grid, h_grid, period=2 * np.pi)
        img = np.interp(self.rho - h, self.r, self.intensity)
        z = ndimage.gaussian_filter(self.rng.standard_normal(img.shape), 0.6)
        return img + self.noise_sd * z / z.std()


def tracker_validation(ref, frames, measured_radii, n_static, n_moving, seed):
    """Noise floor [nm]: SD of the tracked radius on static synthetic frames (the membrane held still).
    Gain: synthetic frames whose boundary moves with this cell's own measured shape modes q = 2..30;
    gain = sum Re(conj(imposed) recovered) / sum |imposed|^2 over modes q = 2..6 (1 = no attenuation)."""
    rng = np.random.default_rng(seed)
    tracker = LevelCrossingTracker().fit(ref)
    cell = SyntheticCell(ref, tracker.cx, tracker.cy, pixel_noise(frames), rng)
    static = np.stack([cell.frame() for _ in range(n_static)])
    radii, _, _ = LevelCrossingTracker().fit(static.mean(axis=0)).track(static)
    floor_nm = float(((radii - radii.mean(axis=0)) * NM_PER_PX).std())

    d = measured_radii[:n_moving] - measured_radii[:n_moving].mean(axis=0)
    H = np.fft.fft(d, axis=1)
    keep = np.zeros(d.shape[1], bool)
    keep[2:31] = keep[-30:-1] = True  # modes q = +-2..30
    imposed = np.real(np.fft.ifft(np.where(keep, H, 0), axis=1))  # px, on the tracker's ray grid
    moving = np.stack([cell.frame(tracker.theta, h) for h in imposed])
    recovered, _, _ = LevelCrossingTracker().fit(moving.mean(axis=0)).track(moving)
    Hi = np.fft.fft(imposed, axis=1)[:, 2:7]
    Hr = np.fft.fft(recovered - recovered.mean(axis=0), axis=1)[:, 2:7]
    gain = float(np.sum(np.real(np.conj(Hi) * Hr)) / np.sum(np.abs(Hi) ** 2))
    return floor_nm, gain


def track_video(job):
    """Track one video in blocks of 500 frames (fitting the tracker on the mean of the first 500), run the
    synthetic validation, and cache everything to tracks/<state>_<cell>.npz."""
    cell, state, data_root, out_dir, n_frames, n_static, n_moving = job
    out = Path(out_dir) / "tracks" / f"{state}_{cell}.npz"
    if out.exists():
        return f"{video_label(cell, state)}: cached"
    t0 = time.time()
    folder = video_folder(data_root, cell, state)
    total = min(len(frame_paths(folder)), n_frames)
    block = min(500, total)
    first = load_frames(folder, 0, block)
    ref = first.mean(axis=0)
    tracker = LevelCrossingTracker().fit(ref)
    parts = []
    for start in range(0, total, block):
        frames = first if start == 0 else load_frames(folder, start, min(block, total - start))
        parts.append(tracker.track(frames))
    radii, centres, found = (np.concatenate(p) for p in zip(*parts))
    floor_nm, gain = tracker_validation(
        ref, first, radii, n_static, n_moving, seed=zlib.crc32(f"{state}{cell}".encode())
    )
    np.savez_compressed(
        out,
        theta=tracker.theta,
        radii_px=radii,
        centres_px=centres,
        found=found.astype(np.uint8),
        mean_frame=first[0],
        floor_nm=floor_nm,
        gain=gain,
    )
    rms = float(((radii - radii.mean(axis=0)) * NM_PER_PX).std())
    return (
        f"{video_label(cell, state)}: {total} frames, rms {rms:.1f} nm, noise floor {floor_nm:.1f} nm, "
        f"gain {gain:.3f}, crossings found {found.mean():.4f} ({time.time() - t0:.0f} s)"
    )


def load_track(out_dir, cell, state):
    z = np.load(Path(out_dir) / "tracks" / f"{state}_{cell}.npz")
    return {k: z[k] for k in z.files}


# =====================================================================================================
# 3. ANALYSIS: features, exploration, model, fits
# =====================================================================================================
def displacement_m(track):
    """Contour displacement from the time-mean shape, delta h(theta, t) [m], shape (frames, rays)."""
    return (track["radii_px"] - track["radii_px"].mean(axis=0)) * NM_PER_PX * 1e-9


def mean_radius_m(track):
    return float(track["radii_px"].mean()) * NM_PER_PX * 1e-9


def contour_modes(d):
    """Complex Fourier modes h_q(t) = (1/N) sum_theta delta h e^{-i q theta}, shape (frames, rays)."""
    return np.fft.fft(d, axis=1) / d.shape[1]


def periodogram(y):
    """Angular frequencies w_j (j = 1 .. N/2 - 1) and ordinates I_j with E I_j ~ S(w_j)."""
    n = len(y)
    Y = np.fft.rfft(y - y.mean())
    w = 2 * np.pi * np.fft.rfftfreq(n, FRAME_DT)
    return w[1:-1], (FRAME_DT / n * np.abs(Y) ** 2)[1:-1]


def mode_periodograms(H, qs=FIT_MODES):
    """[(q, w, I)] for the real and imaginary part of every mode q: the data the models are fitted to."""
    return [(q, *periodogram(part(H[:, q]))) for q in qs for part in (np.real, np.imag)]


def bin_periodograms(data, k=BIN):
    """Average k adjacent ordinates. The mean of k independent exponentials is Gamma(k, S/k), so k times the
    Whittle likelihood of the bin means is a valid likelihood, k times cheaper, for spectra smooth over k."""
    out = []
    for q, w, I in data:
        n = len(w) // k * k
        out.append((q, w[:n].reshape(-1, k).mean(axis=1), I[:n].reshape(-1, k).mean(axis=1)))
    return out


def mode_components(H, qs=FIT_MODES):
    """Real and imaginary parts of modes qs, each standardized: (frames, 2 len(qs))."""
    Y = np.stack([part(H[:, q]) for q in qs for part in (np.real, np.imag)], axis=1)
    return (Y - Y.mean(axis=0)) / Y.std(axis=0)


def surrogate_kurtosis(Y, n_surrogates, rng):
    """Mean kurtosis of the components of multivariate phase-randomized surrogates of Y: one random phase
    per frequency shared by all components keeps every spectrum and cross-spectrum but makes the data
    Gaussian, so this is the null distribution of the statistic for a Gaussian process with Y's memory."""
    F = np.fft.rfft(Y, axis=0)
    out = np.empty(n_surrogates)
    for i in range(n_surrogates):
        phases = np.exp(2j * np.pi * rng.uniform(size=F.shape[0]))
        phases[0] = 1
        Z = np.fft.irfft(F * phases[:, None], n=Y.shape[0], axis=0)
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        out[i] = np.mean(np.mean(Z**4, axis=0))
    return out


def correlation_time(y, max_lag_ms=40):
    """1/e time [s] of an exponential fitted to the autocorrelation of y at lags 1..max_lag_ms (where > 0.05)."""
    y = y - y.mean()
    lags = np.arange(1, max_lag_ms + 1)
    acf = np.array([np.mean(y[:-L] * y[L:]) for L in lags]) / y.var()
    use = acf > 0.05
    return float(-FRAME_DT / np.polyfit(lags[use], np.log(acf[use]), 1)[0])


def explore_video(job):
    """Descriptive statistics of one video: wall amplitude, tracking-noise bound, Gaussianity, memory."""
    cell, state, out_dir, n_surrogates = job
    track = load_track(out_dir, cell, state)
    d_nm = displacement_m(track) * 1e9
    H = contour_modes(d_nm)
    lag1 = np.mean((d_nm[1:] - d_nm[:-1]) ** 2)
    Y = mode_components(H)
    observed = float(np.mean(np.mean(Y**4, axis=0)))
    null = surrogate_kurtosis(Y, n_surrogates, np.random.default_rng(zlib.crc32(f"{state}{cell}".encode())))
    z = (d_nm / d_nm.std()).ravel()
    shape_var = np.sum(np.abs(H[:, 2 : len(H[0]) // 2]) ** 2, axis=0) * 2
    return {
        "cell": cell,
        "state": state,
        "frames": int(len(d_nm)),
        "R_um": mean_radius_m(track) * 1e6,
        "rms_nm": float(d_nm.std()),
        "noise_floor_nm": float(track["floor_nm"]),
        "gain": float(track["gain"]),
        "noise_bound_nm": float(np.sqrt(lag1 / 2)),
        "kurtosis_pooled": float(np.mean(z**4)),
        "kurtosis_modes": observed,
        "kurtosis_modes_null_95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "tau_q2_ms": correlation_time(H[:, 2].real) * 1e3,
        "share_q2_6": float(shape_var[:5].sum() / shape_var.sum()),
        "rms_q2_6_nm": float(np.sqrt(np.mean(np.sum(2 * np.abs(H[:, 2:7]) ** 2, axis=1)))),
    }


# ---- generative model --------------------------------------------------------------------------------
def equator_weights(q, l_max):
    """|Y_lq(pi/2, 0)|^2 for l = q..l_max (0 when l + q is odd). With l + q = 2a and l - q = 2b,
    |Y_lq(pi/2, 0)|^2 = (2l + 1)/(4 pi) C(2a, a) C(2b, b) / 4^l, which tends to 1/pi^2 at large l."""
    l = np.arange(q, l_max + 1)
    a, b = (l + q) // 2, (l - q) // 2
    log_central = lambda n: gammaln(2 * n + 1) - 2 * gammaln(n + 1)  # log C(2n, n)
    w = (2 * l + 1) / (4 * np.pi) * np.exp(log_central(a) + log_central(b) - l * np.log(4.0))
    return l, np.where((l + q) % 2 == 0, w, 0.0)


def physical(p):
    """Natural -> model parameters: sigma_bar = tension / kappa, gamma_bar = confinement / kappa."""
    out = dict(p)
    out["sigma_bar"] = p["tension"] / p["kappa_kT"]
    out["gamma_bar"] = p["confinement"] / p["kappa_kT"]
    return out


def relaxation_terms(q, p, R, l_max, generator="P"):
    """Relaxation rates [1/s] and variances per real part [m^2] of the 3D modes l that project onto contour
    mode q (Milner & Safran 1987; Faizi et al. 2020 eqs. 1-2, with an optional confinement gamma_bar):
        E_l = (l-1)(l+2)[l(l+1) + sigma_bar] + gamma_bar,     var_l = |Y_lq|^2 R^2 / (kappa E_l) / 2,
        r_l = kappa E_l l(l+1) / (eta R^3 (4 l^3 + 6 l^2 - 1)).
    Passive null generators: G1 = inner/outer viscosity contrast 10 (Faizi eq. 2); G2 = each mode relaxes as
    a log-normal mixture of rates r_l e^{s Z} (5-node Gauss-Hermite), with the same variance."""
    pp = physical(p)
    l, w = equator_weights(q, l_max)
    keep = w > 1e-12 * w.max()
    l, w = l[keep], w[keep]
    E = (l - 1) * (l + 2) * (l * (l + 1) + pp["sigma_bar"]) + pp["gamma_bar"]
    kappa = pp["kappa_kT"] * K_B * TEMPERATURE
    drag = 4 * l**3 + 6 * l**2 - 1
    if generator == "G1":
        drag = drag + (2 * l**3 + 3 * l**2 - 5) * (10.0 - 1)
    rate = kappa * E * l * (l + 1) / (pp["eta"] * R**3 * drag)
    var = R**2 * w / (pp["kappa_kT"] * E) / 2
    if generator == "G2":
        nodes, weights = np.polynomial.hermite.hermgauss(5)
        rate = (rate[:, None] * np.exp(pp["s"] * np.sqrt(2) * nodes)[None, :]).ravel()
        var = (var[:, None] * (weights / np.sqrt(np.pi))[None, :]).ravel()
    return rate, var


def ou_spectrum_observed(w, rate, var):
    """Spectral density of a sum of OU terms (rates, stationary variances), each averaged over the 1 ms
    exposure and sampled every 1 ms (an ARMA(1,1) process), so no aliasing sum is needed. With x = rate dt,
    a = 1 - e^{-x}, r = e^{-x}, b = 1 - cos(w dt): S = dt 2 var N / (x^2 (a^2 + 2 b r)),
    N = a^2 (x - b) + 2 b r (x - a). N's leading terms cancel, so for x < 0.05 it is evaluated from its
    Taylor series through x^10 (relative truncation error < 1e-9)."""
    w = np.asarray(w, float)[None, :]
    x = np.asarray(rate, float)[:, None] * FRAME_DT
    v = np.asarray(var, float)[:, None]
    r = np.exp(-x)
    a = -np.expm1(-x)
    b = 2 * np.sin(w * FRAME_DT / 2) ** 2
    N = a**2 * (x - b) + 2 * b * r * (x - a)
    small = np.broadcast_to(x < 0.05, N.shape)
    if small.any():
        xs, bs = np.broadcast_to(x, N.shape)[small], np.broadcast_to(b, N.shape)[small]
        series = [
            (3, -1 / 3, 1),
            (4, 1 / 3, -1),
            (5, -11 / 60, 35 / 60),
            (6, 13 / 180, -45 / 180),
            (7, -57 / 2520, 217 / 2520),
            (8, 5 / 840, -21 / 840),
            (9, -247 / 181440, 1143 / 181440),
            (10, 251 / 907200, -1275 / 907200),
        ]  # x^k (c_b b + c_0)
        N = N.copy()
        N[small] = sum(xs**k * (cb * bs + c0) for k, cb, c0 in series)
    return np.sum(FRAME_DT * 2 * v * N / (x**2 * (a**2 + 2 * b * r)), axis=0)


def model_spectrum(w, q, p, R, l_max, generator="P"):
    """Observed spectrum of one real part of contour mode q [m^2 s]. Model A adds to every 3D mode a colored
    forcing of intensity a D_l and memory tau_a, S_l = 2 D_l [1 + a/(1 + w^2 tau_a^2)] / (w^2 + r_l^2):
    the colored term is the difference of two OU spectra, c [1/(w^2 + mu^2) - 1/(w^2 + r^2)], mu = 1/tau_a,
    c = 2 a D mu^2 / (r^2 - mu^2). Tracking noise adds white noise e2 per real part."""
    rate, var = relaxation_terms(q, p, R, l_max, generator)
    S = ou_spectrum_observed(w, rate, var)
    if p.get("a", 0.0) > 0:
        mu, D = 1.0 / p["tau_a"], var * rate
        c = 2 * p["a"] * D * mu**2 / (rate**2 - mu**2)
        S = (
            S
            + ou_spectrum_observed(w, np.full_like(rate, mu), c / (2 * mu))
            - ou_spectrum_observed(w, rate, c / (2 * rate))
        )
    return S + p["e2"] * FRAME_DT


# ---- fitting ------------------------------------------------------------------------------------------
def negative_log_likelihood(z, names, data, R, l_max, weight, generator="P", fixed=None):
    """Whittle -log L = weight sum [log S + I/S] of log-parameters z; weight = BIN for binned data."""
    with np.errstate(over="ignore"):
        p = dict(zip(names, np.exp(z)))
    if fixed:
        p.update(fixed)
    if generator == "G2" and p["s"] > SPREAD_MAX:
        return 1e30
    total = 0.0
    for q, w, I in data:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # extreme trial points are rejected below
            S = model_spectrum(w, q, p, R, l_max, generator)
        if not np.all(np.isfinite(S)) or np.any(S <= 0):
            return 1e30
        total += weight * float(np.sum(np.log(S) + I / S))
    return total


def nelder_mead(f, x0, max_iter):
    r = minimize(
        f, x0, method="Nelder-Mead", options={"maxiter": max_iter, "maxfev": 2 * max_iter, "xatol": 1e-4, "fatol": 1e-3}
    )
    return r.x, float(r.fun)


def multistart(f, starts, max_iter):
    """Best Nelder-Mead optimum over the starts, polished by one restart from it."""
    best = min((nelder_mead(f, s, max_iter) for s in starts), key=lambda r: r[1])
    polished = nelder_mead(f, best[0], max_iter)
    return polished if polished[1] < best[1] else best


def fit_passive_and_active(binned, R, rng, n_starts, max_iter):
    """Binned-Whittle fits of P (generic start plus random perturbations) and A. A's likelihood has local optima
    in two membrane regimes, so A starts from both, each with several forcings (a, tau_a): P's membrane, and a
    tension-dominated membrane (tension 1e4, eta 1 mPa s, confinement 1e5, a fifth of P's noise), where the
    best colored fits of these cells lie. A nests P (a -> 0), so its optimum is clamped at P's."""
    start_P = np.log([50.0, 500.0, 5e-3, 50.0, 1e-19])  # kappa/kT, tension, eta, confinement, e2
    starts = [start_P] + [start_P + rng.normal(0, 0.7, 5) for _ in range(n_starts - 1)]
    f_P = lambda z: negative_log_likelihood(z, PARAMS["P"], binned, R, L_MAX_FIT, BIN)
    z_P, nll_P = multistart(f_P, starts, max_iter)
    tension_membrane = np.array([z_P[0], np.log(1e4), np.log(1e-3), np.log(1e5)])
    starts_A = [np.concatenate([z_P[:4], np.log([a, 0.03]), z_P[4:]]) for a in (3.0, 30.0, 300.0, 3000.0)] + [
        np.concatenate([tension_membrane, np.log([a, tau]), z_P[4:] - np.log(5.0)])
        for a in (30.0, 300.0, 3000.0)
        for tau in (0.01, 0.03, 0.1)
    ]
    f_A = lambda z: negative_log_likelihood(z, PARAMS["A"], binned, R, L_MAX_FIT, BIN)
    z_A, nll_A = multistart(f_A, starts_A, max_iter)
    return {"P": {"z": z_P.tolist(), "nll_binned": nll_P}, "A": {"z": z_A.tolist(), "nll_binned": min(nll_A, nll_P)}}


def selection_statistic(raw, R, fits, max_iter):
    """Delta AIC = AIC(P) - AIC(A) from the unbinned Whittle likelihood (l_max 80), each model polished from
    its binned optimum. The same function scores the data and every bootstrap replicate."""
    out = {}
    for model in ("P", "A"):
        f = lambda z, names=PARAMS[model]: negative_log_likelihood(z, names, raw, R, L_MAX_SELECT, 1)
        z0 = np.array(fits[model]["z"])
        z, nll = nelder_mead(f, z0, max_iter)
        out[model] = {"z": (z if nll < f(z0) else z0).tolist(), "nll": min(nll, f(z0))}
    out["A"]["nll"] = min(out["A"]["nll"], out["P"]["nll"])
    out["dAIC"] = 2 * (out["P"]["nll"] - out["A"]["nll"]) - 2 * (len(PARAMS["A"]) - len(PARAMS["P"]))
    return out


def video_data(out_dir, cell, state):
    track = load_track(out_dir, cell, state)
    raw = mode_periodograms(contour_modes(displacement_m(track)))
    return raw, bin_periodograms(raw), mean_radius_m(track)


def fit_video(job):
    cell, state, out_dir, n_starts, max_iter = job
    raw, binned, R = video_data(out_dir, cell, state)
    fits = fit_passive_and_active(
        binned, R, np.random.default_rng(zlib.crc32(f"fit{state}{cell}".encode())), n_starts, max_iter
    )
    # transport walls: the A optimum is degenerate (kappa runs to 0 along a flat ridge with the passive
    # relaxation infinitely fast), which leaves the instantaneous wall undetermined. The wall model keeps
    # what the videos identify: the forcing (a, tau_a) from A and the bending modulus from P; tension,
    # viscosity, confinement and noise are refitted around them.
    fixed = {
        "kappa_kT": float(np.exp(fits["P"]["z"][0])),
        "a": float(np.exp(fits["A"]["z"][4])),
        "tau_a": float(np.exp(fits["A"]["z"][5])),
    }
    free = ["tension", "eta", "confinement", "e2"]
    zP = np.array(fits["P"]["z"])
    starts = [
        np.array([np.logaddexp(zP[1], 0.0), zP[2] + np.log(eta_scale), zP[3] + np.log(conf_scale), zP[4]])
        for eta_scale in (0.03, 0.1, 1.0)
        for conf_scale in (1.0, 10.0)
    ]
    f_wall = lambda z: negative_log_likelihood(z, free, binned, R, L_MAX_FIT, BIN, fixed=fixed)
    z_wall, nll_wall = multistart(f_wall, starts, max_iter)
    fits["A_wall"] = {"params": {**fixed, **dict(zip(free, np.exp(z_wall)))}, "nll_binned": nll_wall}
    for model in ("P", "A"):
        fits[model]["params"] = dict(zip(PARAMS[model], np.exp(fits[model]["z"]).tolist()))
    fits["gof"] = {m: goodness_of_fit(binned, fits[m]["params"], R) for m in ("P", "A")}
    return {"cell": cell, "state": state, "R_m": R, **fits}


# =====================================================================================================
# 4. EVALUATION: identifiability, calibrated model selection, goodness of fit
# =====================================================================================================
def goodness_of_fit(binned, p, R):
    """Under the model each bin mean / S is Gamma(BIN, 1/BIN), so BIN (I/S - 1)^2 has mean 1: the ratio
    of the sum to the number of bins is ~1 for an adequate model (approximate: neighbouring bins of a
    finite record leak into each other). Also the share of bins with |standardized residual| > 2."""
    z = np.concatenate([np.sqrt(BIN) * (I / model_spectrum(w, q, p, R, L_MAX_FIT) - 1) for q, w, I in binned])
    return {"chi2_per_bin": float(np.mean(z**2)), "share_outside_2": float(np.mean(np.abs(z) > 2))}


def profile_interval(names, which, z_hat, f_hat, binned, R, iters, n_points, span=0.5, max_span=6.0, min_span=0.02):
    """95% profile-likelihood interval of parameter `which`: on a log grid around the optimum every other
    parameter is re-optimized (warm-started from the nearest finished grid point); the interval is where the
    profile stays within 1.92 of the minimum (chi^2_1 / 2). The grid widens (to +-max_span in log units)
    while a side stays open and narrows while the interval fits inside one grid step. A side that never
    crosses is reported OPEN: the parameter is not practically identifiable (Raue et al. 2009)."""
    others = [i for i in range(len(names)) if i != which]
    for attempt in range(4):
        grid = z_hat[which] + np.linspace(-span, span, n_points)
        prof, warm = np.empty(n_points), {}
        best = (np.inf, None)
        for j in np.argsort(np.abs(grid - z_hat[which])):
            x0 = warm[min(warm, key=lambda k: abs(k - j))] if warm else z_hat[others]

            def f(z_others, j=j, grid=grid):
                z = np.empty(len(names))
                z[which], z[others] = grid[j], z_others
                return negative_log_likelihood(z, names, binned, R, L_MAX_FIT, BIN)

            x, fx = nelder_mead(f, x0, iters)
            prof[j], warm[j] = fx, x
            if fx < best[0]:
                z_best = np.empty(len(names))
                z_best[which], z_best[others] = grid[j], x
                best = (fx, z_best)
        prof = prof - min(f_hat, prof.min())
        lo, hi = threshold_crossings(grid, prof)
        if attempt == 3:
            break
        if (lo is None or hi is None) and span < max_span:
            span *= 2
        elif lo is not None and hi is not None and hi - lo < 2 * (grid[1] - grid[0]) and span > min_span:
            span /= 4
        else:
            break
    return (
        {
            "lo": float(np.exp(grid[0] if lo is None else lo)),
            "hi": float(np.exp(grid[-1] if hi is None else hi)),
            "open_lo": lo is None,
            "open_hi": hi is None,
            "grid": grid.tolist(),
            "profile": prof.tolist(),
        },
        best,
    )


def threshold_crossings(grid, prof, level=1.92):
    """Interpolated grid points where the profile first rises through `level` left and right of its minimum."""
    i = int(np.argmin(prof))

    def walk(indices):
        for j0, j1 in zip(indices[:-1], indices[1:]):
            if prof[j1] > level >= prof[j0]:
                return grid[j0] + (level - prof[j0]) * (grid[j1] - grid[j0]) / (prof[j1] - prof[j0])
        return None

    return walk(list(range(i, -1, -1))), walk(list(range(i, len(grid))))


def profile_video(job):
    """Profiles of every physical parameter of P and A. A profile point below the optimum means the fit had
    stopped at a local optimum: refit from that point and redo every profile (up to three times)."""
    cell, state, out_dir, z_fits, targets, iters, n_points = job
    _, binned, R = video_data(out_dir, cell, state)
    out = {"cell": cell, "state": state}
    for model in ("P", "A"):
        names = PARAMS[model]
        f = lambda z, names=names: negative_log_likelihood(z, names, binned, R, L_MAX_FIT, BIN)
        z_hat = np.array(z_fits[model])
        f_hat = f(z_hat)
        for refit in range(3):
            intervals, better = {}, (f_hat - 1e-3, None)
            for name in targets[model]:
                intervals[name], best = profile_interval(
                    names, names.index(name), z_hat, f_hat, binned, R, iters, n_points
                )
                if best[0] < better[0]:
                    better = best
            if better[1] is None:
                break
            z_hat, f_hat = nelder_mead(f, better[1], 3 * iters)
        out[model] = {
            "estimate": dict(zip(names, np.exp(z_hat).tolist())),
            "nll_binned": f_hat,
            "intervals": intervals,
            "refits": refit,
        }
    return out


# ---- null distribution of Delta AIC under passive but misspecified generators --------------------------
def fit_generator(raw, R, generator, z_P, max_iter):
    """Fit a passive generator to the data (unbinned, l_max 80): G0 = P itself; G1 from eta / 5 (the
    viscosity contrast slows the rates about five-fold); G2 from spreads s = 0.3 and 1."""
    names = PARAMS["P"] if generator == "G0" else PARAMS[generator]
    gen = "P" if generator == "G0" else generator
    z = np.array(z_P)
    if generator == "G1":
        z[2] -= np.log(5.0)
    starts = [np.append(z, np.log(s)) for s in (0.3, 1.0)] if generator == "G2" else [z]
    f = lambda zz: negative_log_likelihood(zz, names, raw, R, L_MAX_SELECT, 1, gen)
    z_fit, nll = multistart(f, starts, max_iter)
    return {"params": dict(zip(names, np.exp(z_fit).tolist())), "nll": nll, "k": len(names)}


def simulate_passive_modes(generator, p, R, rng, n_frames):
    """Exact simulation of the observed contour modes q = 2..6 under a passive generator: every OU term is
    advanced exactly frame to frame, and its exposure average over the frame is drawn jointly with the end
    state (a bivariate Gaussian given the start state); white tracking noise e2 is added per real part."""
    gen = "P" if generator == "G0" else generator
    lam, v, owner = [], [], []
    for i, q in enumerate(FIT_MODES):
        rate, var = relaxation_terms(q, p, R, L_MAX_SELECT, gen)
        for part in (0, 1):
            lam.append(rate)
            v.append(var)
            owner.append(np.full(len(rate), 2 * i + part))
    lam, v, owner = np.concatenate(lam), np.concatenate(v), np.concatenate(owner)
    x = lam * FRAME_DT
    decay = np.exp(-x)
    om, om2 = -np.expm1(-x), -np.expm1(-2 * x)  # 1 - e^{-x}, 1 - e^{-2x}
    var_end = v * om2  # Var(state at frame end | start)
    cov = v / x * om**2  # Cov(end state, exposure mean | start)
    bracket = np.where(x < 1e-3, x**2 / 3 - x**3 / 4 + 7 * x**4 / 60, 1 - 2 * om / x + om2 / (2 * x))
    var_mean = np.maximum(2 * v / x * bracket, cov**2 / np.maximum(var_end, 1e-300) + 1e-30 * v)
    l11 = np.sqrt(var_end)
    l21 = cov / np.where(l11 > 0, l11, 1.0)
    l22 = np.sqrt(np.maximum(var_mean - l21**2, 0.0))
    state = np.sqrt(v) * rng.standard_normal(len(v))
    frames = np.empty((n_frames, len(v)))
    for k in range(n_frames):
        n1, n2 = rng.standard_normal(len(v)), rng.standard_normal(len(v))
        frames[k] = om / x * state + l21 * n1 + l22 * n2
        state = decay * state + l11 * n1
    parts = np.stack([frames[:, owner == j].sum(axis=1) for j in range(2 * len(FIT_MODES))], axis=1)
    parts += np.sqrt(p["e2"]) * rng.standard_normal(parts.shape)
    H = np.zeros((n_frames, max(FIT_MODES) + 1), complex)
    for i, q in enumerate(FIT_MODES):
        H[:, q] = parts[:, 2 * i] + 1j * parts[:, 2 * i + 1]
    return H


def null_replicate(job):
    """Simulate one replicate from a fitted passive generator and score it exactly as the data."""
    generator, cell, state, rep, p, R, n_frames, n_starts, max_iter = job
    rng = np.random.default_rng([28, zlib.crc32(f"{generator}{state}{cell}".encode()), rep])
    raw = mode_periodograms(simulate_passive_modes(generator, p, R, rng, n_frames))
    fits = fit_passive_and_active(bin_periodograms(raw), R, rng, n_starts, max_iter)
    sel = selection_statistic(raw, R, fits, max_iter)
    return {
        "generator": generator,
        "cell": cell,
        "state": state,
        "rep": rep,
        "dAIC": sel["dAIC"],
        "tau_a_ms": float(np.exp(sel["A"]["z"][5]) * 1e3),
    }


# =====================================================================================================
# 5. INTERPRETATION: transport inside the fitted moving wall
# =====================================================================================================
def psd_sqrt(M):
    """Square root of a symmetric positive semi-definite matrix, robust to round-off."""
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))


def simulate_wall(p, R, T_phys, rng, l_max=L_MAX_WALL):
    """Instantaneous wall at every frame time (no exposure average), modes q = 2..6, in units of R.
    Each real part of each 3D mode is the linear SDE
        dh_th = -r_l h_th dt + sqrt(2 D_l) dW1,  dh_act = (-r_l h_act + f) dt,  df = -f/tau_a dt + sqrt(2 a D_l)/tau_a dW2,
    advanced exactly: X_{k+1} = F X_k + L xi_k with F = exp(A dt) and L L^T = P - F P F^T, P the stationary
    covariance (Lyapunov), started in the stationary state. h = h_th + h_act has the law of model A.
    Returns alpha, beta (modes, frames) with wall radius b = R0 + sum alpha_q cos q phi + beta_q sin q phi."""
    a, tau = p.get("a", 0.0), p.get("tau_a", 1.0)
    n_frames = max(int(np.ceil(T_phys / FRAME_DT)) + 1, 2)
    F_list, L_list, X, owner = [], [], [], []
    for q in FIT_MODES:
        rate, var = relaxation_terms(q, p, R, l_max)
        for r_l, v_l in zip(rate, var):
            D = v_l * r_l
            A = np.array([[-r_l, 0.0, 0.0], [0.0, -r_l, 1.0], [0.0, 0.0, -1.0 / tau]])
            Q = np.diag([2 * D, 0.0, 2 * a * D / tau**2])
            F = expm(A * FRAME_DT)
            P = solve_continuous_lyapunov(A, -Q)
            for _ in range(2):  # real and imaginary parts
                F_list.append(F)
                L_list.append(psd_sqrt(P - F @ P @ F.T))
                X.append(psd_sqrt(P) @ rng.standard_normal(3))
                owner.append(q)
    F_all, L_all, X, owner = np.array(F_list), np.array(L_list), np.array(X), np.array(owner)
    h = np.zeros((len(X), n_frames))
    for k in range(n_frames):
        X = np.einsum("mij,mj->mi", F_all, X) + np.einsum("mij,mj->mi", L_all, rng.standard_normal(X.shape))
        h[:, k] = X[:, 0] + X[:, 1]
    alpha = np.array([2 * h[owner == q][0::2].sum(axis=0) for q in FIT_MODES]) / R
    beta = np.array([-2 * h[owner == q][1::2].sum(axis=0) for q in FIT_MODES]) / R
    return alpha, beta


def wall_radius(phi, alpha, beta):
    """b(phi) = R0 + h(phi) on the unit cell, R0 fixed by the area: R0^2 = 1 - (1/2) sum (alpha^2 + beta^2)."""
    ns = np.array(FIT_MODES)
    R0 = np.sqrt(1.0 - 0.5 * np.sum(alpha**2 + beta**2))
    return R0 + np.cos(np.outer(phi, ns)) @ alpha + np.sin(np.outer(phi, ns)) @ beta


def stokes_streamfunction(rho, phi, d_alpha, d_beta):
    """Stream function of the no-slip Stokes flow inside the unit disk driven by a wall increment
    (d_alpha_n, d_beta_n), with an inextensible contour: psi_n = g_n(rho) sin n phi (alpha) or
    -g_n cos n phi (beta), g_n = (n+1)/(2n) rho^n + (1-n)/(2n) rho^{n+2}; velocity v_r = psi_phi / r,
    v_phi = -psi_r. The flow is kinematic: the viscosity does not enter."""
    psi = np.zeros(np.broadcast(rho, phi).shape)
    for n, da, db in zip(FIT_MODES, d_alpha, d_beta):
        g = (n + 1) / (2 * n) * rho**n + (1 - n) / (2 * n) * rho ** (n + 2)
        psi = psi + g * (da * np.sin(n * phi) - db * np.cos(n * phi))
    return psi


def wall_correction(a0, b0, a1, b1, n_fine=4096):
    """c(phi) on a fine periodic grid: the O(amplitude x increment) correction to the stream function that
    makes the deformed wall an exact material line, c' = (1/2)(b1^2 - b0^2) - d psi_lin(1, phi)/d phi."""
    phi = (np.arange(n_fine) + 0.5) * 2 * np.pi / n_fine
    resid = 0.5 * (wall_radius(phi, a1, b1) ** 2 - wall_radius(phi, a0, b0) ** 2)
    for n, da, db in zip(FIT_MODES, a1 - a0, b1 - b0):
        resid -= da * np.cos(n * phi) + db * np.sin(n * phi)
    c = np.concatenate([[0.0], np.cumsum(resid)]) * (2 * np.pi / n_fine)
    return np.concatenate([[0.0], phi + np.pi / n_fine]), c


def first_passage_pair(job):
    """Paired first-passage times to r < r_target for N molecules in a MOVING and a FROZEN copy of one wall.
    Units: lengths in R, time in R^2/D. Molecules start in one membrane channel (phi < 2 pi/144, outer 1/24
    of the radius). Per wall interval of 1 ms, split into m sub-steps so that the wall moves at most one
    Brownian step per sub-step (wall linear in its coefficients within the interval):
      moving: carried by the Stokes flow of the wall increment (Heun step on the stream function), held
              inside the new wall, Brownian step sqrt(2 dt/m) xi, radial reflection at the new wall;
      frozen: the wall held at its t = 0 shape, no flow, the SAME kicks xi (common random numbers).
    Returns the hitting times (nan = not reached by T_max)."""
    alpha, beta, dt, N, T_max, r_target, seed = (
        job[k] for k in ("alpha", "beta", "dt", "N", "T_max", "r_target", "seed")
    )
    rng = np.random.default_rng(seed)
    K = min(alpha.shape[1] - 1, int(np.ceil(T_max / dt)))
    a00, b00 = alpha[:, 0], beta[:, 0]
    n_rings = 24
    phi = rng.random(N) * 2 * np.pi / (6 * n_rings)
    lo2 = (1 - 1 / n_rings) ** 2
    r = np.sqrt(lo2 + (1 - lo2) * rng.random(N)) * wall_radius(phi, a00, b00)
    pos = {arm: [r * np.cos(phi), r * np.sin(phi)] for arm in ("moving", "frozen")}
    hit = {arm: np.full(N, np.nan) for arm in pos}
    eps = 1e-6

    def psi_physical(x, y, am, bm, dal, dbe, cgrid):
        rr, pp = np.hypot(x, y), np.mod(np.arctan2(y, x), 2 * np.pi)
        rho = rr / wall_radius(pp, am, bm)
        return stokes_streamfunction(rho, pp, dal, dbe) + rho**2 * np.interp(pp, *cgrid)

    def displacement(x, y, *args):
        return (
            (psi_physical(x, y + eps, *args) - psi_physical(x, y - eps, *args)) / (2 * eps),
            -(psi_physical(x + eps, y, *args) - psi_physical(x - eps, y, *args)) / (2 * eps),
        )

    def reflect(x, y, a, b):
        rr, pp = np.hypot(x, y), np.arctan2(y, x)
        bb = wall_radius(pp, a, b)
        rr = np.where(rr > bb, np.maximum(2 * bb - rr, 0.0), rr)
        return rr * np.cos(pp), rr * np.sin(pp)

    step_sd = np.sqrt(2 * dt)
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    radii = np.array([wall_radius(angles, alpha[:, k], beta[:, k]) for k in range(K + 1)])
    substeps = np.maximum(1, np.ceil((np.abs(np.diff(radii, axis=0)).max(axis=1) / step_sd) ** 2)).astype(int)
    alive = np.ones(N, bool)
    for k in range(1, K + 1):
        m = substeps[k - 1]
        sd = step_sd / np.sqrt(m)
        for j in range(m):
            idx = np.flatnonzero(alive)
            if idx.size == 0:
                break
            gx, gy = rng.standard_normal(idx.size), rng.standard_normal(idx.size)
            a0 = alpha[:, k - 1] + j / m * (alpha[:, k] - alpha[:, k - 1])
            b0 = beta[:, k - 1] + j / m * (beta[:, k] - beta[:, k - 1])
            a1 = alpha[:, k - 1] + (j + 1) / m * (alpha[:, k] - alpha[:, k - 1])
            b1 = beta[:, k - 1] + (j + 1) / m * (beta[:, k] - beta[:, k - 1])
            x, y = pos["moving"][0][idx], pos["moving"][1][idx]
            args = (0.5 * (a0 + a1), 0.5 * (b0 + b1), a1 - a0, b1 - b0, wall_correction(a0, b0, a1, b1))
            ux, uy = displacement(x, y, *args)
            vx, vy = displacement(x + ux, y + uy, *args)
            x, y = x + 0.5 * (ux + vx), y + 0.5 * (uy + vy)
            rr, pp = np.hypot(x, y), np.arctan2(y, x)
            rr = np.minimum(rr, wall_radius(pp, a1, b1))
            pos["moving"][0][idx], pos["moving"][1][idx] = reflect(
                rr * np.cos(pp) + sd * gx, rr * np.sin(pp) + sd * gy, a1, b1
            )
            pos["frozen"][0][idx], pos["frozen"][1][idx] = reflect(
                pos["frozen"][0][idx] + sd * gx, pos["frozen"][1][idx] + sd * gy, a00, b00
            )
            t_now = (k - 1 + (j + 1) / m) * dt
            for arm in pos:
                arrived = idx[(np.hypot(pos[arm][0][idx], pos[arm][1][idx]) < r_target) & np.isnan(hit[arm][idx])]
                hit[arm][arrived] = t_now
            alive = np.isnan(hit["moving"]) | np.isnan(hit["frozen"])
    return {
        "key": job["key"],
        "moving": hit["moving"],
        "frozen": hit["frozen"],
        "mean_substeps": float(substeps.mean()),
    }


def decay_comparison(t_moving, t_frozen, k):
    """With first-order decay at rate k a molecule that would arrive at tau does so with probability
    e^{-k tau}, so P = E e^{-k tau} and the mean arrival time of the arrivals is T = E[tau e^{-k tau}] / P
    (censored molecules have weight 0). Paired moving - frozen differences, with per-molecule SEs (delta
    method for the ratio T)."""

    def weights(t):
        return np.where(np.isnan(t), 0.0, np.exp(-k * np.nan_to_num(t))), np.nan_to_num(t)

    wm, tm = weights(t_moving)
    wf, tf = weights(t_frozen)
    n = wm.size
    Pm, Pf = wm.mean(), wf.mean()
    Tm, Tf = (wm * tm).mean() / Pm, (wf * tf).mean() / Pf
    influence = wm * (tm - Tm) / Pm - wf * (tf - Tf) / Pf
    return {
        "P_frozen": Pf,
        "dP": Pm - Pf,
        "se_dP": float((wm - wf).std(ddof=1) / np.sqrt(n)),
        "T_frozen": Tf,
        "dT": Tm - Tf,
        "se_dT": float(influence.std(ddof=1) / np.sqrt(n)),
    }


def summarize_transport(results, decay_rates, tscale):
    """For each decay rate: the antithetic even part (d(+wall) + d(-wall))/2 per wall history (the linear
    Gaussian wall law is symmetric under h -> -h, so the odd part has mean zero), averaged over histories.
    SE = the larger of the particle SE and the spread over histories."""
    by_path = {}
    for r in results:
        path, sign = r["key"]
        by_path.setdefault(path, {})[sign] = r
    rows = []
    for k in decay_rates:
        even = []
        for arms in by_path.values():
            d = [decay_comparison(arms[s]["moving"], arms[s]["frozen"], k) for s in (1, -1)]
            even.append(
                [0.5 * (d[0][f] + d[1][f]) for f in ("dP", "dT", "P_frozen", "T_frozen")]
                + [0.5 * np.hypot(d[0][f], d[1][f]) for f in ("se_dP", "se_dT")]
            )
        e = np.array(even)
        n = len(e)
        se_paths = e[:, :2].std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(2, np.nan)
        se_particles = np.sqrt((e[:, 4:6] ** 2).sum(axis=0)) / n
        se = np.fmax(se_paths, se_particles)
        P0, T0 = e[:, 2].mean(), e[:, 3].mean()
        rows.append(
            {
                "k_nondim": k,
                "k_per_s": k / tscale,
                "n_walls": n,
                "P_frozen": P0,
                "T_frozen_ms": T0 * tscale * 1e3,
                "dP_pct": 100 * e[:, 0].mean() / P0,
                "se_dP_pct": 100 * se[0] / P0,
                "dT_pct": 100 * e[:, 1].mean() / T0,
                "se_dT_pct": 100 * se[1] / T0,
            }
        )
    return rows


def run_transport(fits, cell, state, args, workers):
    """Simulate wall histories from the cell's A wall model, run paired molecules on +-wall, summarize."""
    fit = next(f for f in fits if f["cell"] == cell and f["state"] == state)
    p, R = fit["A_wall"]["params"], fit["R_m"]
    tscale = (R * 1e6) ** 2 / args.D  # seconds per unit R^2/D
    jobs, rms = [], []
    for path in range(args.transport_paths):
        rng = np.random.default_rng([zlib.crc32(f"wall{state}{cell}".encode()), path])
        alpha, beta = simulate_wall(p, R, args.T_max * tscale, rng)
        rms.append(np.sqrt(0.5 * np.mean(np.sum(alpha**2 + beta**2, axis=0))) * R * 1e9)
        for sign in (1, -1):
            jobs.append(
                {
                    "key": (path, sign),
                    "alpha": sign * alpha,
                    "beta": sign * beta,
                    "dt": FRAME_DT / tscale,
                    "N": args.transport_N,
                    "T_max": args.T_max,
                    "r_target": 0.3,
                    "seed": [zlib.crc32(b"molecules"), path, int(sign > 0)],
                }
            )
    if np.mean(rms) > 0.05 * R * 1e9:
        raise ValueError(
            f"simulated wall rms {np.mean(rms):.3g} nm is not a small perturbation of R; check the wall fit"
        )
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(first_passage_pair, jobs))
    mfpt = []
    for r in results:
        both = ~np.isnan(r["moving"]) & ~np.isnan(r["frozen"])
        mfpt.append(
            {
                "key": r["key"],
                "frozen_mean_ms": float(np.nanmean(r["frozen"]) * tscale * 1e3),
                "paired_diff_pct": float(
                    100 * np.mean(r["moving"][both] - r["frozen"][both]) / np.mean(r["frozen"][both])
                ),
                "censored": float(np.mean(np.isnan(r["frozen"]))),
            }
        )
    example = results[0]
    return {
        "cell": cell,
        "state": state,
        "R_um": R * 1e6,
        "D_um2_s": args.D,
        "tscale_s": tscale,
        "wall_params": p,
        "wall_rms_nm": float(np.mean(rms)),
        "decay": summarize_transport(results, args.decay, tscale),
        "single_wall": mfpt,
        "example_hits_s": {arm: (example[arm] * tscale).tolist() for arm in ("moving", "frozen")},
    }


# =====================================================================================================
# FIGURES (every axis labelled with units; data source and n in the titles)
# =====================================================================================================
def make_figures(out_dir, explore, fits, profiles, selection, null, transport):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)
    source = "Caselli et al. 2024 deposit"
    first = explore[0]
    track = load_track(out_dir, first["cell"], first["state"])
    d_nm = displacement_m(track) * 1e9
    label = video_label(first["cell"], first["state"])

    # Figure 1: the data and the tracked wall
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    img = track["mean_frame"]
    theta, R0, C0 = track["theta"], track["radii_px"][0], track["centres_px"][0]
    ax[0].imshow(img, cmap="gray")
    ax[0].plot(C0[0] + R0 * np.cos(theta), C0[1] + R0 * np.sin(theta), "r", lw=0.8, label="tracked contour")
    ax[0].plot([15, 15 + 2000 / NM_PER_PX], [285, 285], "w", lw=3)
    ax[0].text(15 + 1000 / NM_PER_PX, 278, "2 µm", color="w", ha="center")
    ax[0].set_title(f"(a) frame 1, {label}\n{source}", fontsize=10)
    ax[0].axis("off")
    n_show = min(1000, len(d_nm))
    im = ax[1].imshow(
        d_nm[:n_show].T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-80,
        vmax=80,
        extent=[0, n_show * FRAME_DT, -180, 180],
    )
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("angle around the contour [deg]")
    ax[1].set_title(f"(b) wall displacement δh(θ, t), first {n_show} frames (1 kHz)", fontsize=10)
    fig.colorbar(im, ax=ax[1], label="δh [nm] (red outward, blue inward)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_data.png", dpi=150)
    plt.close(fig)

    # Figure 2: is the motion Gaussian, and does it have memory?
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    z = (d_nm / d_nm.std()).ravel()
    x = np.linspace(-5, 5, 400)
    ax[0].hist(z, bins=np.linspace(-5, 5, 81), density=True, color="0.6", label=f"{label}, all contour points")
    ax[0].plot(x, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), "k", label="standard Gaussian")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1)
    ax[0].set_xlabel("δh / SD")
    ax[0].set_ylabel("probability density")
    ax[0].set_title(f"(a) displacement distribution, kurtosis {first['kurtosis_pooled']:.2f} (Gaussian 3)", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower center")
    ys = np.arange(len(explore))
    for yi, e in zip(ys, explore):
        lo, hi = e["kurtosis_modes_null_95"]
        ax[1].plot([lo, hi], [yi, yi], color="0.6", lw=4, solid_capstyle="butt")
        ax[1].plot(e["kurtosis_modes"], yi, "o", color="C3" if e["state"] == "free" else "C0")
    ax[1].axvline(3, color="k", lw=0.8, ls="--", label="Gaussian (3)")
    ax[1].set_yticks(ys, [video_label(e["cell"], e["state"]) for e in explore], fontsize=8)
    ax[1].set_xlabel("mean kurtosis of the mode components q = 2–6")
    ax[1].set_title("(b) dot: data; bar: 95% of Gaussian surrogates", fontsize=10)
    ax[1].legend(fontsize=8)
    lags = np.arange(0, 201)
    for e in explore:
        y = contour_modes(displacement_m(load_track(out_dir, e["cell"], e["state"])))[:, 2].real
        y = y - y.mean()
        acf = [1.0] + [np.mean(y[:-L] * y[L:]) / y.var() for L in lags[1:]]
        ax[2].plot(lags, acf, color="C3" if e["state"] == "free" else "C0", lw=0.9)
    ax[2].plot(
        lags,
        np.exp(-lags / first["tau_q2_ms"]),
        "k--",
        lw=1.3,
        label=f"exp(−t/τ), τ = {first['tau_q2_ms']:.0f} ms ({label})",
    )
    ax[2].axhline(0, color="0.6", lw=0.6)
    ax[2].set_xlabel("lag [ms]")
    ax[2].set_ylabel("autocorrelation of Re h₂")
    ax[2].set_title(f"(c) memory, {len(explore)} videos (red free, blue trapped)", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_stochastic.png", dpi=150)
    plt.close(fig)

    # Figure 3: model spectra against the data
    fit = next(f for f in fits if (f["cell"], f["state"]) == (first["cell"], first["state"]))
    _, binned, R = video_data(out_dir, first["cell"], first["state"])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for a_, q in zip(ax, (2, 4, 6)):
        q_, w, I = next(b for b in binned if b[0] == q)
        f_hz = w / (2 * np.pi)
        a_.loglog(f_hz, I * 1e18, ".", color="0.6", ms=3, label=f"data (bins of {BIN} ordinates)")
        for model, col in (("P", "C0"), ("A", "C3")):
            a_.loglog(
                f_hz,
                model_spectrum(w, q, fit[model]["params"], R, L_MAX_FIT) * 1e18,
                color=col,
                label=f"{'passive P' if model == 'P' else 'colored A'} fit",
            )
        a_.set_xlabel("frequency [Hz]")
        a_.set_title(f"contour mode q = {q} (real part), {label}", fontsize=10)
    ax[0].set_ylabel("power spectral density [nm² s]")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_spectra.png", dpi=150)
    plt.close(fig)

    # Figure 4: identifiability (95% profile-likelihood intervals)
    panels = [
        ("P", "kappa_kT", 1.0, "bending κ [k_BT]"),
        ("P", "eta", 1e3, "viscosity η [mPa·s]"),
        ("A", "tau_a", 1e3, "forcing memory τ_a [ms]"),
        ("A", "a", 1.0, "forcing strength a [× thermal]"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    ys = np.arange(len(profiles))[::-1]
    for a_, (model, name, scale, xlabel) in zip(ax, panels):
        closed = 0
        for yi, pr in zip(ys, profiles):
            iv = pr[model]["intervals"].get(name)
            if iv is None:
                continue
            col = "C3" if pr["state"] == "free" else "C0"
            a_.plot(
                [iv["lo"] * scale, iv["hi"] * scale],
                [yi, yi],
                color=col,
                lw=2.2 if not (iv["open_lo"] or iv["open_hi"]) else 0.8,
            )
            a_.plot(pr[model]["estimate"][name] * scale, yi, "ko", ms=4)
            closed += not (iv["open_lo"] or iv["open_hi"])
        a_.set_xscale("log")
        a_.set_xlabel(xlabel)
        a_.set_title(f"{'passive P' if model == 'P' else 'colored A'}: closed in {closed}/{len(profiles)}", fontsize=10)
    ax[0].set_yticks(ys, [video_label(p["cell"], p["state"]) for p in profiles], fontsize=8)
    fig.suptitle(
        "95% profile-likelihood intervals (thick: closed; thin: open = not identifiable; "
        "dot: estimate; profile threshold 1.92)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_intervals.png", dpi=150)
    plt.close(fig)

    # Figure 5: model selection against the calibrated null
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [video_label(s["cell"], s["state"]) for s in selection]
    ax.bar(
        range(len(selection)),
        [s["dAIC"] for s in selection],
        color=["C3" if s["state"] == "free" else "C0" for s in selection],
    )
    for gen, ls in zip(("G0", "G1", "G2"), (":", "--", "-.")):
        if null.get("q95", {}).get(gen) is not None:
            ax.axhline(
                null["q95"][gen],
                color="k",
                ls=ls,
                lw=1,
                label=f"{gen} null, 95th percentile ({null['n'][gen]} replicates)",
            )
    ax.set_xticks(range(len(selection)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔAIC = AIC(P) − AIC(A)")
    ax.set_title(
        "Colored (A) vs passive (P) model; nulls: passive generators G0 (P), G1 (viscosity\n"
        "contrast 10), G2 (spread of rates), simulated and refitted like the data",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_model_selection.png", dpi=150)
    plt.close(fig)

    # Figure 6: transport with decay
    if transport:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        t = np.linspace(0, transport["tscale_s"] * 1.0, 300)
        for arm, col in (("frozen", "k"), ("moving", "C3")):
            hits = np.array(transport["example_hits_s"][arm], float)
            ax[0].plot(
                t * 1e3,
                [np.mean(hits <= ti) for ti in t],
                color=col,
                lw=1.5 if arm == "frozen" else 1,
                ls="-" if arm == "frozen" else "--",
                label=f"{arm} wall",
            )
        ax[0].set_xlabel("time since entry [ms]")
        ax[0].set_ylabel("share reached r < 0.3R (no decay)")
        ax[0].set_title(
            f"(a) one wall history, {transport['state']} {transport['cell']}, D = {transport['D_um2_s']:g} µm²/s",
            fontsize=10,
        )
        ax[0].legend(fontsize=8)
        rows = transport["decay"]
        k = np.array([r["k_per_s"] for r in rows])
        for key, se, col, lab in (
            ("dP_pct", "se_dP_pct", "C0", "share arriving before decay"),
            ("dT_pct", "se_dT_pct", "C3", "mean arrival time"),
        ):
            ax[1].errorbar(
                k, [r[key] for r in rows], yerr=[2 * r[se] for r in rows], fmt="o-", color=col, capsize=3, label=lab
            )
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("decay rate k [1/s]")
        ax[1].set_ylabel("moving − frozen [%]  (± 2 SE)")
        ax[1].set_title(f"(b) wall-motion effect, {rows[0]['n_walls']} walls × ± sign, cytosol carried", fontsize=10)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig6_transport.png", dpi=150)
        plt.close(fig)


# =====================================================================================================
# REPORT
# =====================================================================================================
def answers(explore, profiles, selection, null, transport):
    """The answers to Q1 and Q2 as sentences built from the results, so they cannot drift from the numbers."""

    def closed(model, name):
        ivs = [pr[model]["intervals"][name] for pr in profiles if name in pr[model]["intervals"]]
        return sum(not (iv["open_lo"] or iv["open_hi"]) for iv in ivs), len(ivs)

    gaussian = sum(lo <= e["kurtosis_modes"] <= hi for e in explore for lo, hi in [e["kurtosis_modes_null_95"]])
    taus = [e["tau_q2_ms"] for e in explore]
    out = [
        f"  Stochastic: the wall moves by {min(e['rms_nm'] for e in explore):.0f}-{max(e['rms_nm'] for e in explore):.0f}"
        f" nm rms; its shape modes are Gaussian (inside the surrogate band) in {gaussian}/{len(explore)} videos, with"
        f" memory: tau(q=2) = {min(taus):.0f}-{max(taus):.0f} ms.",
    ]
    ident = [
        f"{m} {n} {c}/{t}"
        for m, n in (("P", "kappa_kT"), ("P", "eta"), ("A", "tau_a"), ("A", "a"), ("A", "kappa_kT"))
        for c, t in [closed(m, n)]
        if t
    ]
    out.append("  Q1 identifiability (closed 95% intervals): " + ", ".join(ident) + ".")
    for gen in ("G0", "G1", "G2"):
        if null.get("q95", {}).get(gen) is not None:
            exceed = sum(s["dAIC"] > null["q95"][gen] for s in selection)
            out.append(
                f"  Q1 model selection: Delta AIC {min(s['dAIC'] for s in selection):+.0f} to "
                f"{max(s['dAIC'] for s in selection):+.0f}; above the {gen} null's 95th percentile "
                f"({null['q95'][gen]:+.0f}) in {exceed}/{len(selection)} videos."
            )
    if transport:
        rows = transport["decay"]
        bound = max(max(abs(r["dP_pct"]) + 2 * r["se_dP_pct"], abs(r["dT_pct"]) + 2 * r["se_dT_pct"]) for r in rows)
        significant = sum(
            abs(r[k]) > 2 * r[s] for r in rows for k, s in (("dP_pct", "se_dP_pct"), ("dT_pct", "se_dT_pct"))
        )
        out.append(
            f"  Q2: over decay rates {rows[0]['k_per_s']:.1f}-{rows[-1]['k_per_s']:.1f}/s, wall motion changes the "
            f"share arriving and the travel time by less than {bound:.2f}% (2 SE); {significant}/{2 * len(rows)} "
            f"changes differ from zero by more than 2 SE."
        )
    return out


def write_report(out_dir, explore, fits, profiles, selection, null, transport):
    lines = ["STOCHASTIC CELL WALLS - pipeline report", "=" * 40, ""]
    lines += [
        "1-2 DATA AND PREPROCESSING (per video)",
        f"{'video':15s} {'frames':>6s} {'R um':>5s} {'rms nm':>6s} {'floor nm':>8s} {'bound nm':>8s} {'gain':>5s}",
    ]
    for e in explore:
        lines.append(
            f"{video_label(e['cell'], e['state']):15s} {e['frames']:6d} {e['R_um']:5.2f} {e['rms_nm']:6.1f} "
            f"{e['noise_floor_nm']:8.1f} {e['noise_bound_nm']:8.1f} {e['gain']:5.3f}"
        )
    lines += [
        "  floor: tracker noise on static synthetic frames; bound: sqrt(MSD(1 ms)/2), the noise level at which",
        "  the whole 1 ms displacement would be noise; gain: recovered/imposed motion on moving synthetic frames.",
        "",
    ]
    lines += ["3 ANALYSIS: is the motion stochastic and Gaussian, with memory?"]
    outside = 0
    for e in explore:
        lo, hi = e["kurtosis_modes_null_95"]
        out = not lo <= e["kurtosis_modes"] <= hi
        outside += out
        lines.append(
            f"{video_label(e['cell'], e['state']):15s} pooled kurtosis {e['kurtosis_pooled']:.2f} | mode kurtosis "
            f"{e['kurtosis_modes']:.3f}, Gaussian surrogates [{lo:.3f}, {hi:.3f}]{'  OUTSIDE' if out else ''} | "
            f"tau(q=2) {e['tau_q2_ms']:.0f} ms | modes 2-6 carry {100 * e['share_q2_6']:.0f}% of shape variance"
        )
    lines += [
        f"  Mode kurtosis outside the Gaussian surrogate band in {outside}/{len(explore)} videos. The pooled kurtosis mixes",
        "  sites of different variance, so it exceeds 3 even for a Gaussian process (scale mixture).",
        "",
    ]
    lines += ["3 ANALYSIS: fitted parameters (binned Whittle, natural parameters; 95% profile intervals, < > = open)"]
    for pr in profiles:
        parts = []
        for model, name, scale, unit in (
            ("P", "kappa_kT", 1, "kT"),
            ("P", "eta", 1e3, "mPa s"),
            ("A", "tau_a", 1e3, "ms"),
            ("A", "a", 1, ""),
        ):
            iv = pr[model]["intervals"].get(name)
            if iv:
                parts.append(
                    f"{model} {name} {pr[model]['estimate'][name] * scale:.3g} "
                    f"{'<' if iv['open_lo'] else '['}{iv['lo'] * scale:.3g}, {iv['hi'] * scale:.3g}"
                    f"{'>' if iv['open_hi'] else ']'} {unit}"
                )
        lines.append(f"{video_label(pr['cell'], pr['state']):15s} " + " | ".join(parts))
    for model in ("P", "A"):
        for name in PARAMS[model][:-1]:
            ivs = [pr[model]["intervals"][name] for pr in profiles if name in pr[model]["intervals"]]
            if ivs:
                closed = sum(not (iv["open_lo"] or iv["open_hi"]) for iv in ivs)
                lines.append(f"  {model} {name}: closed (identifiable) in {closed}/{len(ivs)} videos")
    lines.append("")
    lines += ["4 EVALUATION: model selection, Delta AIC = AIC(P) - AIC(A) (unbinned Whittle, l_max 80)"]
    for s in selection:
        lines.append(f"{video_label(s['cell'], s['state']):15s} Delta AIC {s['dAIC']:+9.1f}")
    for gen in ("G0", "G1", "G2"):
        if null.get("q95", {}).get(gen) is not None:
            exceed = sum(s["dAIC"] > null["q95"][gen] for s in selection)
            lines.append(
                f"  null {gen}: {null['n'][gen]} replicates, median {null['median'][gen]:+.1f}, 95th percentile "
                f"{null['q95'][gen]:+.1f}; data exceed it in {exceed}/{len(selection)} videos"
            )
    lines.append("  goodness of fit, chi^2 per bin (~1 if adequate) and share of |residual| > 2 (~0.05):")
    for f in fits:
        lines.append(
            f"  {video_label(f['cell'], f['state']):15s} P {f['gof']['P']['chi2_per_bin']:.2f} "
            f"({f['gof']['P']['share_outside_2']:.2f}) | A {f['gof']['A']['chi2_per_bin']:.2f} "
            f"({f['gof']['A']['share_outside_2']:.2f})"
        )
    lines.append("")
    if transport:
        measured_rms = next(
            e["rms_q2_6_nm"] for e in explore if (e["cell"], e["state"]) == (transport["cell"], transport["state"])
        )
        lines += [
            f"5 INTERPRETATION: travel from a membrane channel to r < 0.3R with decay ({transport['state']} "
            f"{transport['cell']}, R = {transport['R_um']:.2f} um, D = {transport['D_um2_s']:g} um^2/s, cytosol carried)",
            f"  wall model (A's a, tau_a; P's kappa; rest refitted): {', '.join(f'{k} {v:.3g}' for k, v in transport['wall_params'].items())}",
            f"  simulated wall rms over q = 2-6: {transport['wall_rms_nm']:.1f} nm (instantaneous); measured "
            f"{measured_rms:.1f} nm (exposure-averaged, includes tracking noise)",
            f"{'decay 1/s':>10s} {'P frozen':>9s} {'dP (moving-frozen)':>22s} {'T frozen':>9s} {'dT':>22s}",
        ]
        for r in transport["decay"]:
            lines.append(
                f"{r['k_per_s']:10.2f} {100 * r['P_frozen']:8.1f}% {r['dP_pct']:+10.3f} +- {r['se_dP_pct']:.3f}% "
                f"{r['T_frozen_ms']:7.0f}ms {r['dT_pct']:+10.3f} +- {r['se_dT_pct']:.3f}%"
            )
        bound = max(
            max(abs(r["dP_pct"]) + 2 * r["se_dP_pct"], abs(r["dT_pct"]) + 2 * r["se_dT_pct"])
            for r in transport["decay"]
        )
        single = max(abs(m["paired_diff_pct"]) for m in transport["single_wall"])
        lines += [
            f"  Every change is below {bound:.3f}% at 2 SE (averaged over each wall and its mirror image).",
            f"  A single wall history shifts the mean first-passage time by up to {single:.2f}%; that part flips",
            "  sign with the wall and cancels on average (second order in the wall amplitude).",
            "",
        ]
    lines += ["ANSWERS (computed from the sections above)"] + answers(explore, profiles, selection, null, transport)
    text = "\n".join(lines) + "\n"
    (Path(out_dir) / "report.txt").write_text(text)
    print(text)


# =====================================================================================================
# MAIN
# =====================================================================================================
STAGES = ("track", "explore", "fit", "profile", "select", "null", "transport", "report")


def save(out_dir, name, obj):
    (Path(out_dir) / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def load(out_dir, name):
    path = Path(out_dir) / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="folder holding Free_cells/ and 'Trapped Cells/'")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument(
        "--quick",
        action="store_true",
        help="two videos (free and trapped RBC1), 1000 frames, small profiles, null and transport",
    )
    ap.add_argument("--transport-cell", nargs=2, default=["RBC1", "free"], metavar=("CELL", "STATE"))
    ap.add_argument("--D", type=float, default=5.0, help="molecule diffusion coefficient [um^2/s]")
    ap.add_argument(
        "--decay",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 11.1, 30.0],
        help="decay rates in units of D/R^2 (0.4-12 per second for free RBC1 at D = 5)",
    )
    a = ap.parse_args()
    quick = a.quick
    a.videos = [("RBC1", "free"), ("RBC1", "trapped")] if quick else all_videos()
    a.n_frames = 1000 if quick else 5000
    a.n_static, a.n_moving = (50, 100) if quick else (300, 300)
    a.n_surrogates = 50 if quick else 200
    a.n_starts, a.max_iter = (2, 1500) if quick else (4, 3000)
    a.profile_iters, a.profile_points = (250, 7) if quick else (600, 11)
    a.profile_targets = (
        {"P": ["kappa_kT", "eta"], "A": ["a", "tau_a"]}
        if quick
        else {m: [n for n in PARAMS[m] if n != "e2"] for m in ("P", "A")}
    )
    a.null_cells = [("RBC1", "free")] if quick else list(NULL_CELLS)
    a.null_generators = ["G0"] if quick else ["G0", "G1", "G2"]
    a.null_reps = 2 if quick else 20
    a.transport_paths, a.transport_N, a.T_max = (2, 2000, 1.0) if quick else (12, 50000, 3.0)
    return a


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def stage(name):
        if name in args.stages:
            print(f"\n== {name} ({time.time() - t_start:.0f} s) ==", flush=True)
            return True
        return False

    with ProcessPoolExecutor(args.workers) as ex:
        if stage("track"):
            jobs = [(c, s, args.data, str(out), args.n_frames, args.n_static, args.n_moving) for c, s in args.videos]
            for msg in ex.map(track_video, jobs):
                print(msg, flush=True)
        if stage("explore"):
            save(
                out,
                "explore",
                list(ex.map(explore_video, [(c, s, str(out), args.n_surrogates) for c, s in args.videos])),
            )
        if stage("fit"):
            fits = list(ex.map(fit_video, [(c, s, str(out), args.n_starts, args.max_iter) for c, s in args.videos]))
            save(out, "fits", fits)
        if stage("profile"):
            fits = load(out, "fits")
            jobs = [
                (
                    f["cell"],
                    f["state"],
                    str(out),
                    {m: f[m]["z"] for m in ("P", "A")},
                    args.profile_targets,
                    args.profile_iters,
                    args.profile_points,
                )
                for f in fits
            ]
            save(out, "profiles", list(ex.map(profile_video, jobs)))
        if stage("select"):
            fits = load(out, "fits")
            data = [video_data(out, f["cell"], f["state"]) for f in fits]
            sel = list(
                ex.map(
                    selection_statistic,
                    [d[0] for d in data],
                    [f["R_m"] for f in fits],
                    fits,
                    [args.max_iter] * len(fits),
                )
            )
            save(out, "selection", [{"cell": f["cell"], "state": f["state"], **s} for f, s in zip(fits, sel)])
        if stage("null"):
            selection = {(s["cell"], s["state"]): s for s in load(out, "selection")}
            fits = {(f["cell"], f["state"]): f for f in load(out, "fits")}
            cells = [c for c in args.null_cells if c in fits]
            generators = {}
            for cell, state in cells:
                raw, _, R = video_data(out, cell, state)
                gens = list(
                    ex.map(
                        fit_generator,
                        [raw] * len(args.null_generators),
                        [R] * len(args.null_generators),
                        args.null_generators,
                        [selection[(cell, state)]["P"]["z"]] * len(args.null_generators),
                        [args.max_iter] * len(args.null_generators),
                    )
                )
                generators[f"{state} {cell}"] = dict(zip(args.null_generators, gens))
            jobs = [
                (
                    g,
                    c,
                    s,
                    rep,
                    generators[f"{s} {c}"][g]["params"],
                    fits[(c, s)]["R_m"],
                    args.n_frames,
                    args.n_starts,
                    args.max_iter,
                )
                for rep in range(args.null_reps)
                for c, s in cells
                for g in args.null_generators
            ]
            reps = list(ex.map(null_replicate, jobs))
            summary = {"generators": generators, "replicates": reps, "n": {}, "median": {}, "q95": {}}
            for g in args.null_generators:
                d = np.array([r["dAIC"] for r in reps if r["generator"] == g])
                summary["n"][g], summary["median"][g], summary["q95"][g] = (
                    len(d),
                    float(np.median(d)),
                    float(np.quantile(d, 0.95)),
                )
            save(out, "null", summary)
    if stage("transport"):
        cell, state = args.transport_cell
        save(out, "transport", run_transport(load(out, "fits"), cell, state, args, args.workers))
    if stage("report"):
        results = [load(out, n) for n in ("explore", "fits", "profiles", "selection", "null", "transport")]
        write_report(out, *results)
        make_figures(out, *results)
    print(f"\ndone in {time.time() - t_start:.0f} s; outputs in {out}")


if __name__ == "__main__":
    main()
