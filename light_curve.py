import numpy as np
import plotly.graph_objects as go

# ── Parameters — change these to see different light curves ──────────────────

PLANET_RADIUS   = 4.0    # planet radius in Earth radii
STAR_RADIUS     = 1.0    # star radius in Solar radii
ORBITAL_PERIOD  = 8.0    # orbital period in days
NOISE_LEVEL     = 0.3    # noise level 0.0 (none) to 1.0 (heavy)
TOTAL_DAYS      = 30     # length of observation window in days
NUM_POINTS      = 600    # number of data points to generate

# ── Constants ────────────────────────────────────────────────────────────────

# 1 solar radius = 109.2 earth radii — needed to convert units for transit depth
EARTH_RADII_PER_SOLAR = 109.2

# ── Generate light curve ─────────────────────────────────────────────────────

# transit depth = (r_planet / r_star)^2, both in the same units
r_planet_solar = PLANET_RADIUS / EARTH_RADII_PER_SOLAR
depth = (r_planet_solar / STAR_RADIUS) ** 2

# transit half-duration in days
half_dur = min(0.08 * ORBITAL_PERIOD, 0.6)

# time array
t = np.linspace(0, TOTAL_DAYS, NUM_POINTS)

# flux array - start everything at 1.0
flux = np.ones(NUM_POINTS)

# apply transit dips
for i, time in enumerate(t):
    phase = (time % ORBITAL_PERIOD) / ORBITAL_PERIOD  # 0 to 1 over each orbit
    phase_frac = half_dur / ORBITAL_PERIOD

    # check if we're inside a transit window (phase near 0 or 1)
    if phase < phase_frac or phase > 1 - phase_frac:
        # offset from transit center in days
        offset = phase * ORBITAL_PERIOD if phase < 0.5 else (phase - 1) * ORBITAL_PERIOD

        # cosine taper for smooth ingress/egress
        smoothing = np.cos((offset / half_dur) * (np.pi / 2))
        flux[i] -= depth * max(0, smoothing)

# add noise
np.random.seed(42)  # fixed seed so the plot looks the same every run
noise = np.random.normal(0, NOISE_LEVEL * 0.001, NUM_POINTS)
flux += noise

# ── Compute stats to display ─────────────────────────────────────────────────

depth_ppm    = round(depth * 1e6)
duration_hrs = round(half_dur * 2 * 24, 1)
num_transits = int(TOTAL_DAYS / ORBITAL_PERIOD)

# ── Plot ─────────────────────────────────────────────────────────────────────

fig = go.Figure()

# main light curve line
fig.add_trace(go.Scatter(
    x=t,
    y=flux,
    mode="lines",
    line=dict(color="#378ADD", width=1.2),
    name="Stellar flux",
    hovertemplate="t = %{x:.2f} days<br>flux = %{y:.5f}<extra></extra>"
))

# add a dashed horizontal line at flux = 1.0 (baseline reference)
fig.add_hline(
    y=1.0,
    line=dict(color="gray", width=1, dash="dash"),
    annotation_text="Baseline",
    annotation_position="top right"
)

fig.update_layout(
    title=dict(
        text=(
            f"Stellar Light Curve — Planet {PLANET_RADIUS} R⊕ | "
            f"Period {ORBITAL_PERIOD} d | "
            f"Depth {depth_ppm:,} ppm | "
            f"Duration {duration_hrs} hrs | "
            f"{num_transits} transits"
        ),
        font=dict(size=14)
    ),
    xaxis=dict(title="Time (days)", showgrid=True, gridcolor="#eeeeee"),
    yaxis=dict(title="Relative flux", showgrid=True, gridcolor="#eeeeee",
               tickformat=".4f"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    width=1000,
    height=450,
    margin=dict(t=60, b=60, l=70, r=40)
)

fig.show()  # opens in browser automatically
