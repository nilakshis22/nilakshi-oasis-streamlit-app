#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:02:58 2026

@author: nilakshisenapati
"""

# app.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from numba import njit
from polarTransform import convertToCartesianImage


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(layout="wide")


# =========================================================
# UTILITIES
# =========================================================

def parse_pi_expression(expr: str) -> float:

    expr = expr.replace("π", "np.pi")

    try:
        return float(eval(expr, {"np": np}))
    except Exception:
        return None


# =========================================================
# OAM BEAM VISUALIZATION
# =========================================================

def generate_lg_beam(
    l: int,
    w0: float = 1.0,
    grid_size: int = 700,
    extent: float = 4.0
):

    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)

    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    amplitude = (
        (r / w0) ** abs(l)
        * np.exp(-(r**2) / w0**2)
    )

    phase = np.exp(1j * l * phi)

    field = amplitude * phase

    return field


def phase_color_plot(
    field: np.ndarray,
    cmap_name: str = "Greys"
):

    phase = np.angle(field)

    intensity = np.abs(field)

    phase_norm = (
        phase + np.pi
    ) / (2 * np.pi)

    intensity_norm = (
        intensity / intensity.max()
    )

    cmap = plt.get_cmap(cmap_name)

    rgb = cmap(phase_norm)[..., :3]

    rgb = rgb * intensity_norm[..., None]

    return rgb


def create_oam_figure(
    l: int,
    cmap: str = "Greys"
):

    field = generate_lg_beam(l=l)

    rgb = phase_color_plot(field, cmap)

    fig, ax = plt.subplots(
        figsize=(5, 4),
        facecolor="white"
    )

    ax.imshow(
        rgb,
        origin="lower",
        interpolation="bicubic"
    )


    ax.axis("off")

    fig.tight_layout()

    return fig


# =========================================================
# PHYSICS SIMULATION
# =========================================================

@njit
def compute_field(
    r,
    theta,
    l,
    theta_0,
    sig_ps,
    a,
    sig_a,
    A
):

    E1 = np.zeros(
        r.shape,
        dtype=np.complex128
    )

    E2 = np.zeros(
        r.shape,
        dtype=np.complex128
    )

    r1_vals = np.arange(
        0,
        2,
        0.04
    )

    theta_p_vals = np.arange(
        -np.pi,
        np.pi,
        0.002 * np.pi
    )

    for r1 in r1_vals:

        for theta_p in theta_p_vals:

            exp_common_phase = np.exp(
                1j * l * theta_p
            )

            g1 = np.exp(
                -(
                    theta_p
                    - (theta_0 / 2)
                )**2
                / (2 * sig_a**2)
            )

            g2 = np.exp(
                -(
                    theta_p
                    + (theta_0 / 2)
                )**2
                / (2 * sig_a**2)
            )

            for i in range(r.shape[0]):

                for j in range(r.shape[1]):

                    common = (
                        r1
                        * exp_common_phase
                        * np.exp(
                            -r[i, j]**2
                            / (2 * sig_ps**2)
                        )
                        * np.exp(
                            -a * r1**2
                            / (2 * sig_ps**2)
                        )
                        * np.exp(
                            -(
                                r1
                                * r[i, j]
                                * np.cos(
                                    theta[i, j]
                                    - theta_p
                                )
                                / (sig_ps**2)
                            )
                        )
                        * A
                    )

                    E1[i, j] += common * g1
                    E2[i, j] += common * g2

    return E1, E2


@st.cache_data
def run_simulation(
    l,
    theta_0
):

    w = 0.8
    D = 2
    d = 2.8
    f = 400

    sig_a = 0.002

    lambda_ = 0.633 * 10**(-3)

    k = 2 * np.pi / lambda_

    theta, r = np.meshgrid(
        np.linspace(
            np.pi / 2,
            3 * np.pi / 2,
            200
        ),
        np.arange(
            0,
            0.2,
            0.005
        )
    )

    sig_ps = (2 * f) / (k * d)

    a = (
        1
        + sig_ps**2 / (2 * D**2)
        - (
            1j
            * k
            * sig_ps**2
        ) / (2 * f)
        + sig_ps**2 / (2 * w**2)
    )

    A = 2 / (sig_ps)**2

    E1, E2 = compute_field(
        r,
        theta,
        l,
        theta_0,
        sig_ps,
        a,
        sig_a,
        A
    )

    E = E1 + E2

    I = E * np.conj(E)

    _, rr = np.meshgrid(
        np.linspace(
            np.pi / 2,
            3 * np.pi / 2,
            200
        ),
        np.linspace(
            0.08,
            0.2,
            40
        )
    )

    I_th = np.sum(
        rr * I,
        axis=0
    )

    th = np.linspace(
        np.pi / 2,
        3 * np.pi / 2,
        200
    )

    I_norm = (
        I_th / np.max(I_th)
    ).real

    a_val = I_norm

    vis = (
        (
            np.max(a_val)
            - round(a_val[100], 2)
        )
        /
        (
            np.max(a_val)
            + round(a_val[100], 2)
        )
        * 100
    )

    return (
        th,
        I_norm,
        vis,
        I.real
    )


# =========================================================
# HEADER
# =========================================================

st.title("OASIS")

st.markdown(
    "#### OAM-based Azimuthal Super-resolution Imaging Scheme"
)

st.markdown(
    """
    To know more details, please visit the paper:

    **Super-resolution imaging of azimuthal features with illumination carrying OAM**  
    *Applied Physics Letters* **128, 181107 (2026)**

    [https://doi.org/10.1063/5.0319922](https://doi.org/10.1063/5.0319922)
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Parameters")


# =========================================================
# l CONTROL
# =========================================================

if "l_val" not in st.session_state:
    st.session_state.l_val = 10


def update_l_from_slider():

    st.session_state.l_val = int(
        st.session_state.l_slider
    )

    st.session_state.l_input = (
        st.session_state.l_val
    )


def update_l_from_input():

    val = int(
        st.session_state.l_input
    )

    st.session_state.l_val = val

    st.session_state.l_slider = val


st.sidebar.slider(
    "OAM-mode index (l)",
    min_value=0,
    max_value=40,
    value=st.session_state.l_val,
    step=1,
    key="l_slider",
    on_change=update_l_from_slider
)

st.sidebar.number_input(
    "Manual l",
    min_value=0,
    max_value=40,
    step=1,
    value=st.session_state.l_val,
    key="l_input",
    on_change=update_l_from_input
)

l = st.session_state.l_val


# =========================================================
# THETA CONTROL
# =========================================================

if "theta_val" not in st.session_state:

    st.session_state.theta_val = (
        0.1 * np.pi
    )

if "theta_text" not in st.session_state:

    st.session_state.theta_text = (
        "0.1*np.pi"
    )


def update_theta_from_slider():

    val = st.session_state.theta_slider

    st.session_state.theta_val = val

    st.session_state.theta_text = (
        f"{val / np.pi:.3f}*np.pi"
    )


def update_theta_from_text():

    parsed = parse_pi_expression(
        st.session_state.theta_text
    )

    if parsed is not None:
        st.session_state.theta_val = parsed


st.sidebar.slider(
    r"Slit Separation in rad ($\alpha$)",
    0.0,
    float(np.pi),
    value=st.session_state.theta_val,
    key="theta_slider",
    on_change=update_theta_from_slider
)

st.sidebar.text_input(
    r"$\alpha$ (manual, use π)",
    key="theta_text",
    on_change=update_theta_from_text
)

theta_0 = st.session_state.theta_val


# =========================================================
# RUN BUTTON
# =========================================================

run = st.sidebar.button("Run")

if "has_run" not in st.session_state:
    st.session_state.has_run = False

if run:

    with st.spinner(
        "Running simulation..."
    ):

        (
            th,
            I,
            vis,
            I_2d
        ) = run_simulation(
            l,
            theta_0
        )

    st.session_state.result = (
        th,
        I,
        vis,
        I_2d,
        l,
        theta_0
    )

    st.session_state.has_run = True


# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2, col3, col4 = st.columns(
    [3, 1, 3, 1]
)


# =========================================================
# LEFT : OAM BEAM
# =========================================================

with col1:

    st.subheader("Input OAM Beam")

    if not st.session_state.has_run:

        st.info(
            "Click 'Run' to visualize the beam."
        )

    else:

        beam_fig = create_oam_figure(
            l,
            cmap="Greys"
        )

        st.pyplot(
            beam_fig,
            use_container_width=True
        )


# =========================================================
# CENTER : LENS
# =========================================================

with col2:

    st.markdown(
        "<br><br><br><br>",
        unsafe_allow_html=True
    )

    fig_lens, ax_lens = plt.subplots(
        figsize=(2, 6)
    )

    y = np.linspace(
        -1,
        1,
        400
    )

    x1 = -0.15 * (1 - y**2)
    x2 = 0.15 * (1 - y**2)

    ax_lens.fill_betweenx(
        y,
        x1,
        x2,
        color="lightgray",
        alpha=0.9
    )

    ax_lens.text(
        0,
        1.15,
        "Lens",
        ha="center",
        fontsize=14
    )

    ax_lens.set_xlim(-0.5, 0.5)
    ax_lens.set_ylim(-1.2, 1.2)

    ax_lens.axis("off")

    st.pyplot(
        fig_lens,
        use_container_width=True
    )


# =========================================================
# RIGHT : OUTPUT
# =========================================================

with col3:

    st.subheader("Imaging Output")

    if not st.session_state.has_run:

        st.info(
            "Click 'Run' to generate results."
        )

    else:

        (
            th,
            I,
            vis,
            I_2d,
            used_l,
            used_theta
        ) = st.session_state.result

        # =================================================
        # 2D OUTPUT
        # =================================================

        transform_result = convertToCartesianImage(
            np.transpose(I_2d),
            center=None,
            initialAngle=3 * np.pi / 2
        )

        cartesian_image = transform_result[0]

        fig2, ax2 = plt.subplots(
            figsize=(5, 4)
        )

        ax2.imshow(
            np.transpose(
                np.real(cartesian_image)
            ),
            cmap="gray",
            origin="lower"
        )

        ax2.axis("off")

        fig2.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )

        # =================================================
        # 1D OUTPUT
        # =================================================

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        ax.plot(
            th,
            I,
            linewidth=2,
            color="black"
        )

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$I(\theta)$")

        ax.set_xticks(
            np.linspace(
                np.pi / 2,
                3 * np.pi / 2,
                3
            ),
            ['-π/2', '0', 'π/2']
        )

        ax.set_yticks(
            np.linspace(
                0,
                1,
                3
            ),
            ['0', '0.5', '1']
        )

        ax.grid(alpha=0.3)

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )


# =========================================================
# RESULTS PANEL
# =========================================================

with col4:

    st.subheader("Results")

    if st.session_state.has_run:

        (
            _,
            _,
            vis,
            _,
            used_l,
            used_theta
        ) = st.session_state.result

        st.metric(
            "Visibility (%)",
            f"{vis:.2f}"
        )

        st.markdown("---")

        st.markdown(
            "**Current Parameters**"
        )

        st.write(f"l = {used_l}")

        st.write(
            f"α = {used_theta:.4f} rad"
        )

    else:

        st.info("No results yet.")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        right: 20px;
        color: gray;
        font-size: 13px;
        opacity: 0.7;
        z-index: 100;
    }
    </style>

    <div class="footer">
        Developed by <b>Nilakshi Senapati</b>
    </div>
    """,
    unsafe_allow_html=True
)