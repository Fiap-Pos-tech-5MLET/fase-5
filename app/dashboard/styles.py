"""
Estilos customizados do dashboard.
"""

from __future__ import annotations

import streamlit as st


def apply_custom_css() -> None:
    """
    Aplica CSS customizado ao dashboard.

    Returns:
        None
    """
    st.markdown(
        """
<style>
    /* Metric cards */
    .metric-card {
        background: #1A1D29;
        border: 1px solid #2D3348;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border-color: #7C3AED;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #A78BFA;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Risk result */
    .risk-high {
        background: #1C1520;
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .risk-low {
        background: #101C18;
        border: 2px solid #22C55E;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .risk-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 10px;
    }
    .risk-prob {
        font-size: 3rem;
        font-weight: 800;
    }
    .risk-desc {
        color: #94A3B8;
        margin-top: 10px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #CBD5E1;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 30px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2D3348;
    }

    /* Info box */
    .info-box {
        background: #1A1D29;
        border: 1px solid #7C3AED;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #E2E8F0;
    }
    .info-box strong {
        color: #A78BFA;
    }

    /* Card description text */
    .card-desc {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 12px;
    }
    .card-title {
        font-size: 1rem;
        color: #E2E8F0;
    }
</style>
""",
        unsafe_allow_html=True,
    )
