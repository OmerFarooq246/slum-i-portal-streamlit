"""Slum detection spatial analysis dashboard."""

import shutil
import warnings

import streamlit as st
from PIL import Image

from config.models import ModelConfig, get_available_models
from config.settings import FIXED_ZOOM_LEVEL, PRESET_LOCATIONS, TILE_SIZE
from pipeline_omer import load_ENB5_Seg, perform_segmentation
from ui.map import create_interactive_map
from ui.results import display_detection_results, display_results_summary
from utils.state import init_session_state

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(
    page_title="Slum Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/fonts.css');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ── Page foundation ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #fffff8 !important;
    color: #111111 !important;
}

/* Apply Computer Modern to everything */
html, body, button, input, select, textarea,
[class*="st-"], .stMarkdown, .stText, p, li, label, caption {
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif !important;
}

/* Global text color */
body, .stMarkdown, .stText, p, li, label, caption,
[data-testid="stMarkdownContainer"] * {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    background-clip: unset !important;
    -webkit-background-clip: unset !important;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"]   { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
footer                           { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* Main container — document margins */
.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    padding-left: 4rem !important;
    padding-right: 4rem !important;
    max-width: 1400px !important;
    background-color: #fffff8 !important;
}

/* Title — LaTeX \title style */
.latex-title {
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif;
    font-size: 3rem !important;
    font-weight: normal !important;
    text-align: center !important;
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    background: none !important;
    -webkit-background-clip: unset !important;
    background-clip: unset !important;
    letter-spacing: 0.01em;
    margin-bottom: 0.15rem;
}
.latex-rule {
    border: none;
    border-top: 1px solid #111111;
    margin: 0.6rem 0 1.5rem 0;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif !important;
    font-weight: normal !important;
    color: #111111 !important;
}

/* Buttons — LaTeX-ish, clean borders */
[data-testid="stBaseButton-primary"] {
    background-color: #111111 !important;
    border: 1px solid #111111 !important;
    border-radius: 2px !important;
}
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] * {
    color: #fffff8 !important;
    -webkit-text-fill-color: #fffff8 !important;
    background-clip: unset !important;
    -webkit-background-clip: unset !important;
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif !important;
    font-size: 0.85rem !important;
    font-weight: bold !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stBaseButton-secondary"] {
    background-color: #fffff8 !important;
    border: 1px solid #aaaaaa !important;
    border-radius: 2px !important;
}
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] * {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    background-clip: unset !important;
    -webkit-background-clip: unset !important;
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif !important;
    font-size: 0.85rem !important;
    font-weight: bold !important;
    letter-spacing: 0.03em !important;
}
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    opacity: 0.8 !important;
}

/* Info box — muted paper style */
[data-testid="stInfo"] {
    background-color: #f0f0e8 !important;
    border: 1px solid #cccccc !important;
    border-radius: 2px !important;
    color: #111111 !important;
}
[data-testid="stInfo"] p, [data-testid="stInfo"] * {
    color: #111111 !important;
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif !important;
}

/* Captions */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #555555 !important;
    font-family: "Computer Modern Serif", "CMU Serif", Georgia, serif !important;
    font-size: 0.8rem !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid #cccccc !important;
    margin: 1rem 0 !important;
}

/* Map frame */
iframe[title="streamlit_folium.st_folium"] {
    border: 1px solid #aaaaaa !important;
    border-radius: 0 !important;
    display: block !important;
}


/* Column padding */
[data-testid="column"] { padding: 0 0.3rem !important; }

/* Success/warning */
[data-testid="stSuccess"] {
    background-color: #f0f0e8 !important;
    border: 1px solid #999 !important;
    border-radius: 2px !important;
    color: #111111 !important;
}
[data-testid="stSuccess"] * { color: #111111 !important; }
</style>
"""


def _get_model_for_city(city: str) -> tuple[str, ModelConfig]:
    for key, cfg in get_available_models().items():
        if city in cfg.regions:
            return key, cfg
    first_key = next(iter(get_available_models()))
    return first_key, get_available_models()[first_key]


def _city_location(city: str) -> dict:
    return PRESET_LOCATIONS.get(city, list(PRESET_LOCATIONS.values())[0])


def main() -> None:
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    ENB5_Seg = load_ENB5_Seg()

    city = st.session_state.selected_city
    model_key, model_config = _get_model_for_city(city)
    location_info = _city_location(city)

    # ── Title ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<p class='latex-title'>Slum Detection Software</p>"
        "<hr class='latex-rule'>",
        unsafe_allow_html=True,
    )

    # ── Main layout: map left, panel right ────────────────────────────────────
    map_col, panel_col = st.columns([4, 1])

    with map_col:
        map_output = create_interactive_map(location_info=location_info)

    with panel_col:
        st.markdown("**Location**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Lahore", use_container_width=True,
                         type="primary" if city == "Lahore" else "secondary"):
                st.session_state.selected_city = "Lahore"
                st.rerun()
        with c2:
            if st.button("Islamabad", use_container_width=True,
                         type="primary" if city == "Islamabad" else "secondary"):
                st.session_state.selected_city = "Islamabad"
                st.rerun()

        st.info(model_config.name)

        st.markdown("---")

        drawings = (map_output or {}).get("all_drawings") or []
        polygons = [d for d in drawings if d["geometry"]["type"] == "Polygon"]

        if polygons:
            st.markdown("**Areas drawn**")
            for idx, drawing in enumerate(polygons):
                if st.button(
                    f"Analyse area {idx + 1}",
                    key=f"process_{idx}",
                    use_container_width=True,
                    type="primary",
                ):
                    perform_segmentation(
                        polygon_coords=drawing["geometry"]["coordinates"][0],
                        model_config=model_config,
                        model=ENB5_Seg,
                    )
        else:
            st.caption("Draw a rectangle on the map to begin.")

        st.markdown("---")

        total = len(st.session_state.processing_results)
        if total:
            st.markdown(f"**{total} result{'s' if total > 1 else ''}**")
            display_results_summary()
            if st.button("Clear all", use_container_width=True):
                for r in st.session_state.processing_results.values():
                    if d := r.get("session_dir"):
                        shutil.rmtree(d, ignore_errors=True)
                st.session_state.processing_results = {}
                st.rerun()

    # ── Full-width results ────────────────────────────────────────────────────
    if st.session_state.processing_results:
        st.markdown("---")
        display_detection_results()


if __name__ == "__main__":
    main()
