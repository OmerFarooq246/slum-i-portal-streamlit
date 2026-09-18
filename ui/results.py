"""Results display for spatial analysis."""

import io
import os
import shutil
import zipfile
from typing import Any

import streamlit as st


def display_results_summary() -> None:
    for session_id, result in list(st.session_state.processing_results.items()):
        st.markdown(f"**Analysis · {session_id[:8]}**")
        if "primary_result" in result:
            tiles = result["primary_result"].get("tiles_info")
            if tiles is not None:
                st.caption(f"{len(tiles)} tiles processed")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Remove", key=f"remove_{session_id}", use_container_width=True):
                _cleanup_session(session_id, result)
        with c2:
            st.download_button(
                label="Download",
                data=_build_zip(session_id, result),
                file_name=f"analysis_{session_id[:8]}.zip",
                mime="application/zip",
                key=f"dl_{session_id}",
                use_container_width=True,
            )

        st.markdown("---")


def display_detection_results() -> None:
    for result in st.session_state.processing_results.values():
        if "primary_result" in result:
            _display_date_result(result["primary_result"])
        else:
            st.warning("No results available.")


def _display_date_result(date_result: dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Original")
        if date_result.get("original_image"):
            st.image(date_result["original_image"], use_container_width=True)

    with col2:
        st.caption("Detection mask")
        if date_result.get("stitched_mask"):
            st.image(date_result["stitched_mask"], use_container_width=True)

    with col3:
        st.caption("Overlay")
        if date_result.get("overlay_image"):
            st.image(date_result["overlay_image"], use_container_width=True)


def _build_zip(session_id: str, result: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        pr = result.get("primary_result", {})
        for fname, img_key in [
            (f"overlay_{session_id[:8]}.png", "overlay_image"),
            (f"original_{session_id[:8]}.png", "original_image"),
            (f"mask_{session_id[:8]}.png", "stitched_mask"),
        ]:
            if pr.get(img_key):
                b = io.BytesIO()
                pr[img_key].save(b, format="PNG")
                zf.writestr(fname, b.getvalue())
    buf.seek(0)
    return buf.getvalue()


def _cleanup_session(session_id: str, result: dict[str, Any]) -> None:
    try:
        session_dir = result.get("session_dir")
        if session_dir and os.path.exists(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
    except Exception:
        pass
    del st.session_state.processing_results[session_id]
    st.rerun()
