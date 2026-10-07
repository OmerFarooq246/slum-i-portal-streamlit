from typing import Any

import folium
from folium.plugins import Draw
from streamlit_folium import st_folium

_SATELLITE_TILES = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"


def create_interactive_map(location_info: dict[str, Any]) -> dict[str, Any]:
    m = folium.Map(location=location_info["center"], zoom_start=location_info["zoom"])

    folium.TileLayer(tiles=_SATELLITE_TILES, attr="Google", name="Satellite").add_to(m)

    Draw(
        draw_options={
            "rectangle": True,
            "polygon": False,
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    folium.Marker(
        location=location_info["center"],
        popup=f"📍 {location_info['description']}",
        tooltip="Preset Location",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    map_key = f"map_{location_info['center'][0]}_{location_info['center'][1]}"
    return st_folium(
        m,
        center=location_info["center"],
        zoom=location_info["zoom"],
        returned_objects=["all_drawings"],
        height=700,
        use_container_width=True,
        key=map_key,
    )
