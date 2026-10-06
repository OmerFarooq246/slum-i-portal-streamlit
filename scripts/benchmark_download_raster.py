import leafmap

BBOX = [
    72.99126713424454,
    33.67408465650475,
    73.00225346236954,
    33.68322704762860,
]


def main():
    print("[+] Downloading satellite imagery...")
    geotiff_path = "benchmarks/data/raster_benchmark.tif"
    leafmap.map_tiles_to_geotiff(
        output=geotiff_path,
        bbox=BBOX,
        zoom=19,
        source="SATELLITE",
        overwrite=True,
        quiet=False,
        options=["BIGTIFF=YES"],
    )


if __name__ == "__main__":
    main()
