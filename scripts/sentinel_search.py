"""
sentinel_search.py
Module for searching Sentinel-2 imagery using Microsoft Planetary Computer STAC API.
Filters for cloud-free RGB images.
"""
import planetary_computer
from pystac_client import Client
from datetime import datetime, timedelta

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def search_sentinel2_images(bbox, start_date, end_date, max_cloud_cover=10):
    """
    Search for Sentinel-2 images in a bounding box with low cloud coverage.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        max_cloud_cover: Maximum cloud coverage percentage (0-100)
    
    Returns:
        List of STAC items (Sentinel-2 scenes)
    """
    catalog = Client.open(STAC_API_URL)
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )
    
    items = list(search.items())
    return items

def get_rgb_image_url(item):
    """
    Get the RGB band URLs from a Sentinel-2 STAC item.
    ALWAYS returns individual bands (B04, B03, B02) to match EuroSAT preprocessing.
    
    Args:
        item: STAC item
    
    Returns:
        Dictionary with band URLs (B04, B03, B02)
    """
    # Sign the assets to get access
    signed_item = planetary_computer.sign(item)
    
    # ALWAYS use raw bands (B04, B03, B02) to match EuroSAT preprocessing
    # EuroSAT uses: gdal_translate -scale 0 2750 1 255 -b 4 -b 3 -b 2
    # Visual composite is pre-processed and won't match the training data
    bands = {}
    for band in ["B04", "B03", "B02"]:
        if band in signed_item.assets:
            bands[band] = signed_item.assets[band].href
    
    return bands

def get_item_metadata(item):
    """
    Extract useful metadata from a STAC item.
    
    Args:
        item: STAC item
    
    Returns:
        Dictionary with metadata
    """
    return {
        "id": item.id,
        "datetime": item.datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "cloud_cover": item.properties.get("eo:cloud_cover", "N/A"),
        "bbox": item.bbox,
        "thumbnail": item.assets.get("thumbnail", {}).get("href", None)
    }

if __name__ == "__main__":
    # Example usage
    bbox = [13.0, 52.0, 14.0, 53.0]  # Berlin area
    start = "2023-06-01"
    end = "2023-06-30"
    
    results = search_sentinel2_images(bbox, start, end, max_cloud_cover=5)
    print(f"Found {len(results)} Sentinel-2 scenes with <5% cloud cover")
    
    if results:
        item = results[0]
        metadata = get_item_metadata(item)
        print(f"Example: {metadata}")
        url = get_rgb_image_url(item)
        print(f"RGB URL: {url}")
