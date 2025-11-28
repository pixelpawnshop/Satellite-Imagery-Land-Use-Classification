"""
Interactive Streamlit app for EuroSAT Land Use Classification with Sentinel-2 imagery.
Users can search for Sentinel-2 images, view them, select a 64x64 region, and get predictions.
"""
import streamlit as st
import requests
import sys
import os
from PIL import Image, ImageDraw
import io
import numpy as np
from datetime import datetime, timedelta
import rasterio
from rasterio.windows import Window

# Optional: streamlit-image-coordinates for click-to-position
try:
    from streamlit_image_coordinates import streamlit_image_coordinates  # type: ignore
    HAS_IMAGE_COORDS = True
except ImportError:
    HAS_IMAGE_COORDS = False

# Add scripts to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from sentinel_search import search_sentinel2_images, get_rgb_image_url, get_item_metadata  # type: ignore

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="EuroSAT Land Use Classifier", layout="wide")
st.title("🛰️ EuroSAT Land Use Classification with Sentinel-2")

# Sidebar for search parameters
st.sidebar.header("Search Sentinel-2 Images")
st.sidebar.write("Enter coordinates for your area of interest:")

col1, col2 = st.sidebar.columns(2)
with col1:
    min_lon = st.number_input("Min Longitude", value=13.0, format="%.4f")
    min_lat = st.number_input("Min Latitude", value=52.0, format="%.4f")
with col2:
    max_lon = st.number_input("Max Longitude", value=14.0, format="%.4f")
    max_lat = st.number_input("Max Latitude", value=53.0, format="%.4f")

start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=90))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
max_cloud = st.sidebar.slider("Max Cloud Cover (%)", 0, 20, 10)

if st.sidebar.button("Search Images"):
    with st.spinner("Searching for Sentinel-2 images..."):
        bbox = [min_lon, min_lat, max_lon, max_lat]
        results = search_sentinel2_images(
            bbox, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d"), 
            max_cloud
        )
        st.session_state['search_results'] = results
        st.session_state['selected_item'] = None
        if results:
            st.sidebar.success(f"Found {len(results)} images!")
        else:
            st.sidebar.warning("No images found. Try adjusting your search.")

# Display search results
if 'search_results' in st.session_state and st.session_state['search_results']:
    st.sidebar.subheader("Available Images")
    for idx, item in enumerate(st.session_state['search_results'][:10]):  # Show first 10
        metadata = get_item_metadata(item)
        if st.sidebar.button(f"📅 {metadata['datetime'][:10]} | ☁️ {metadata['cloud_cover']:.1f}%", key=f"item_{idx}"):
            st.session_state['selected_item'] = item
            st.session_state['image_loaded'] = False

# Main area: Display selected image
if 'selected_item' in st.session_state and st.session_state['selected_item']:
    item = st.session_state['selected_item']
    metadata = get_item_metadata(item)
    
    st.subheader(f"Selected Image: {metadata['datetime']}")
    st.write(f"**Cloud Cover:** {metadata['cloud_cover']:.1f}% | **ID:** {metadata['id']}")
    
    # Load image if not already loaded
    if 'image_loaded' not in st.session_state or not st.session_state['image_loaded']:
        with st.spinner("Loading Sentinel-2 image..."):
            try:
                url_data = get_rgb_image_url(item)
                
                # Load individual bands (B04, B03, B02) and create RGB with EuroSAT preprocessing
                st.info("Loading raw Sentinel-2 bands (B04, B03, B02) with EuroSAT preprocessing...")
                
                # Find valid region using first band
                with rasterio.open(url_data["B04"]) as src:
                        height, width = src.height, src.width
                        
                        found_valid_region = False
                        
                        # Create a grid of positions to sample across the entire image
                        search_positions = []
                        for row_frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                            for col_frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                                search_positions.append((int(height * row_frac), int(width * col_frac)))
                        
                        best_position = None
                        best_mean = 0
                        
                        for row_center, col_center in search_positions:
                            row_off = max(0, min(height - 1024, row_center - 512))
                            col_off = max(0, min(width - 1024, col_center - 512))
                            window_height = min(1024, height - row_off)
                            window_width = min(1024, width - col_off)
                            
                            # Read a small sample to check if valid
                            test_window = Window(col_off + 256, row_off + 256, min(64, window_width - 256), min(64, window_height - 256))
                            test_data = src.read(1, window=test_window)
                            
                            # Calculate mean value
                            mean_val = test_data.mean()
                            
                            # Keep track of the brightest region
                            if mean_val > best_mean:
                                best_mean = mean_val
                                best_position = (row_off, col_off, window_height, window_width)
                            
                            # If we found a region with good data, use it
                            if mean_val > 100:  # Has real data
                                found_valid_region = True
                                break
                        
                        if not found_valid_region and best_mean > 1:
                            # Use the brightest region we found
                            st.info(f"Using brightest region found (mean value: {best_mean:.1f})")
                            row_off, col_off, window_height, window_width = best_position
                            found_valid_region = True
                        
                        if not found_valid_region:
                            st.error("This image appears to be all black/nodata. Try selecting a different image from the list.")
                            st.session_state['image_loaded'] = False
                            st.stop()
                    
                # Load all bands with the found valid window
                bands = []
                for band_name in ["B04", "B03", "B02"]:
                    with rasterio.open(url_data[band_name]) as src:
                        window = Window(col_off, row_off, window_width, window_height)
                        band = src.read(1, window=window)
                        bands.append(band)
                
                rgb = np.stack(bands, axis=-1)
                
                # Create two versions:
                # 1. Model version with EuroSAT preprocessing (1-255) for accurate predictions
                rgb_model = np.clip(rgb, 0, 2750)
                rgb_model = ((rgb_model / 2750.0) * 254 + 1).astype(np.uint8)
                
                # 2. Display version with better contrast (0-255) for visualization
                rgb_display = np.clip(rgb, 0, 2750)
                rgb_display = ((rgb_display / 2750.0) * 255).astype(np.uint8)
                
                img_model = Image.fromarray(rgb_model)
                img_display = Image.fromarray(rgb_display)
                
                st.session_state['sentinel_image'] = img_display  # For display
                st.session_state['sentinel_image_model'] = img_model  # For model predictions
                st.session_state['image_loaded'] = True
                st.session_state['box_position'] = [480, 480]  # Center of 1024x1024 image
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.session_state['image_loaded'] = False

    # Display image with 64x64 box
    if 'sentinel_image' in st.session_state:
        img = st.session_state['sentinel_image'].copy()
        box_x, box_y = st.session_state.get('box_position', [480, 480])
        
        # Draw 64x64 box with crosshair in center
        draw = ImageDraw.Draw(img)
        draw.rectangle([box_x, box_y, box_x + 64, box_y + 64], outline="red", width=3)
        # Draw crosshair at box center
        center_x, center_y = box_x + 32, box_y + 32
        draw.line([(center_x - 10, center_y), (center_x + 10, center_y)], fill="red", width=2)
        draw.line([(center_x, center_y - 10), (center_x, center_y + 10)], fill="red", width=2)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(img, caption="Sentinel-2 Image (1024x1024 at native 10m resolution) - Click to move red box", use_column_width=True)
            
            # Interactive box positioning
            st.write("**Position the 64x64 box:**")
            st.write("💡 Click on the image above to move the box, or use precise controls below")
            
            # Slider controls for precise positioning
            col_slider1, col_slider2 = st.columns(2)
            with col_slider1:
                new_x = st.slider("X Position", 0, 960, box_x, key="x_slider")
            with col_slider2:
                new_y = st.slider("Y Position", 0, 960, box_y, key="y_slider")
            
            # Update position if sliders changed
            if new_x != box_x or new_y != box_y:
                st.session_state['box_position'] = [new_x, new_y]
                st.rerun()
            
            # Click to position (if streamlit-image-coordinates is available)
            if HAS_IMAGE_COORDS:
                # Get click coordinates
                coords = streamlit_image_coordinates(img, key="image_click")
                
                if coords is not None:
                    # Center the box on the clicked position
                    click_x = coords["x"]
                    click_y = coords["y"]
                    
                    # Adjust for box centering (subtract half box size)
                    new_box_x = max(0, min(960, click_x - 32))
                    new_box_y = max(0, min(960, click_y - 32))
                    
                    st.session_state['box_position'] = [new_box_x, new_box_y]
                    st.rerun()
            else:
                st.info("💡 Install `streamlit-image-coordinates` for click-to-position: `pip install streamlit-image-coordinates`")
        
        with col2:
            st.subheader("Prediction")
            if st.button("🔍 Predict Land Use"):
                with st.spinner("Predicting..."):
                    # Extract 64x64 crop from MODEL version (with correct 1-255 preprocessing)
                    crop_model = st.session_state['sentinel_image_model'].crop((box_x, box_y, box_x + 64, box_y + 64))
                    
                    # Extract 64x64 crop from DISPLAY version (for better preview)
                    crop_display = st.session_state['sentinel_image'].crop((box_x, box_y, box_x + 64, box_y + 64))
                    
                    # Convert model version to bytes for API
                    buf = io.BytesIO()
                    crop_model.save(buf, format='PNG')
                    buf.seek(0)
                    
                    # Send to API
                    files = {"file": buf.getvalue()}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"**Predicted Class:** {result['predicted_class']}")
                        st.info(f"**Confidence:** {result['confidence']:.2%}")
                        
                        # Display the enhanced contrast version
                        st.image(crop_display, caption="64x64 Crop", use_column_width=True)
                    else:
                        st.error(f"Error: {response.text}")

else:
    st.info("👈 Use the sidebar to search for Sentinel-2 images in your area of interest.")
    st.write("### How to use:")
    st.write("1. Enter coordinates (longitude/latitude) for your area")
    st.write("2. Set date range and maximum cloud cover")
    st.write("3. Click 'Search Images'")
    st.write("4. Select an image from the results")
    st.write("5. Move the red box to select a 64x64 region")
    st.write("6. Click 'Predict Land Use' to get classification results")
