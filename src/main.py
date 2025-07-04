from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import requests

parazitized_class_names = ['cellules saines', 'cellules infectées']

def download_model_from_cloud(share_url, local_path):
    """Download model from OneDrive or Google Drive share link"""
    try:
        download_url = convert_to_direct_download_url(share_url)
        
        print("Downloading model from cloud storage...")
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            # Check if response is HTML (Google Drive virus scan warning)
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print("Got HTML response, trying alternative download method...")
                return download_large_file_from_google_drive(share_url, local_path)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate the downloaded file
            if validate_h5_file(local_path):
                print("Model downloaded successfully!")
                return True
            else:
                print("Downloaded file is corrupted, removing...")
                os.remove(local_path)
                return False
        else:
            print(f"Failed to download: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def download_large_file_from_google_drive(share_url, local_path):
    """Handle large Google Drive files that trigger virus scan warning"""
    try:
        file_id = extract_google_drive_file_id(share_url)
        if not file_id:
            return False
        
        session = requests.Session()
        
        # First request to get the download warning
        response = session.get(f"https://drive.google.com/uc?export=download&id={file_id}")
        
        # Look for the download warning token
        for line in response.text.splitlines():
            if 'download_warning' in line and 'confirm=' in line:
                # Extract the confirm token
                import re
                match = re.search(r'confirm=([^&]+)', line)
                if match:
                    confirm_token = match.group(1)
                    # Download with confirm token
                    download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                    response = session.get(download_url, stream=True)
                    
                    if response.status_code == 200:
                        with open(local_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return validate_h5_file(local_path)
        return False
    except Exception as e:
        print(f"Error downloading large file: {e}")
        return False

def extract_google_drive_file_id(share_url):
    """Extract file ID from Google Drive URL"""
    if '/file/d/' in share_url:
        return share_url.split('/file/d/')[1].split('/')[0]
    elif 'id=' in share_url:
        return share_url.split('id=')[1].split('&')[0]
    return None

def validate_h5_file(file_path):
    """Validate if the file is a proper HDF5 file"""
    try:
        import h5py
        with h5py.File(file_path, 'r') as f:
            # Check if it has the basic structure of a Keras model
            return True
    except:
        return False

def convert_to_direct_download_url(share_url):
    """Convert OneDrive or Google Drive share URL to direct download URL"""
    # OneDrive URLs
    if 'onedrive.live.com' in share_url and 'view.aspx' in share_url:
        return share_url.replace('view.aspx', 'download.aspx')
    elif '1drv.ms' in share_url:
        return share_url + '&download=1'
    
    # Google Drive URLs
    elif 'drive.google.com' in share_url:
        # Extract file ID from Google Drive URL
        if '/file/d/' in share_url:
            # Format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            file_id = share_url.split('/file/d/')[1].split('/')[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        elif 'id=' in share_url:
            # Format: https://drive.google.com/open?id=FILE_ID
            file_id = share_url.split('id=')[1].split('&')[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Dropbox URLs
    elif 'dropbox.com' in share_url:
        return share_url.replace('?dl=0', '?dl=1')
    
    # Return original URL if no conversion needed
    return share_url

def load_model_with_cache(cloud_url, local_model_path='models/model2.h5'):
    """Load model from cloud storage with local caching"""
    # Check if local model exists and is valid
    if os.path.exists(local_model_path):
        if validate_h5_file(local_model_path):
            print("Loading cached model...")
            try:
                return load_model(local_model_path)
            except Exception as e:
                print(f"Cached model is corrupted: {e}")
                print("Removing corrupted cache and re-downloading...")
                os.remove(local_model_path)
        else:
            print("Cached model file is invalid, removing...")
            os.remove(local_model_path)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    
    # Download from cloud storage
    if download_model_from_cloud(cloud_url, local_model_path):
        return load_model(local_model_path)
    else:
        raise Exception("Failed to load model from cloud storage and no local cache available")