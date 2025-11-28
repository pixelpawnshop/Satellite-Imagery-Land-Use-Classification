"""
download_eurosat.py
Downloads and extracts the EuroSAT dataset into the data/ folder.
"""
import os
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
EUROSAT_URL = 'http://madm.dfki.de/files/sentinel/EuroSAT.zip'
EUROSAT_ZIP = os.path.join(DATA_DIR, 'EuroSAT.zip')
EUROSAT_EXTRACTED = os.path.join(DATA_DIR, 'EuroSAT')


def download_eurosat():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EUROSAT_EXTRACTED):
        print('EuroSAT dataset already downloaded and extracted.')
        return
    print('Downloading EuroSAT dataset...')
    response = requests.get(EUROSAT_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(EUROSAT_ZIP, 'wb') as f, tqdm(
        desc='EuroSAT.zip',
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    print('Extracting EuroSAT dataset...')
    with zipfile.ZipFile(EUROSAT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print('Extraction complete.')
    os.remove(EUROSAT_ZIP)

if __name__ == '__main__':
    download_eurosat()
