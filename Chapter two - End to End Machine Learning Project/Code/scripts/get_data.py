from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def download_and_extract_housing_data():
    filePath = Path("../data/raw/housing.tgz")
    if not filePath.is_file():
        Path("../data").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url,filePath)
        with tarfile.open(filePath) as housing_tar:
            housing_tar.extractall(path="../data/raw")


if __name__ == "__main__":
    download_and_extract_housing_data()

