import os
import urllib.request
import zipfile

def download_and_extract(url, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    zip_path = os.path.join(extract_to, "temp.zip")
    print(f"Downloading {url}...")
    
    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    )

    try:
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
        print(f"Extracted to {extract_to}")
    except Exception as e:
        print(f"Failed to download or extract {url}: {e}")

if __name__ == "__main__":
    datasets_dir = "datasets"
    
    football_url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
    download_and_extract(football_url, os.path.join(datasets_dir, "football"))
    
    jazz_url = "http://www-personal.umich.edu/~mejn/netdata/jazz.zip"
    download_and_extract(jazz_url, os.path.join(datasets_dir, "jazz"))
    
    for root, dirs, files in os.walk(datasets_dir):
        for f in files:
            print(os.path.join(root, f))
