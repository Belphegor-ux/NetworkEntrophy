import os
import urllib.request
import tarfile
import ssl

def download_and_extract_tar(url, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    tar_path = os.path.join(extract_to, "temp.tar.bz2")
    print(f"Downloading {url}...")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    )

    try:
        with urllib.request.urlopen(req, context=ctx) as response, open(tar_path, 'wb') as out_file:
            out_file.write(response.read())
        
        with tarfile.open(tar_path, 'r:bz2') as tar_ref:
            tar_ref.extractall(extract_to)
        os.remove(tar_path)
        print(f"Extracted to {extract_to}")
    except Exception as e:
        print(f"Failed to download or extract {url}: {e}")

if __name__ == "__main__":
    datasets_dir = "datasets"
    
    jazz_url = "http://konect.cc/files/download.tsv.arenas-jazz.tar.bz2"
    download_and_extract_tar(jazz_url, os.path.join(datasets_dir, "jazz"))
    
    for root, dirs, files in os.walk(datasets_dir):
        for f in files:
            print(os.path.join(root, f))
