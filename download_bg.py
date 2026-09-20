import urllib.request
import os

url = "https://images.unsplash.com/photo-1544735716-392fe2489ffa?q=80&w=2070&auto=format&fit=crop"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    os.makedirs("static/images", exist_ok=True)
    with open("static/images/himalaya-background.jpg", "wb") as f:
        f.write(data)
    print(f"Downloaded Himalayan landscape image ({len(data)} bytes).")
except Exception as e:
    print(f"Download failed: {e}")

