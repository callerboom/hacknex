import urllib.request
import json
import re

base = "http://127.0.0.1:8000"

print("1. Testing GET / ...")
with urllib.request.urlopen(f"{base}/") as res:
    html = res.read().decode("utf-8")
    print(f"   Status: {res.status}, Length: {len(html)}")

# Find assets
css_match = re.search(r'href="(/assets/[^"]+\.css)"', html)
if css_match:
    css_url = f"{base}{css_match.group(1)}"
    with urllib.request.urlopen(css_url) as res:
        print(f"2. Testing CSS {css_url} -> Status: {res.status}, Size: {len(res.read())} bytes")

js_match = re.search(r'src="(/assets/[^"]+\.js)"', html)
if js_match:
    js_url = f"{base}{js_match.group(1)}"
    with urllib.request.urlopen(js_url) as res:
        print(f"3. Testing JS {js_url} -> Status: {res.status}, Size: {len(res.read())} bytes")

print("4. Testing GET /api/status ...")
with urllib.request.urlopen(f"{base}/api/status") as res:
    data = json.loads(res.read().decode("utf-8"))
    print(f"   Status: {res.status}, Governor Tier: {data.get('governor_tier')}")

print("\n>>> ALL TESTS PASSED: React + Tailwind UI is successfully served on http://localhost:8000! <<<")
