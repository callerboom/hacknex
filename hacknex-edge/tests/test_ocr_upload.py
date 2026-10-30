import io
import json
import urllib.request
from PIL import Image, ImageDraw

# Create image with text
img = Image.new("RGB", (400, 150), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((20, 20), "Question 1: What is Newton's First Law?", fill=(0, 0, 0))
d.text((20, 60), "Question 2: Calculate velocity if d=50m, t=5s.", fill=(0, 0, 0))
d.text((20, 100), "Answer: Velocity = 10 m/s", fill=(0, 0, 0))

buf = io.BytesIO()
img.save(buf, format="PNG")
content = buf.getvalue()

boundary = '----Boundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="physics_test.png"\r\n'
    f'Content-Type: image/png\r\n\r\n'
).encode() + content + f'\r\n--{boundary}--\r\n'.encode()

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/upload',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("STATUS:", data.get("status"))
    res = data.get("result", {})
    print("FILE TYPE:", res.get("file_type"))
    print("METHOD:", res.get("method"))
    print("CLEAN TEXT:\n", res.get("clean_text"))
    print("WORD URL:", res.get("word_url"))
