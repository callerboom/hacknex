import io
import json
import urllib.request
import pypdf

# 1. Create a minimal valid PDF
writer = pypdf.PdfWriter()
page = writer.add_blank_page(width=612, height=792)
pdf_buf = io.BytesIO()
writer.write(pdf_buf)
content = pdf_buf.getvalue()

boundary = '----Boundary7MA4YWxkTrZu0gW'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="sample_quiz.pdf"\r\n'
    f'Content-Type: application/pdf\r\n\r\n'
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
    print("FILENAME:", res.get("filename"))
    print("WORD URL:", res.get("word_url"))
    print("PAGE COUNT:", res.get("page_count"))

    word_url = res.get("word_url")
    if word_url:
        dl_req = urllib.request.Request(f"http://127.0.0.1:8000{word_url}")
        with urllib.request.urlopen(dl_req) as dl_resp:
            docx_data = dl_resp.read()
            print("DOWNLOADED DOCX SIZE BYTES:", len(docx_data))
            # Verify DOCX zip header
            is_zip = docx_data[:4] == b"PK\x03\x04"
            print("IS VALID DOCX ARCHIVE:", is_zip)
