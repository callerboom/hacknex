import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEST = os.path.join("models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
TOTAL_SIZE = 491400032

def download_with_resume():
    os.makedirs("models", exist_ok=True)
    
    while True:
        current_size = os.path.getsize(DEST) if os.path.exists(DEST) else 0
        if current_size >= TOTAL_SIZE:
            print(f"\n[DONE] Model fully downloaded! Total size: {current_size / (1024*1024):.1f} MB")
            return True

        req = urllib.request.Request(URL)
        if current_size > 0:
            req.add_header("Range", f"bytes={current_size}-")
            print(f"[RESUME] Resuming from {current_size / (1024*1024):.1f} MB ({current_size}/{TOTAL_SIZE} bytes)")
        else:
            print(f"[START] Starting download from 0 MB ({TOTAL_SIZE / (1024*1024):.1f} MB total)")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(DEST, "ab" if current_size > 0 else "wb") as f:
                start_time = time.time()
                bytes_this_session = 0
                last_print = 0

                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_this_session += len(chunk)
                    current_size += len(chunk)

                    now = time.time()
                    if now - last_print > 1.5 or current_size >= TOTAL_SIZE:
                        last_print = now
                        pct = (current_size / TOTAL_SIZE) * 100
                        mb_done = current_size / (1024 * 1024)
                        mb_tot = TOTAL_SIZE / (1024 * 1024)
                        speed = (bytes_this_session / (1024 * 1024)) / (now - start_time) if (now - start_time) > 0 else 0
                        sys.stdout.write(f"\r  Progress: {pct:5.1f}% | {mb_done:.1f}/{mb_tot:.1f} MB | {speed:.1f} MB/s   ")
                        sys.stdout.flush()

        except Exception as e:
            print(f"\n[RECONNECTING] Connection reset ({e}). Resuming in 2 seconds...")
            time.sleep(2)
            continue

if __name__ == "__main__":
    download_with_resume()
