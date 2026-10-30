import os
import sys
import time
import urllib.request

URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEST = os.path.join("models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

def main():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) > 400 * 1024 * 1024:
        print(f"[OK] Model already exists at {DEST} ({os.path.getsize(DEST) / (1024*1024):.1f} MB)")
        return

    print("=" * 60)
    print("  Downloading Qwen2.5-0.5B-Instruct-Q4_K_M GGUF (~468 MB)")
    print(f"  Destination: {DEST}")
    print("=" * 60)

    start_time = time.time()
    last_print = 0

    def report_progress(block_num, block_size, total_size):
        nonlocal last_print
        downloaded = block_num * block_size
        now = time.time()
        if now - last_print > 2 or downloaded >= total_size:
            last_print = now
            pct = (downloaded / total_size) * 100 if total_size > 0 else 0
            mb_done = downloaded / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            elapsed = now - start_time
            speed = mb_done / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r  Progress: {pct:5.1f}% | {mb_done:.1f}/{mb_tot:.1f} MB | {speed:.1f} MB/s")
            sys.stdout.flush()

    try:
        temp_dest = DEST + ".tmp"
        urllib.request.urlretrieve(URL, temp_dest, reporthook=report_progress)
        print()
        if os.path.exists(DEST):
            os.remove(DEST)
        os.rename(temp_dest, DEST)
        print(f"[SUCCESS] Download completed! Saved to {DEST}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        sys.exit(1)

if __name__ == "__main__":
    main()
