import ctypes
from ctypes import wintypes
import os
from PIL import Image

# Windows GDI32 and USER32
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

SRCCOPY = 0x00CC0020
SM_CXSCREEN = 0
SM_CYSCREEN = 1
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD),
        ('biWidth', wintypes.LONG),
        ('biHeight', wintypes.LONG),
        ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD),
        ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD),
        ('biXPelsPerMeter', wintypes.LONG),
        ('biYPelsPerMeter', wintypes.LONG),
        ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD),
    ]

def capture_desktop_gdi(filepath: str) -> bool:
    """Capture full desktop using native Windows GDI BitBlt and save as PNG."""
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    try:
        x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        w = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        h = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

        if w <= 0 or h <= 0:
            x = 0
            y = 0
            w = user32.GetSystemMetrics(SM_CXSCREEN)
            h = user32.GetSystemMetrics(SM_CYSCREEN)

        if w <= 0 or h <= 0:
            w, h = 1920, 1080

        hdesktop = user32.GetDesktopWindow()
        desktop_dc = user32.GetDC(hdesktop)
        if not desktop_dc:
            desktop_dc = user32.GetDC(0)

        mem_dc = gdi32.CreateCompatibleDC(desktop_dc)
        hbitmap = gdi32.CreateCompatibleBitmap(desktop_dc, w, h)
        old_bmp = gdi32.SelectObject(mem_dc, hbitmap)

        gdi32.BitBlt(mem_dc, 0, 0, w, h, desktop_dc, x, y, SRCCOPY)

        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth = w
        bi.biHeight = -h  # top-down DIB
        bi.biPlanes = 1
        bi.biBitCount = 32
        bi.biCompression = 0

        buffer_len = w * h * 4
        buf = ctypes.create_string_buffer(buffer_len)

        lines_copied = gdi32.GetDIBits(
            mem_dc, hbitmap, 0, h, buf, ctypes.byref(bi), 0
        )

        # Cleanup GDI handles
        gdi32.SelectObject(mem_dc, old_bmp)
        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hdesktop, desktop_dc)

        if lines_copied > 0:
            img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1)
            img = img.convert('RGB')
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            img.save(filepath, 'PNG')
            return True
        return False
    except Exception as e:
        print("GDI capture exception:", e)
        return False
