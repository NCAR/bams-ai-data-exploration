import io

from PIL import Image


def open_rgb_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    return img


def resize_image(img, width: int, height: int):
    return img.resize((width, height), resample=Image.BICUBIC)


def image_to_png_bytes(img) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_to_jpeg_bytes(img, quality=85):
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
