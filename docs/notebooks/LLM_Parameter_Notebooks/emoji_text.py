"""
Utility to render emoji text on matplotlib figures using PIL.

This bypasses Matplotlib's font glyph limitations by compositing
emoji-enabled text directly onto saved PNG files using Pillow.
"""

from PIL import Image, ImageDraw, ImageFont
import os


def get_emoji_font(size=20):
    """
    Find and load an emoji-capable font on the system.

    Tries (in order):
    1. Noto Color Emoji (installed via brew install --cask font-noto-color-emoji)
    2. Symbola (installed via brew install --cask font-symbola)
    3. System fallback
    """
    font_paths = [
        "/Library/Fonts/NotoColorEmoji.ttf",  # Homebrew cask
        "/Library/Fonts/Symbola.otf",  # Homebrew cask
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS system
        "/usr/share/fonts/opentype/noto/NotoColorEmoji.ttf",  # Linux
    ]

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception as e:
                print(f"Warning: Could not load font {path}: {e}")
                continue

    # Fallback to default
    print("Warning: No emoji font found, using default font")
    return ImageFont.load_default()


def overlay_emoji_text(image_path, positions, texts, font_size=24, color=(230, 237, 243)):
    """
    Overlay emoji text onto a saved PNG image using PIL.

    Args:
        image_path: Path to the PNG file to modify
        positions: List of (x, y) tuples (pixel coordinates)
        texts: List of text strings (can contain emoji)
        font_size: Size of the text (default 24)
        color: RGB tuple for text color (default GitHub dark mode light text)

    Returns:
        PIL Image object (not saved; call .save() to persist)
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = get_emoji_font(font_size)

    for (x, y), text in zip(positions, texts):
        draw.text((x, y), text, font=font, fill=color)

    return img


def inject_emoji_titles(image_path, emoji_title, output_path=None, title_y=30, title_x=50):
    """
    Simple helper: add emoji title text to the top of a figure.

    Args:
        image_path: Input PNG
        emoji_title: Title text (can include emoji)
        output_path: Where to save (default: overwrite input)
        title_y: Y position from top (pixels)
        title_x: X position from left (pixels)
    """
    if output_path is None:
        output_path = image_path

    img = overlay_emoji_text(image_path, [(title_x, title_y)], [emoji_title], font_size=28)
    img.convert("RGB").save(output_path, "PNG")
    print(f"✅ Saved emoji-enhanced figure: {output_path}")

