# ocr.py
import re
from typing import Dict, Tuple, Union, IO

import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Ticker pattern: 1–5 uppercase letters (simple version, good for NVDA, AMD, VOO, IVV, etc.)
TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# Number pattern: 123 or 123.456 (used for searching inside a token)
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


# --- Lazy-loaded EasyOCR reader (prevents Streamlit Cloud startup crashes) ---
_reader = None

def get_reader():
    """
    Lazily import and initialize EasyOCR reader.
    This avoids ModuleNotFoundError / heavy init during app startup.
    """
    global _reader
    if _reader is None:
        import easyocr  # imported only when OCR is actually used
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Light preprocessing to help OCR:
    - convert to grayscale
    - upscale smaller images
    - median filter to reduce noise
    """
    img = ImageOps.grayscale(image)
    w, h = img.size

    # Upscale if small
    if max(w, h) < 1200:
        img = img.resize((int(w * 1.5), int(h * 1.5)))

    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img


def extract_shares_from_image(uploaded_file: Union[str, IO]) -> Tuple[Dict[str, float], str]:
    """
    Takes a file-like object (Streamlit upload) or a filepath of a Robinhood screenshot,
    returns:
        positions: dict {ticker: shares}
        raw_text: the OCR'd text (for debugging).

    Logic:
    - Use EasyOCR to read text chunks
    - Look for any chunk containing 'shares'
    - Parse the number from the same or previous chunk
    - Walk backwards to find the nearest uppercase ticker-like chunk
    """
    image = Image.open(uploaded_file)
    image = preprocess_image(image)

    reader = get_reader()

    # EasyOCR returns a list of strings if detail=0
    chunks = reader.readtext(np.array(image), detail=0)

    # Normalize: strip and drop empty
    chunks = [str(c).strip() for c in chunks if str(c).strip()]

    positions: Dict[str, float] = {}

    for i, token in enumerate(chunks):
        token_lower = token.lower()
        if "shares" not in token_lower:
            continue

        shares = None

        # 1) Try to find a number in the same token (e.g., "0.593853 shares")
        m_same = NUMBER_RE.search(token)
        if m_same:
            try:
                shares = float(m_same.group(1))
            except ValueError:
                shares = None

        # 2) If not found, maybe previous chunk is just the number:
        #    e.g., "0.593853"  (prev)  +  "shares" (current)
        if shares is None and i > 0:
            prev = chunks[i - 1].strip()
            # fullmatch against the entire token as a number
            if re.fullmatch(r"\d+(?:\.\d+)?", prev):
                try:
                    shares = float(prev)
                except ValueError:
                    shares = None

        if shares is None:
            continue

        # 3) Find ticker by scanning backwards a few chunks
        ticker = None
        for j in range(i - 1, max(-1, i - 6), -1):  # look back up to 5 chunks
            cand = chunks[j].strip()

            # crude clean: BRK.B -> BRKB (optional)
            cand_clean = cand.replace(".", "")
            if TICKER_RE.fullmatch(cand_clean):
                ticker = cand_clean
                break

        if ticker is None:
            continue

        # Aggregate shares if ticker appears multiple times
        positions[ticker] = positions.get(ticker, 0.0) + shares

    raw_text = "\n".join(chunks)
    return positions, raw_text
