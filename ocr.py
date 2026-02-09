# ocr.py
import re

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import easyocr

# Initialize EasyOCR reader once (English only)
_reader = easyocr.Reader(['en'], gpu=False)

# Ticker pattern: 1–5 uppercase letters (simple version, good for NVDA, AMD, VOO, IVV, etc.)
TICKER_RE = re.compile(r'^[A-Z]{1,5}$')
# Number pattern: 123 or 123.456
NUMBER_RE = re.compile(r'(\d+(?:\.\d+)?)')


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


def extract_shares_from_image(uploaded_file):
    """
    Takes a file-like object (Streamlit upload) of a Robinhood screenshot,
    returns:
        positions: dict {ticker: shares}
        raw_text: the OCR'd text (for debugging).
    Logic:
    - Use EasyOCR to read text chunks in layout order
    - Look for any chunk containing 'shares'
    - Parse the number from the same or previous chunk
    - Walk backwards to find the nearest uppercase ticker-like chunk
    """
    image = Image.open(uploaded_file)
    image = preprocess_image(image)

    # EasyOCR returns a list of strings if detail=0
    chunks = _reader.readtext(np.array(image), detail=0)
    # Normalize: strip and drop empty
    chunks = [str(c).strip() for c in chunks if str(c).strip()]

    positions = {}

    for i, token in enumerate(chunks):
        token_lower = token.lower()
        if "shares" not in token_lower:
            continue

        # 1) Try to find number in the same token (e.g., "0.593853 shares")
        shares = None
        m_same = NUMBER_RE.search(token)
        if m_same:
            try:
                shares = float(m_same.group(1))
            except ValueError:
                shares = None

        # 2) If not found, maybe previous chunk is just the number:
        #    e.g., "0.593853"  (prev)  +  "shares" (current)
        if shares is None and i > 0:
            prev = chunks[i - 1]
            m_prev = NUMBER_RE.fullmatch(prev)
            if m_prev:
                try:
                    shares = float(m_prev.group(1))
                except ValueError:
                    shares = None

        if shares is None:
            # Can't find a valid number, skip this 'shares' occurrence
            continue

        # 3) Find ticker by scanning backwards a few chunks
        ticker = None
        for j in range(i - 1, max(-1, i - 6), -1):  # look back up to 5 chunks
            cand = chunks[j].strip()
            cand_clean = cand.replace(".", "")  # handle things like BRK.B crudely -> BRKB
            if TICKER_RE.fullmatch(cand_clean):
                ticker = cand_clean
                break

        if ticker is None:
            # Can't associate with a ticker, skip
            continue

        # Aggregate shares if ticker appears multiple times
        positions[ticker] = positions.get(ticker, 0.0) + shares

    raw_text = "\n".join(chunks)
    return positions, raw_text
