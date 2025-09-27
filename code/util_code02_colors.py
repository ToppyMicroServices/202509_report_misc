"""util_code02_colors

Centralized color palette for all figures (paper consistency).

Design choices:
  US   : Golden / ocher tone (good contrast + print friendly)
  Japan: Sky blue tone (clear on white background)

If you need ad‑hoc overrides, individual ``fig_codeXX`` scripts can expose
``--color-us`` / ``--color-jp`` style CLI switches instead of modifying this file.
Keep the palette minimal to encourage consistent visual identity.
"""

# Approximate HEX tones tuned to match draft figure style.
COLOR_US = '#d19a00'   # golden / ocher
COLOR_JP = '#56aee8'   # sky blue

# Neutral accent (e.g. credit growth, baseline reference lines)
COLOR_NEUTRAL = '#555555'

__all__ = [
    'COLOR_US', 'COLOR_JP', 'COLOR_NEUTRAL'
]
