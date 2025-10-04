"""util_code04_jhf_excel_loader

Loader for Japan Housing Finance Agency (JHF) monthly RMBS balance Excel.

Assumptions (legacy provisional):
    * First row contains month identifiers parsable as YYYY/MM or YYYY-M.
    * Second row (index 1) holds the corresponding outstanding balance values.
    * Values are assumed in 100 million yen ("oku yen") unless caller rescales.

Output:
    Quarterly (calendar quarter-end) DataFrame with column ``JP_JHF_RMBS``.
    Aggregation uses last monthly observation of each quarter (stock variable).

Note: Parsing logic is intentionally conservative; real-world format drift may
require extending column selection heuristics.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

FREQ_Q_END = 'QE'

def load_jhf_mbs_excel(excel_path: str) -> pd.DataFrame:
    """Parse a raw JHF monthly Excel file into a quarterly stock series.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file containing monthly balances.

    Returns
    -------
    pd.DataFrame
        Index: quarter-end Timestamp (freq Q)
        Columns: ``JP_JHF_RMBS`` (float)
    """
    p = Path(excel_path)
    if not p.exists():
        raise FileNotFoundError(f"JHF Excel not found: {excel_path}")
    raw = pd.read_excel(p, header=None)
    try:
        first_row = raw.iloc[0].dropna().astype(str)
        date_like = first_row[first_row.str.contains(r"\d{4}[/\-]\d{1,2}")]
        if date_like.empty:
            raise ValueError("date header not found")
        dates = pd.to_datetime(date_like, errors="coerce")
        values_row = raw.iloc[1:2, date_like.index]
        values = values_row.T.squeeze().astype(float)
        monthly = pd.Series(values.values, index=dates, name="JP_JHF_RMBS")
    except Exception as e:
        raise ValueError(f"JHF Excel parse error: {e}")
    monthly = monthly.sort_index()
    # Convert via PeriodIndex to quarter-end (avoids deprecated resample freq warnings)
    q = monthly.to_frame().groupby(monthly.index.to_period('Q')).last()
    q.index = q.index.to_timestamp('Q')
    return q
