import pandas as pd
import re
from pathlib import Path

STANDARD_NAMES = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']

def detect_separator(filepath: Path) -> str:
    with open(filepath, 'r') as f:
        first_line = f.readline()
    for sep in ['; ', '; ', ', ', ',', ';']:
        if len(first_line.strip().split(sep)) >= 6:
            return sep
    return ','

def read_vel_profile(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    sep = detect_separator(filepath)
    print(f"[read_vel_profile] separator detected: {repr(sep)}")
    df = pd.read_csv(filepath, sep=re.escape(sep), engine='python')
    df.columns = df.columns.str.strip()
    if df.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns, got {df.shape[1]}. Columns found: {list(df.columns)}")
    print(f"[read_vel_profile] original columns : {list(df.columns)}")
    rename_map = {old: new for old, new in zip(df.columns[:6], STANDARD_NAMES)}
    df = df.rename(columns=rename_map)
    print(f"[read_vel_profile] renamed columns  : {list(df.columns)}")
    print(f"[read_vel_profile] rows              : {len(df)}")
    return df

def save_clean_csv(df: pd.DataFrame, source_path: Path) -> Path:
    out = source_path.with_stem(source_path.stem + '_clean')
    df.to_csv(out, index=False)
    print(f"[read_vel_profile] saved clean CSV  : {out}")
    return out

if __name__ == '__main__':
    path = Path(input("Paste the path to your CSV file: ").strip().strip('"'))
    df = read_vel_profile(path)
    save_clean_csv(df, path)
    print()
    print(df.head(10).to_string())
