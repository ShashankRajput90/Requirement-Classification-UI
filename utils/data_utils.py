from typing import Tuple, Optional
import pandas as pd


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect likely 'user_story' and 'label' column names in a dataframe.

    Returns (user_story_col, label_col) or (None, None) if not found.
    Tries common variations and heuristics.
    """
    candidates_story = ['user_story', 'story', 'text', 'requirement', 'requirement_text']
    candidates_label = ['label', 'gold_label', 'nfr', 'class', 'category', 'target', 'label_text']

    story_col = None
    label_col = None

    lc = {c.lower(): c for c in df.columns}

    for cand in candidates_story:
        if cand in lc:
            story_col = lc[cand]
            break

    for cand in candidates_label:
        if cand in lc:
            label_col = lc[cand]
            break

    # if still not found, try inference by datatype/unique values
    if not story_col:
        # pick the first string-like column with longer average length
        string_cols = [c for c in df.columns if df[c].dtype == object]
        if string_cols:
            # choose column with largest average length
            avg_lens = {c: df[c].astype(str).map(len).mean() for c in string_cols}
            story_col = max(avg_lens, key=avg_lens.get)

    if not label_col:
        # choose a low-cardinality string/integer column
        candidate_cols = [c for c in df.columns if df[c].nunique() < max(50, len(df) * 0.5)]
        if candidate_cols:
            # prefer columns containing FR/NFR-like values
            for c in candidate_cols:
                sample_values = df[c].astype(str).str.lower().unique()[:10]
                if any(s in ('fr', 'nfr', 'yes', 'no', 'functional', 'non-functional', 'non functional') for s in sample_values):
                    label_col = c
                    break
            if not label_col:
                # fallback to the column with smallest uniqueness
                label_col = min(candidate_cols, key=lambda x: df[x].nunique())

    return story_col, label_col


def normalize_label(value: object) -> str:
    """Normalize a label value to 'FR' or 'NFR'.

    Accepts many variants: 'yes'/'no', 'nfr'/'fr', full words, numeric codes etc.
    Defaults to original string uppercased if unknown.
    """
    if pd.isna(value):
        return 'Unknown'
    s = str(value).strip().lower()
    if s in ('yes', 'y', 'nfr', 'non-functional', 'non functional', 'nonfunctional', 'n-f-r'):
        return 'NFR'
    if s in ('no', 'n', 'fr', 'functional', 'func'):
        return 'FR'
    # numeric encodings
    if s in ('1', '0'):
        # guess: 1 -> NFR, 0 -> FR (common), but ambiguous; default to NFR for 1
        return 'NFR' if s == '1' else 'FR'
    # try to find words
    if 'non' in s and 'functional' in s:
        return 'NFR'
    if 'functional' in s and 'non' not in s:
        return 'FR'
    # otherwise return capitalized original
    return str(value).strip().upper()
