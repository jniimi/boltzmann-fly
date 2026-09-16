"""The regenerated Purchase-World panel is byte-for-byte the one used in all reported runs.

The stored digests are sha256 over the column-wise contents (names, float64 values, dates as day
numbers) of the train / val / test frames generated with the ICONIP configuration (seed 42). They were
computed from the panel that was verified to match the original ICONIP experiment's data, so this test
does not need anything outside this repository.
"""
import hashlib

import numpy as np
import pandas as pd

EXPECTED = {
    "train": "484856e5dc701eb0a8cff5e8bb39ae2b520617d2e870149a0498243d9413381f",
    "val": "00f02e7ea5987eef58c3ae7a5ff9cec44dd8fbcd51755fd5ac5aec8890348271",
    "test": "b2d71a3b4a5c9c467823c671e8f76a4ced8d7be915afe3c164ed4511006773e6",
}


def content_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for c in df.columns:
        col = df[c]
        if c == "date":
            arr = pd.to_datetime(col).values.astype("datetime64[D]").astype(np.int64)
        else:
            arr = col.to_numpy(dtype=np.float64)
        h.update(c.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def test_generated_panel_matches_stored_digest():
    from boltzmann_fly.data import ICONIP_DATA, load_or_generate
    tr, va, te = load_or_generate(ICONIP_DATA, verbose=False)
    assert (len(tr), len(va), len(te)) == (312320, 30720, 30720)
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        assert content_hash(df) == EXPECTED[name], name
