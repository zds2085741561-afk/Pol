from pathlib import Path
import pandas as pd

base = Path("data/NYC_STEPS")
files = [
    "NYC_STEPS.train.inter",
    "NYC_STEPS.valid.inter",
    "NYC_STEPS.test.inter",
]

max_len = 50
col = "item_id_list:token_seq"

for f in files:
    path = base / f
    df = pd.read_csv(path, sep="\t")

    df[col] = df[col].astype(str).apply(
        lambda s: " ".join(s.split()[-max_len:])
    )

    df.to_csv(path, sep="\t", index=False)

    lens = df[col].astype(str).apply(lambda s: len(s.split()))
    print(f"{f}: rows={len(df)}, max_seq_len={lens.max()}, min_seq_len={lens.min()}")
