"""Shared synthetic-catalog helpers for tests. No real catalog content."""

import json

VOCAB = {
    "AC": "account access privilege authorize login permission",
    "AU": "audit log record event timestamp review",
    "SC": "encrypt network boundary transmission cryptographic channel",
}


def rows_for(prefix_map, n_bases=12, per_base=2, pad=False):
    rows = []
    for fam, words in VOCAB.items():
        w = words.split()
        for b in range(1, n_bases + 1):
            for k in range(per_base):
                base = f"{prefix_map[fam]}-{b:02d}" if pad else f"{prefix_map[fam]}-{b}"
                rows.append(
                    {
                        "id": base if k == 0 else f"{base}({k})",
                        "base_id": base,
                        "family": prefix_map[fam],
                        "family_title": f"{prefix_map[fam]} title",
                        "title": f"title {base}",
                        "text": " ".join(w[(b + k + i) % len(w)] for i in range(4)) + f" item{b}{k}",
                    }
                )
    return rows


def write(dirpath, rows):
    dirpath.mkdir(parents=True, exist_ok=True)
    test = [r for r in rows if int(r["base_id"].split("-")[1]) > 9]
    train = [r for r in rows if int(r["base_id"].split("-")[1]) <= 9]
    for name, rs in (("train.jsonl", train), ("test.jsonl", test)):
        (dirpath / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
