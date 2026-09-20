"""Route each name to plain or IPA F1 and compare policies on one paired set.

Policies, from the hard-subset screen:

  plain                    always the plain clip
  always IPA               always the IPA clip
  era=new                  IPA only when strata era is new
  new+generic              IPA only when era is new and the row is generic
  new OR biologic suffix   IPA when era is new or the name has an FDA
                           four-letter biologic suffix

Mean is SpeechBERTScore F1 (best human clip). Items missing either score
are dropped from every row so the deltas share one denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_scores(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("f1_scoreable") and row.get("best_f1") is not None:
            out[row["drug"].lower()] = float(row["best_f1"])
    return out


def _strata() -> dict[str, dict]:
    out = {}
    for line in (ROOT / "dose_r" / "strata" / "strata.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["name"].lower()] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--ipa", required=True)
    args = ap.parse_args()
    plain = _load_scores(Path(args.plain))
    ipa = _load_scores(Path(args.ipa))
    strata = _strata()
    keys = sorted(set(plain) & set(ipa) & set(strata))

    def use_ipa(row: dict, policy: str) -> bool:
        new = row["era"] == "new"
        generic = row["name_type"] == "generic"
        bio = bool(row["biologic_suffix"])
        if policy == "plain":
            return False
        if policy == "always IPA":
            return True
        if policy == "era=new":
            return new
        if policy == "new+generic":
            return new and generic
        if policy == "new OR biologic suffix":
            return new or bio
        raise ValueError(policy)

    policies = (
        "new OR biologic suffix",
        "era=new",
        "new+generic",
        "plain",
        "always IPA",
    )
    base = sum(plain[k] for k in keys) / len(keys)
    print(f"paired n={len(keys)} plain_mean={base:.3f}")
    fired = {p: 0 for p in policies}
    for name in keys:
        for policy in policies:
            if use_ipa(strata[name], policy):
                fired[policy] += 1
    print("policy\tmean_f1\tvs_plain\tn_ipa")
    for policy in policies:
        vals = []
        for name in keys:
            row = strata[name]
            vals.append(ipa[name] if use_ipa(row, policy) else plain[name])
        mean = sum(vals) / len(vals)
        delta = mean - base
        sign = f"{delta:+.3f}" if policy != "plain" else "—"
        print(f"{policy}\t{mean:.3f}\t{sign}\t{fired[policy]}")


if __name__ == "__main__":
    main()
