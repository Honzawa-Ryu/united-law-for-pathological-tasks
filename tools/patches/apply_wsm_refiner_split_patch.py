"""libraries/WSMs の分類枝（categorical refiner）に、キャッシュの明示的な分割を尊重させる修正を当てる。

背景: `src/pipelines/categorical_refiner.py` の `categorical_record_split` は `sample_sources["split"]` を見ず、
AVIS 以外のデータを `"<dataset>:<sample_id>"` のハッシュで 5% val / 5% test / 残り train に割る
（コミット c706edc の 520-538 行付近）。学習スクリプト側の `_record_split` は明示的な分割を使うので食い違い、
BRACS では公式の train / val / test が混ざって漏れる。

libraries/WSMs は git 管理外なので、clone し直したらこのスクリプトをもう一度実行する（何度実行しても同じ結果）。
    python3 tools/patches/apply_wsm_refiner_split_patch.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "libraries" / "WSMs" / "WSM_v1" / "src" / "pipelines" / "categorical_refiner.py"
MARKER = "# [united-law patch] honour explicit split in sample_sources"

source = TARGET.read_text()
if MARKER in source:
    print("already patched")
else:
    m = re.search(r"^def categorical_record_split\((?P<args>[^)]*)\)[^\n]*:\n", source, flags=re.M)
    assert m, "categorical_record_split not found; the upstream code changed"
    args = [a.split(":")[0].split("=")[0].strip() for a in m.group("args").split(",") if a.strip()]
    src_arg = next((a for a in args if a in ("source", "sample_source")), None)
    assert src_arg, f"no source argument in {args}"
    body_start = m.end()
    # 関数の先頭（docstring があればその後）に入れる
    doc = re.match(r'[ \t]*("""|\'\'\')[\s\S]*?\1[ \t]*\n', source[body_start:])
    insert_at = body_start + (doc.end() if doc else 0)
    patch = (
        f"    {MARKER}\n"
        f"    _explicit = str(({src_arg} or {{}}).get(\"split\") or \"\").strip().lower()\n"
        f"    if _explicit in {{\"train\", \"val\", \"test\"}}:\n"
        f"        return _explicit\n"
    )
    TARGET.write_text(source[:insert_at] + patch + source[insert_at:])
    print("patched", TARGET, "source arg:", src_arg)
