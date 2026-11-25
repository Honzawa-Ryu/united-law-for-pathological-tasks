"""libraries/WSMs の学習スクリプトに、--inference-only で特徴条件の変換層を読み込む修正を当てる。

背景: `scripts/train_TunnelingNet_Flux2_4B_wsg.py` の --inference-only は LoRA だけを読み込み、
`audio_condition_projector.pt` / `audio_modulation_projector.pt` / `audio_modulation_gate.pt` を読まない
（コミット c706edc の 4583-4598 行付近）。そのままだと、特徴ベクトルを条件にしたトンネルの評価が
初期値の変換層で行われ、結果が無効になる。

libraries/WSMs は git 管理外なので、clone し直したらこのスクリプトをもう一度実行する（何度実行しても同じ結果）。
    python3 tools/patches/apply_wsm_projector_patch.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "libraries" / "WSMs" / "WSM_v1" / "scripts" / "train_TunnelingNet_Flux2_4B_wsg.py"
MARKER = "# [united-law patch] load feature-condition projector in --inference-only"
ANCHOR = '''        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        resolved_refiner = resolve_categorical_refiner(
'''
PATCH = f'''        {MARKER}
        for _fn, _attr in (
            ("audio_condition_projector.pt", "audio_condition_projector"),
            ("audio_modulation_projector.pt", "audio_modulation_projector"),
        ):
            _module = getattr(model, _attr, None)
            if _module is not None:
                _path = checkpoint.parent / _fn
                if not _path.exists():
                    raise FileNotFoundError(_path)
                _module.load_state_dict(torch.load(_path, map_location="cpu"))
                print(f"[united-law patch] loaded {{_path}}", flush=True)
        if getattr(model, "audio_modulation_gate", None) is not None:
            _gate = checkpoint.parent / "audio_modulation_gate.pt"
            model.audio_modulation_gate.data.copy_(
                torch.load(_gate, map_location=model.audio_modulation_gate.device)
            )
'''

source = TARGET.read_text()
if MARKER in source:
    print("already patched")
else:
    assert source.count(ANCHOR) == 1, f"anchor found {source.count(ANCHOR)} times; the upstream code changed"
    TARGET.write_text(source.replace(ANCHOR, PATCH + ANCHOR))
    print("patched", TARGET)
