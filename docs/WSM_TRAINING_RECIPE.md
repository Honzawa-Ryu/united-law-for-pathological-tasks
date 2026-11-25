# WSM v1 で独自データのトンネルを学習する手順

最終更新: 2026-11-22。出典は `libraries/WSMs/WSM_v1/` のコード（コミット c706edc）をサブエージェントが読んだ結果。行番号はそのコミットのもの。**実際に動かして確かめたのは「確認済み」と書いた項目だけ。**

結論: コードを変えずに「RGB → パレット画像」のトンネルを学習できる。必要なのは latent キャッシュ（shards）、custom 設定への追記、学習スクリプトの直接実行、推論前に手書きする `resolved_config.yaml`。

## 1. latent キャッシュ

- 置き場所: `<cache>/shards/<モダリティ名の - を _ にしたもの>/shard_XXXXX_of_YYYYY.pt`（`scripts/train_TunnelingNet_Flux2_4B_wsg.py:287-297`）。shard 数は自由。
- 学習側が読むキー: `latents`、`sample_ids`、`sample_sources`（`dataset_name`、任意で `split`）、任意で `video_ids` / `frame_indices`（`:411-450`）。
- `latents`: `[N, 32, 1, H/8, W/8]`、既定は bf16。`vae.encode(x).latent_dist.sample()` の生の出力で、スケールもパッチ化もしない（`manual_processing/flux2_preprocessing/vae_latents/precompute_wsg_flux_frame_latents.py:1177-1189`）。正規化と 2×2 パッチ化は学習側が行う（`train...py:2069-2078`）。
- 条件と目標の対応付けは `sample_ids` の共通部分（`:1214`）。別の索引ファイルは不要。
- 分割: `sample_sources[i]["split"]` が `train` / `val` / `test` ならそれを使う。無ければ `dataset_name:sample_id` のハッシュで 90 / 5 / 5 に割る（`:1433-1452`）。**スライド単位の漏れを防ぐため、必ず `split` を自分で書く。**
- 著者の変換スクリプトは名前付きキーの `.wsg.npz`（`{"0-2": HxWx3, "<目標名>": HxWx3}`、値 0–255）を入力にする。リサイズはせず、`patch_hw`（既定 480）に切り抜くか -1 で埋める（`src/dataset/wsg_dataset.py:338-384`）。
- **このリポジトリでは著者のスクリプトを使わず、実験 0004 で shards を直接書く**（split を確実に書くため。形式は上のとおり）。
- キャプション無しならテキスト埋め込みのキャッシュは不要。プロンプトは「condition modality 0-2. target modality X.」の固定文で、毎ステップその場で符号化される（`:1455-1470`、`:2056-2067`）。

## 2. 設定（`configs/wsg_v1_modalities_custom.json` に追記）

```json
"modality_specs": { "BRACS_Class_Palette": {"storage":"image_like","channels":3,"layout":"HWC",
   "value_range":[0.0,255.0],"encoding":"rgb_0_255_to_minus1_1","description":"BRACS class palette"} },
"normalization": { "modalities": { "BRACS_Class_Palette": "linear" } },
"wsg_v1": { "Channel_Head_000003": {
   "task_type":"estimation_or_reconstruction", "pair_exists":true, "target_density":"dense",
   "condition":"0-2", "condition_description":"0-2: H&E RGB patch",
   "target":"BRACS_Class_Palette", "target_description":"BRACS_Class_Palette: class palette" } }
```

- 既存のオブジェクトに足す（置き換えない）。custom の 000000〜000002 は使用済み。
- `task_type` は `estimation_or_reconstruction` / `conditional_control` / `inverse_rendering` / `conditional_control+narrow_generation`（`train...py:664-700`）。RGB → マスクは `estimation_or_reconstruction`（目標側だけ学習、条件側は凍結）。
- `libraries/WSMs` は git 管理外なので、追記した設定ファイルはリポジトリ側（実験ディレクトリ）に置き、`--modality-config` で渡す。

## 3. 分類枝

- `--categorical-loss-mode`（既定 off）: `prototype_ce` / `balanced_ce` / `margin_global` / `combined`（`:816-831`、`:3123-3149`）。**ラベルは渡さない。** 目標 latent をその場で k-means（`--categorical-num-clusters`、既定 8）して補助損失にするだけ。
- 推論時にクラスを割り当てる「refiner」は別の分類器（`src/pipelines/categorical_refiner.py`）。学習には `categorical_targets/<モダリティ>/shard_*.pt` が要り、著者のスクリプトは既定の 5 モダリティ（111 / 131 / 135 / 136 / 195）にしか書かない。独自の目標では自作が必要。
- 画像全体のラベルは「クラスの色で塗った画像」として表す（著者の 111 Object Category も同じ考え方、`precompute...py:201-221`）。**復号は自前で、最も近いパレット色の多数決にする。**

## 4. 学習ループ

- 既定: バッチ 1、勾配蓄積 1、学習率 1e-4、AdamW（wd 0.01）、スケジューラなし、勾配クリップ 0.5（`:837-841`、`:2114-2121`）。
- LoRA rank 8。**alpha は学習スクリプトの既定が 8、著者のランチャの既定が 274**（`launch_scripts/train_dense_structure_models.sh:238`）。著者に合わせるなら `--channel-head-alpha 274`。
- ステップ数: スクリプト既定 500、ランチャ 1000、custom の例は 2000〜3000。
- **チェックポイントは検証損失が改善したときだけ保存される**（`--eval-every-steps` ごとに評価、`:3778-3808`）。`final_step_N` は学習終了時のみ。`--save-every-steps` は実質使われていない。
  - 検証データが空だと検証が黙って無効になり、途中のチェックポイントが一切残らない（`:4196-4199`）。**4 時間で打ち切られると何も残らないので、`split: val` を必ず入れる。**
- 改善が `--early-stop-seconds`（既定 7200）無ければ早期終了。
- 再開: `--resume-from-checkpoint <output-dir>/state/step_XXXXXXX`（`:3874-3958`）。LoRA とステップ数は戻る。オプティマイザの状態は戻らない可能性が高い（推定）。
- wandb は不要。ログは標準出力と `stats.npz` / `stats.png`。
- 1 GPU なら `python` で直接実行できる。`PYTHONPATH=<WSM_v1>` が必要。
- 他の環境で壊れる箇所: ランチャ（`train_dense_structure_models.sh`）は著者の絶対パスと conda、tmux を前提にしている。**ランチャは使わない。**
- VRAM の目安は「24 GB 以上」というランチャの条件だけ。A6000 での実測は未確認。

## 5. コマンド

学習（`WSM_v1` をカレントにして実行）:

```bash
export PYTHONPATH=$PWD
WSG_READER_VISUALIZATION=0 python scripts/train_TunnelingNet_Flux2_4B_wsg.py \
  --model-path <FLUX.2-klein-4B のスナップショット> --modality-config <追記した custom json> \
  --dense-latent-cache-dir <cache> --channel-head-index 3 \
  --target-modality BRACS_Class_Palette --condition-modalities 0-2 \
  --channel-head-rank 8 --channel-head-alpha 274 --max-steps 2000 --learning-rate 1e-4 --num-workers 4 \
  --eval-every-steps 20 --viz-every-steps 500 --early-stop-seconds 7200 --preview-num-samples 3 \
  --categorical-loss-mode off --output-dir <run dir> [--resume-from-checkpoint <run dir>/state/step_XXXXXXX]
```

推論には `<run dir>/resolved_config.yaml` が要る（`scripts/infer_TunnelingNet_Flux2_4B_wsg.py:74-84`）。学習スクリプトは書かないので手で作る。キーは `base_model`、`modality_config`、`target_modality`、`condition_modalities: "0-2"`、`channel_head_rank: 8`、`channel_head_alpha: 274`、`resolved_condition_attention_mode: modality_tunneling`、`resolved_trainable_target: true`、`resolved_trainable_condition: false`、`resolved_single_step_x0: false`、`resolved_high_noise_anchor: true`、`resolved_multires_noise: false`、`seed`（`:120-158`）。

```bash
python scripts/infer_TunnelingNet_Flux2_4B_wsg.py --checkpoint <run dir>/ckpt/<step>/pytorch_lora_weights.safetensors \
  --input 0-2=patch.png --size 512 --steps 10 --categorical-refiner off --paper-demo --output out.png
```

キャッシュからまとめて推論する方法: 学習コマンドに `--inference-only --inference-checkpoint <ckpt> --preview-split test --preview-num-samples N` を足す（`train...py:4590-4705`）。

## 6. 未確認・注意

- A6000 での 1 ステップの時間と VRAM。
- 学習時の精度設定（Lightning の `precision` 指定が無く、LoRA の重みの dtype は未確認）。
- 名前付きの目標が `--paper-demo` 無しで正しく可視化されるか。
- プレビューは shard 0 だけを読む（`:4471-4477`）。各 shard に全 split を混ぜておく。

## 7. 特徴ベクトルを条件に足す（UNI などの埋め込み）

2026-11-22 追記。出典は同じコードの読解（サブエージェント）。学習が起動して train 3658 / val 312 を認識し、変換層が学習対象に入ることまでは確認済み（ジョブ 24677、24679）。精度は未確認。

- 学習コードには、サンプルごとの特徴ベクトル 1 本を条件にする経路がある（著者は音声の AST 特徴に使用。`WSGAudioConditionDenseDataset`、`train...py:1568` 付近）。次元は `--audio-condition-dim` で自由。
- 渡し方は 2 つ（`--audio-conditioning-mode`）: `attention` は学習可能な MLP で条件トークンにする（`--audio-global-condition` で 1 トークン。無いと目標の全トークンに複製、`:2088-2095`）。`modulation` は目標側への FiLM 変調で、ゲートは sigmoid(-4) から始まる（`:2009`）。
- **NPZ の形式**（`:1591-1630`）: `audio_feat` [N, D]（1 サンプル 1 本。系列は不可）、`wsv_path` と `frame_key`（どちらも unicode 文字列配列。object 配列は不可）。正規化は MLP 内の LayerNorm のみ。
- **突き合わせ**: latent shard の `sample_id` が `f"{wsv_path}::{frame_key}-{frame_key}"` と一致するものだけが使われる（`:1593`、`:1603`）。既存キャッシュの id は `::` を含まないので、id を `<元の id>::0-0` に付け替えたコピーを作る（実験 0008）。
- **RGB も条件にする**: `--condition-modalities Audio_AST,CurrentFrame_RGB_Condition`（`:4056-4072`、`:4162`）。`Audio_AST,0-2` は不可（視覚条件 2 つは inverse_rendering 以外でエラー、`:4091-4093`）。コードが読むのは `--condition-modalities` だけで、JSON の `condition` は説明用。
- **分割**: この経路は `sample_sources["split"]` を見ない。NPZ に入っている id がそのまま対象になる（`:1603-1611`）。train と val は別の NPZ（`--audio-dense-condition-train` / `--audio-dense-condition-val`）。**val の NPZ を省くと train が検証に使われる**（`:4176`）。
- **チェックポイント**: 変換層は `ckpt/<step>/audio_condition_projector.pt`（modulation は `audio_modulation_projector.pt` と `audio_modulation_gate.pt`）に保存され、`state/step_*` からの再開で復元される（`:2123-2140`、`:3866-3923`）。
- **不具合と修正**: `--inference-only` は変換層を読み込まない（`:4583-4598`）。そのまま評価すると初期値の変換層で推論され、結果が無効になる。`tools/patches/apply_wsm_projector_patch.py` で読み込み処理を足す（適用済み。`libraries/WSMs` を clone し直したら再実行）。
- **推論の対象**: `--audio-dense-condition-val` の NPZ で決まる。`--preview-split`、`--inference-shard-id` は効かない。test を回すには test の NPZ を train / val の両方に渡す（実験 0006 がそうしている）。
- プロンプトは「audio-conditioned dense structure prediction. target modality X. <説明>」に変わる（`:1638`）。
- このリポジトリでの使い方: `sbatch --export=ALL,CACHE=bracs_cls_10x_uni,MAX_STEPS=3000,UNI=attention experiments/0005_*/run_slurm.sh`（`UNI=modulation` も可）。評価は run 名に `__uni-` が入っていれば自動で特徴条件の設定になる。
