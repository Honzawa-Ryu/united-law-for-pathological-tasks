# 進捗記録

再開するときは、いちばん下の「次にやること」から読む。

## 2026-11-21

### やったこと
- 計画を作成（`docs/PLAN.md`）。Codex と Agy に相談（`docs/consult/`）。
- ライセンスと利用条件を確認（PLAN.md「着手前」）。
- クラスタに clone: `/workspace/filesrv02/honzawa/united-law-for-pathological-tasks`
  - `make setup` 済み。`env/env.sif` は `00-utils/sif/env.sif` へのシンボリックリンク。
  - WSM のコードを `libraries/WSMs` に clone（コミット c706edc、847 MB、`.gitignore` 済み）。サブモジュール `WSM_v1/env`、`wsg_reader` は SSH の URL で、取得していない。
- `pyproject.toml` に extra `wsm` を追加（WSMs の依存。`llm` extra とは同時に入れない設定）。
- 実験 `0001_20261121_wsm_infer_smoke` を作成（クラスタ側のみ。手元には未同期）。
  - 公開トンネル 2 つ（dense:000003 RGB→法線、dense:000038 RGB→意味セグメンテーション）× サイズ 256 / 512 を順に実行。
  - 読み込み時間、秒/枚、ピーク VRAM を `results.json` に記録。
  - HF キャッシュは `data/hf_cache`（`HF_HOME`）。

### 投入中のジョブ
| ジョブ | 内容 | 状態（16:15 時点） |
|---|---|---|
| 24258 | `make uv_sync p=small-creator extras=wsm` | 待機。開始見込み 23:10 |
| 24265 | 実験 0001（24258 の成功後に開始） | 待機 |

待機の理由: 自分の他ジョブ（tca-gates など）で Creator の通常枠 2 ノードが埋まっており、Creator の GPU も全台使用中。

### 分かったこと
- HF の `World-Snapshot/WSM_v1` は合計 115 GiB。内訳は `continued_training_materials` 86.6 GiB、`train_log`（全トンネル）27.8 GiB。`wsms` パッケージは指定したトンネルの分だけ取得する。ゲートなし。
- `WSMPipeline` の推論サイズの既定は **256 px**（512 ではない）。病理では解像度が効くので 256 / 512 の両方を測る。
- `libraries/` はテンプレートの `.gitignore` に入っていなかった。

### 未確認
- 追加した `wsm` extra で `uv sync` が解決するか（transformers>=5 と torch 2.11 の組み合わせ）。
- A6000 での VRAM と速度。
- BRACS の取得（利用登録待ち）。

### 19:40 時点の追記
- 24258（環境構築）成功。`wsm` extra は解決した。HF キャッシュ 23 GB（backbone + トンネル 2 つ）。
- 24265（実験 0001 の初回）失敗: `wsg_reader` サブモジュール未取得。`git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --depth 1` で取得し、24405 として再投入。
  - 失敗でも sacct は COMPLETED と出る（seq 実行で子の失敗が終了コードに反映されない）。成否はログと `results.json` で見る。テンプレート側は未修正。
  - creator6 に `/scratch/honzawa/0001_20261121_wsm_infer_smoke_24265` が残っている。
- BRACS は NAS にあった: `<NAS>:/share/main/image/BRACS/`（parent から `ssh -i <NAS 用の鍵> -o IdentitiesOnly=yes -p <port>`。鍵名が既定でないので `-i` が必須）。
  - `BRACS_WSI/`（984 GB、547 枚、train 395 / val 65 / test 87）、`BRACS_WSI_Annotations/`（.qpdata 387 件）、`BRACS.xlsx`、`processed/TRIDENT`（45 GB、20 倍 256 px）、`processed/CLAM_v2`（55 GB）。
  - 公式の ROI 画像セット（PNG）は NAS に無い。利用者の判断で、WSI と注釈から ROI を切り出す（方法 1）。
- `/workspace/filesrv02/honzawa/data/bracs/` に `BRACS.xlsx`、`BRACS_WSI_Annotations/`、`roi_annotations_geojson/` を置いた。
  - 変換は `tools/bracs/export_qpdata_geojson.{sh,groovy}`（QuPath 0.5.1、ジョブ 24404、19 秒）。
  - 結果: 387 ファイル、ROI 4540 個（多角形 4539、MultiPolygon 1）。分割別は train 3658 / val 312 / test 570 で、公式（3657 / 312 / 570）と train が 1 つ違うだけ。
  - クラス名は表記ゆれあり: `Benign-sure`/`Benign sure`/`BENIGN`→N、`Pathological-benign`/`Pathologica benign`→PB、`UDH(-sure)`、`FEA(-sure)`、`ADH(-sure)`、`DCIS(-sure)`、`Malignant(-sure)`/`MALIGNANT`→IC。`Pathological-benign (Benign-sure)` が 1 個あり、扱いは未定（train の余分 1 個の候補）。
  - ROI の大きさ（40 倍の px）: 幅の中央値 1628、5〜95% は 650〜4240、最大 17611。

### 20:10 時点の追記
- 実験 0001 成功（24405、24409。A6000、bfloat16、10 ステップ）。
  | トンネル | サイズ | 秒/枚 | ピーク VRAM |
  |---|---|---|---|
  | dense:000003 RGB→法線 | 256 / 480 / 512 | 4.7 / （results.json 参照）/ 7.8 | 14.9 GiB |
  | dense:000038 RGB→意味セグメンテーション | 256 / 480 / 512 | 4.4 / （同）/ 7.4 | 14.9 GiB |
  - 初回読み込み 81 秒、2 回目以降 2〜9 秒。秒/枚には前後処理と可視化の書き出しを含む（拡散の計算だけの時間ではない）。
  - variant 名に `__demo` が付かない 4 件は入力が説明図（cover.png）で、質の判断には使えない。`__demo` は論文の図の入力画像（サイズ 480 が論文の設定）。
  - 法線: 人物の輪郭や机の面を捉えた出力。ただし `result.png` はカラーマップ表示で、通常の RGB 法線マップの見た目ではない。論文の図と同じ表示かは未確認。
  - 意味セグメンテーション: ライオンの領域をきれいに切り出した（`generated_pseudo_rgb_unrefined.png`）。分類枝は領域 1 個にクラス 23、確信度 0.97。**`result.png`（選択ビュー）は真っ黒になるので、このトンネルでは `generated_pseudo_rgb_*.png` を見る。**
  - トンネル付属の検証値: 精度 0.796、バランス精度 0.710（`categorical_refiner_result.json`）。
- 24406 の結果: 計算ノード（creator6）から NAS へは通信できるが、鍵 `<NAS 用の鍵>` が計算ノードに無い（ホームが共有されていない）。venv に openslide / shapely / openpyxl が無い。
- `pyproject.toml` に extra `patho`（openslide-python、openslide-bin、shapely、openpyxl）を追加。`make uv_sync p=small-creator extras="wsm patho"`（24410）を投入。**今後は必ず両方の extra を付ける**（片方だけだともう一方が消える）。

### 21:30 時点の追記
- 24410（`wsm patho` の uv sync）成功。
- WSI の転送は parent で rsync（利用者了承）。`data/bracs/lists/fetch_wsi.sh <split>` を `nohup` で実行、ログは `data/bracs/logs/fetch_<split>.log`（末尾に DONE が出れば完了）。再実行で続きから。
  - 注釈付き WSI: train 281 枚 531.8 GiB、val 37 枚 68.6 GiB、test 69 枚 118.9 GiB（一覧は `data/bracs/lists/`）。
  - val は転送済み（`data/bracs/wsi_staging/`、69 GB）。test は 21:27 に転送開始。
- 実験 `0002_20261121_bracs_roi_extract` を作成。投入は `sbatch --export=ALL,SPLIT=<split> experiments/0002_20261121_bracs_roi_extract/run_slurm.sh`（GPU なし、8 CPU）。
  - 注釈の外接矩形をレベル 0（40 倍）で切り出し、`data/bracs/roi_png/{split}/{CLASS}/{slide}_{CLASS}_{k}.png` に保存。一覧は `roi_png/manifest_{split}.csv`。
  - スライドごとに `roi_png/_done/{split}/{slide}.json` を書き、書けたら WSI を消す（`delete_wsi_after: true`）。再投入で続きから。未取得や失敗が残る間は completed にならない。
  - val を 24453 として投入（待機中）。

### 22:25 時点の追記
- val の ROI 切り出し成功（24453、2 分 31 秒）: 37 枚すべて done、ROI 312 個（公式と同数）。クラス別は FEA 49、IC 47、UDH 46、N 46、PB 43、ADH 41、DCIS 40。`roi_png/val` は 2.8 GB。mpp は 0.2519（292 個）と 0.2524（20 個）。
  - 目視（`val/DCIS/BRACS_1368_DCIS_4.png`、`val/N/BRACS_1886_N_1.png`）: ラベルと矛盾しない。切り出しは外接矩形で、多角形の外側も含む。
  - val の WSI は切り出し後に消えた（設定どおり）。
- test の転送完了（22:05）。切り出しを 24464 として投入。
- train は 3 分割（`lists/annotated_wsi_train_part00〜02`、94 / 94 / 93 枚）。part00 の転送を 22:23 に開始（`logs/fetch_train_part00.log`）。
  - 切り出しは毎回 `SPLIT=train` で投入する。未転送の WSI は `wsi_missing` になり exit 1 で終わるが、済んだ分は `_done` に残る。3 つとも済んだ回で completed になる。

## 2026-11-22

### 00:30 時点
- test の ROI 切り出し成功（24464）: 69 枚、ROI 570 個（公式と同数）。DCIS 85、FEA 83、UDH 82、N 81、IC 81、ADH 79、PB 79。
- train part00 の最初の転送は、`split` で作ったリストに `.txt` が付いておらず即失敗していた（35 分の空振り）。リネームして 22:59 に再開、23:55 完了。
- train の切り出し 1 回目（24487）: 94 枚 done、ROI 836 個。残り 187 枚は `wsi_missing`（想定どおり exit 1）。part01 の転送を 00:22 に開始。
- **PNG の iCCP 問題**: openslide の画像にスキャナの ICC プロファイル（約 120 KB）が付き、そのまま保存した PNG は PIL が `Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK` で読めない。
  - 0002 の保存処理で ICC を落とすよう修正（train は最初から ICC なし）。
  - val / test の 882 枚は `tools/bracs/strip_png_icc.py` で iCCP チャンクだけ除去（24485。画素データは不変）。
- **実験 0003（VAE 往復）成功（24486）。Phase 0 の関門は通過。**
  | 倍率 | タイル数 | PSNR | SSIM | H 成分の相関 | 核領域 Dice（画像の往復） | マスクの往復 Dice | 核の個数比（中央値） |
  |---|---|---|---|---|---|---|---|
  | 40 倍 | 291 | 33.7 | 0.947 | 0.984 | 0.939 | 0.9993 | 1.008 |
  | 20 倍 | 215 | 29.9 | 0.931 | 0.969 | 0.909 | 0.9959 | 1.016 |
  | 10 倍 | 88 | 27.4 | 0.916 | 0.946 | 0.877 | 0.9911 | 1.014 |
  - 値は平均。5 パーセンタイルでも 40 倍の核領域 Dice は 0.906、マスク Dice は 0.9986。
  - 目視（`outputs/0003_*/40x/examples/`）: 元画像と再構成の見分けはほぼつかない。細かい斑点を含む二値マスクも保たれる。
  - 限界: 核マスクは正解注釈ではなく H 成分の大津二値化（細かいノイズを含み、1 タイルあたりの連結成分が 800 以上と多い）。UNI 埋め込みでの比較は未実施。VAE の往復は「情報が潜在に残るか」の診断で、トンネルが学習できるかは別。
  - タイル数が倍率で違うのは、ROI が切り出し寸法（20 倍で 1024 px、10 倍で 2048 px）より小さいものを除くため。

### 08:15 時点
- 手元の `.claude/settings.local.json` に `ssh parent_h *` / `scp * parent_h:*` / `rsync * parent_h:*` の事前許可を追加（利用者の依頼）。コマンド置換 `$(...)` を含む呼び出しはサンドボックス内に残るので、その場合はサンドボックス外を明示する。
- train part01 の転送は 01:19 に完了していた。切り出し 2 回目（24538）: 188 枚 done、ROI 1895 個。part02（93 枚）の転送を 08:10 に開始。
- parent で `nohup ... &` を ssh 越しに起動すると ssh が戻らないことがある（転送自体は続く）。手元側の待機が打ち切られても問題ない。

### 09:20 時点
- **BRACS の ROI 切り出し完了。** train part02 の転送は 09:05 完了、切り出し 3 回目（24552）で train 281 枚すべて done。
  | 分割 | WSI | ROI | 公式 | 容量 |
  |---|---|---|---|---|
  | train | 281 | 3658（PB 715、DCIS 665、FEA 624、IC 521、UDH 389、ADH 386、N 358） | 3657 | 31 GB |
  | val | 37 | 312 | 312 | 2.8 GB |
  | test | 69 | 570 | 570 | 5.0 GB |
  - `data/bracs/wsi_staging/` は空（WSI は削除済み、原本は NAS）。
  - toxpatho-kb の `data/storage_and_features.md` に場所と注意点を追記（手元のみ、未コミット）。
- 状況確認は `ssh parent_h bash <repo>/tools/bracs/status.sh`。
- WSM の学習コードの読解をサブエージェントに依頼中（結果待ち）。

### 11:45 時点
- WSM の学習手順を `docs/WSM_TRAINING_RECIPE.md` にまとめた（コード読解。1 回目のサブエージェントは ssh の承認待ちで、2 回目は利用上限で失敗。コードを手元の一時ディレクトリにコピーし、3 回目で完了）。
- 実験 `0004_20261122_bracs_wsm_latent_cache`（24570、21 分）: BRACS 全 ROI を latent キャッシュに変換。`data/wsm_cache/bracs_cls_10x/`（2.3 GB、`samples.csv`、`palette.json` 付き）。
  - 入力: ROI 中央の 2048 px 四方（40 倍）を 512 px に縮小（10 倍相当）。小さい ROI は白で埋める。目標: クラス色の単色画像。
  - **視野より小さい ROI が 72%、組織の割合は平均 0.50。** 10 倍は余白が多い。20 倍（1024 px 四方）のキャッシュを 24586 で作成中。
  - 投入は `sbatch --export=ALL,MAG=20 experiments/0004_*/run_slurm.sh`。
- 実験 `0005_20261122_bracs_wsm_train_cls` の試運転成功（24571、200 ステップ、バッチ 1）:
  - train 3658 / val 312 を認識。学習対象 12.5M パラメータ。
  - **約 3.1 秒/ステップ、検証（24 バッチ）は 1 回約 70 秒、ピーク VRAM 18.1 GB（nvidia-smi）。** 全体 16 分。
  - 学習損失 1.11 → 0.6 前後、検証損失 0.192（50）→ 0.175（100）→ 0.154（150）→ 0.157（200）。
  - チェックポイントは `data/wsm_runs/0005_*/<variant>/ckpt/<step>/`、再開用は `state/step_<step>/`。再投入すると最新の state から自動で再開する。
  - 投入は `sbatch --partition=large-creator --time=03:55:00 --export=ALL,CACHE=bracs_cls_10x,MAX_STEPS=3000 experiments/0005_*/run_slurm.sh`。検証間隔は 250 に変更済み。
- 実験 `0006_20261122_bracs_wsm_eval_cls` を作成: 学習済みトンネルで val / test を推論（`--inference-only`、生成 latent を保存）→ VAE で復号 → 画素ごとに最も近いパレット色 → 多数決。accuracy、macro-F1、weighted-F1、混同行列を出す。
  - 投入は `sbatch --export=ALL,RUN=<0005 の variant>,SPLIT=val experiments/0006_*/run_slurm.sh`。

### 投入中（11:45 時点。自分の他ジョブ 0009 が Creator を使用中で、開始見込みは 12:18）
| ジョブ | 内容 |
|---|---|
| 24585 | 0005 本番: 10 倍、3000 ステップ（large-creator、3:55） |
| 24586 | 0004: 20 倍のキャッシュ作成 |
| 24587 | 0005 本番: 20 倍、3000 ステップ（24586 の後） |
| 24588 | 0006: 200 ステップの試運転モデルで val を評価（評価の流れの動作確認。精度は期待しない） |

### 12:40 時点
- 利用者から「ノードが空いているのでジョブを多く投げてよい」との指示。学習を 4 条件に増やした（各 3000 ステップ、large-creator 3:55）。
  | ジョブ | variant | 12:24 時点 |
  |---|---|---|
  | 24585 | `bracs_cls_10x__steps3000` | 850 ステップ、検証損失 0.149 → 0.149 → 0.144 |
  | 24587 | `bracs_cls_20x__steps3000` | 550 ステップ、検証損失 0.153 → 0.154 |
  | 24598 | `bracs_cls_20x__steps3000__bs4`（`BATCH=4`） | 300 ステップ。約 5.2 秒/ステップで、3:55 では 3000 に届かない見込み（再投入で再開） |
  | 24599 | `bracs_cls_20x__steps3000__cat-combined`（`CATEGORICAL=combined`、k=7） | 実行中 |
  - 24585 と 24599 は `x-large-creator-extra`（横取りされうる枠）で走っている。PREEMPTED になったら同じコマンドで再投入。
  - 学習後の val 評価を依存ジョブで予約済み: 24600（10x）、24601（20x）、24602（bs4）、24603（cat）。
- 評価の動作確認（24588、試運転 200 ステップのモデル、val 312 枚）: accuracy 0.25、macro-F1 0.14。生成画像はほぼ単色（最多色の割合 0.976）。予測は FEA 134、IC 118、ADH 48 に偏る。IC は 47 枚中 40 枚正解。推論 4.66 秒/枚。評価の流れは正常。
- **比較対象（実験 `0007_20261122_bracs_uni_roi_probe`、24606）**: NAS の UNI2-h パッチ埋め込み（TRIDENT、20 倍 256 px）を `data/bracs/trident_uni_v2/` に転送（387 枚、32 GB、`tools/bracs/fetch_uni_features.sh`）。ROI 内のパッチを平均 → L2 正規化 → 標準化 → ロジスティック回帰（C は val の macro-F1 で選択）。
  | 使う範囲 | パッチ数/ROI（中央値） | val acc / macro-F1 | test acc / macro-F1 / weighted-F1 |
  |---|---|---|---|
  | ROI 中央 1024 px（WSM の 20 倍入力と同じ範囲） | 4 | 0.692 / 0.673 | 0.590 / 0.573 / 0.575 |
  | ROI 中央 2048 px（WSM の 10 倍入力と同じ範囲） | 9 | 0.657 / 0.647 | 0.611 / 0.600 / 0.602 |
  | ROI 全体 | 9 | 0.647 / 0.635 | 0.609 / 0.600 / 0.602 |
  - test のクラス別 F1（ROI 全体）: N 0.71、PB 0.47、UDH 0.45、FEA 0.71、ADH 0.32、DCIS 0.62、IC 0.92。
  - 注意: C はどの条件も格子の下端 0.01 が選ばれた（もっと小さい値が良い可能性）。パッチ中心が 1 つも入らない ROI が 29 個あり、最も近いパッチで代用。パッチは WSI の格子上にあり ROI に位置合わせされていないので、ROI の外の組織も混ざる。
  - h5py は `.venv` に無く、学習中に uv sync したくないため `data/pydeps` に別途入れて読んでいる（`pyproject.toml` の patho extra には追加済み）。

### 16:00 時点
- WSM の val 評価（312 枚、色の多数決）。比較対象の UNI2-h + 線形は val macro-F1 0.63〜0.67。
  | run | 重み | シード | acc | macro-F1 | 備考 |
  |---|---|---|---|---|---|
  | `bracs_cls_10x__steps3000` | 3000 | 既定 | 0.458 | 0.406 | IC 0.90、DCIS 0.70、N 0.0 |
  | 同上 | 3000 | 1 | 0.458 | 0.404 | |
  | 同上 | 3000 | 2 | 0.458 | 0.411 | |
  | `bracs_cls_20x__steps3000` | 2750（検証損失最良） | 既定 | 0.186 | 0.166 | ほぼ偶然 |
  | 同上 | final 3000 | 既定 | 0.353 | 0.317 | |
  | `bracs_cls_20x__steps3000__cat-combined` | 3000 | 既定 | 0.359 | 0.323 | |
  - **推論はシードに対して安定**（10 倍で 3 シードとも acc 0.458）。複数回生成の多数決は不要。
  - **学習の途中経過は不安定**: 20 倍は 250 ステップ違うだけで acc 0.19 ↔ 0.35。検証損失（0.141 前後で横ばい）は分類精度の目安にならない。
  - 3000 ステップ時点では 10 倍 > 20 倍。補助損失（combined）の効果は見えない（20 倍の final と同程度）。
  - どの条件も出力はほぼ単色（最多色の割合 0.99）。N はほとんど予測されない。
- 対応: `early_stop_seconds` を 100000 に変更（検証損失の頭打ちで止まらないように）。10 倍を 9000 ステップまで延長中（24662、`tools/wsm_extend_run.sh bracs_cls_10x 3000 9000`、`x-large-creator-extra` で実行中）。
- 0006 に `CKPT`（best / final / ディレクトリ名）と `INFER_SEED` を追加。
- バッチ 4（24598）は 2500 ステップ付近で 4 時間枠の打ち切り。続きを 24673、final の val 評価を 24674 で予約。
- 利用者から「UNI をエンコーダーにできないか」との質問。VAE の置き換えは不可だが、学習コードの音声条件の経路（`WSGAudioConditionDenseDataset`、`--audio-condition-dim`、`--audio-conditioning-mode attention|modulation`、`--audio-global-condition`）が、サンプルごとの特徴ベクトルを学習可能な MLP で条件トークン（または FiLM 変調）にする仕組みで、UNI の ROI 特徴（1536 次元）を流せそう。条件: NPZ（`audio_feat`、`wsv_path`、`frame_key`）、sample_id は `{wsv_path}::{frame_key}-{frame_key}` の形、条件モダリティは `Audio_AST`。詳細はサブエージェントが調査中。

### 16:35 時点
- UNI 条件付き学習の手順を確定（`docs/WSM_TRAINING_RECIPE.md` の 7 節）。
- `tools/patches/apply_wsm_projector_patch.py` を作成し、クラスタの `libraries/WSMs` に適用（`--inference-only` で変換層を読み込む修正）。
- 実験 `0008_20261122_bracs_uni_condition_inputs`（24676）: `data/wsm_cache/bracs_cls_10x_uni/` を作成。latent shards は id を `<元の id>::0-0` に付け替えたコピー、`uni_{train,val,test}.npz` は WSM の入力と同じ範囲（ROI 中央 2048 px）のパッチ平均（L2 正規化、1536 次元）。パッチ中心が入らない ROI は 29 個（最も近いパッチで代用）。
- 0005 に `UNI`（none / attention / modulation）、0006 に UNI 条件付き run の評価を追加。
- 学習を投入: 24677（`bracs_cls_10x_uni__steps3000__uni-attention`、変換層 216K、学習対象 12.7M）、24679（`...__uni-modulation`、学習対象 36.1M）。どちらも train 3658 / val 312 を認識して開始。val 評価（final）は 24678、24680 で予約。
- 実行中: 24662（10 倍、9000 まで延長）、24673（20 倍バッチ 4 の続き）→ 24674（その val 評価）。

### 17:40 時点
- val 評価の追加結果（312 枚）:
  | run | 重み | acc | macro-F1 | 備考 |
  |---|---|---|---|---|
  | `bracs_cls_20x__steps3000__bs4` | final 3000 | **0.510** | **0.465** | これまでの最良。IC 0.92、DCIS 0.76、FEA 0.58、PB 0.39、UDH 0.39、ADH 0.21、N 0.0 |
  | `bracs_cls_10x__steps9000` | 5000（途中） | 0.439 | 0.384 | 3000 の 0.458 から伸びていない。N 0.29、UDH 0.0 |
  - **バッチ 1 のままステップを増やしても伸びない。バッチ 4 は同じ 20 倍・3000 ステップのバッチ 1（0.353）より大きく良い。** 学習の安定にはバッチサイズが効く。
  - バッチ 4 のピーク VRAM は 43.8 GB（nvidia-smi）、3000 ステップで計 4.6 時間。バッチ 8 は A6000 では載らない見込み（投入した 24702 は取り消した）。勾配蓄積（`--gradient-accumulation-steps`）は 0005 に未実装。
- 追加投入: 24701（`bracs_cls_20x__steps9000__bs4`、3000 から延長）→ 24705（val、final）。24703（`bracs_cls_10x__steps3000__bs4`）→ 24704（val、final）。
- 実行中: 24662（10 倍バッチ 1、9000 まで。5000 時点で伸びなしだが、結論を確かめるため継続）、24677 / 24679（UNI 条件付き）→ 24678 / 24680。

### 19:30 時点
- 24662（10 倍バッチ 1 の 9000 への延長）は 17:39 に PREEMPTED（`x-large-creator-extra` で実行中に、自分の通常枠ジョブが入った）。`state/step_0005000` まで保存済み。5000 時点で伸びが無かったので再投入しない。
- **UNI 条件付き（10 倍、バッチ 1、3000 ステップ）の val 評価**（24678、24680。変換層の読み込みはログの `[united-law patch] loaded` で確認）:
  | run | acc | macro-F1 | クラス別 F1 |
  |---|---|---|---|
  | `bracs_cls_10x_uni__steps3000__uni-attention` | 0.529 | 0.505 | N 0.44、PB 0.46、UDH 0.32、FEA 0.48、ADH 0.21、DCIS 0.83、IC 0.80 |
  | `bracs_cls_10x_uni__steps3000__uni-modulation` | 0.522 | 0.500 | N 0.45、PB 0.38、UDH 0.43、FEA 0.40、ADH 0.21、DCIS 0.72、IC 0.92 |
  | （参考）UNI なし 10 倍バッチ 1 | 0.458 | 0.406 | N 0.0 |
  | （参考）UNI なし 20 倍バッチ 4 | 0.510 | 0.465 | N 0.0 |
  | （参考）UNI2-h + 線形（同じ範囲の特徴） | 0.657 | 0.647 | |
  - UNI を足すと macro-F1 が 0.41 → 0.50（同じ 10 倍・バッチ 1 比）。N が予測されるようになった。ただし、同じ UNI 特徴の線形分類（0.65）には届かない。
  - attention と modulation はほぼ同じ。modulation は検証損失の上下が大きかった。
- 追加投入: UNI（attention）× バッチ 4。24712（`bracs_cls_10x_uni__steps3000__bs4__uni-attention`）→ 24713（val）。24714（0008 で 20 倍の入力作成）→ 24715（`bracs_cls_20x_uni__steps3000__bs4__uni-attention`）→ 24716（val）。
- 実行中: 24701（20 倍バッチ 4、9000 まで）→ 24705、24703（10 倍バッチ 4）→ 24704。

### 22:30 時点
- **`bracs_cls_10x__steps3000__bs4` の val（24704）: acc 0.577、macro-F1 0.535。UNI なしで現時点の最良。** クラス別 F1: N 0.65、PB 0.53、UDH 0.31、FEA 0.54、ADH 0.21、DCIS 0.66、IC 0.84。バッチ 4 同士では 10 倍（0.577）> 20 倍（0.510）。
- 24701（20 倍バッチ 4 の 9000 への延長）は 21:27 に **OUT_OF_MEMORY**（ホストメモリ 64 GB の上限。約 3 時間 50 分、5600 ステップ付近）。`state/step_0005500` まで保存済み。長時間の学習ではホストメモリが増え続ける可能性がある。
  - `SBATCH_MEM_PER_NODE=110G` を付けて再投入（24723、5500 から再開）。24705（依存が満たされない評価）は取り消し、5500 時点の val 評価を 24725 で投入。
- 10 倍バッチ 4 を 9000 まで延長（24724、110 GB、`tools/wsm_extend_run.sh bracs_cls_10x 3000 9000 __bs4 BATCH=4`）。評価は未予約（途中の重みを手動で評価する）。
- 実行中: 24712 / 24715（UNI × バッチ 4、2000 / 3000 付近、メモリ 64 GB のまま）→ 24713 / 24716。

## 2026-11-23

### 00:25 時点
- val 評価（312 枚、バッチ 4）:
  | run | 重み | acc | macro-F1 | クラス別 F1 |
  |---|---|---|---|---|
  | `bracs_cls_20x__steps9000__bs4` | 5500（途中、24725） | 0.574 | 0.534 | N 0.30、PB 0.47、UDH 0.44、FEA 0.66、ADH 0.24、DCIS 0.71、IC 0.93 |
  | `bracs_cls_10x_uni__steps3000__bs4__uni-attention` | final（24713） | **0.638** | **0.630** | N 0.66、PB 0.53、UDH 0.39、FEA 0.64、ADH 0.37、DCIS 0.87、IC 0.95 |
  | `bracs_cls_20x_uni__steps3000__bs4__uni-attention` | final（24716） | 0.625 | 0.614 | N 0.52、PB 0.53、UDH 0.43、FEA 0.64、ADH 0.39、DCIS 0.84、IC 0.94 |
  | （参考）UNI2-h + 線形、範囲 2048 / 1024 | | 0.657 / 0.692 | 0.647 / 0.673 | |
  - バッチ 4 なら延長が効く（20 倍: 3000 で 0.510 → 5500 で 0.574）。
  - UNI × バッチ 4 は、UNI なしバッチ 4（10 倍 0.577 / 0.535）から macro-F1 で約 0.10 上がり、UNI の線形分類に近い水準。ADH が 0.21 → 0.37 に上がった。
- test 評価（570 枚）を投入: 24734（10 倍 + UNI）、24735（20 倍 + UNI）、24736（10 倍バッチ 4、UNI なし）。比較対象の UNI + 線形の test は macro-F1 0.573〜0.600。
- 実行中: 24723（20 倍バッチ 4、6860 / 9000）、24724（10 倍バッチ 4、4360 / 9000）。どちらもメモリ 110 GB。

### 01:20 時点
- **test 評価（570 枚、公式分割。条件は val で選択済み）**:
  | モデル | acc | macro-F1 | weighted-F1 | クラス別 F1（N / PB / UDH / FEA / ADH / DCIS / IC） |
  |---|---|---|---|---|
  | WSM 10 倍、バッチ 4、3000（24736） | 0.537 | 0.499 | 0.501 | 0.71 / 0.39 / 0.25 / 0.63 / 0.14 / 0.57 / 0.81 |
  | WSM 10 倍 + UNI、バッチ 4、3000（24734） | 0.547 | 0.525 | 0.526 | 0.68 / 0.40 / 0.24 / 0.60 / 0.33 / 0.57 / 0.86 |
  | WSM 20 倍 + UNI、バッチ 4、3000（24735） | 0.570 | 0.557 | 0.558 | 0.67 / 0.49 / 0.28 / 0.59 / 0.36 / 0.60 / 0.91 |
  | UNI2-h + 線形、範囲 1024（20 倍の入力と同じ） | 0.590 | 0.573 | 0.575 | 0.73 / 0.45 / 0.39 / 0.67 / 0.21 / 0.65 / 0.91 |
  | UNI2-h + 線形、範囲 2048（10 倍の入力と同じ） | 0.611 | 0.600 | 0.602 | 0.72 / 0.49 / 0.43 / 0.70 / 0.30 / 0.62 / 0.93 |
  - test でも順位は同じ: UNI 線形 > WSM + UNI > WSM 単体。WSM + UNI と UNI 線形の差は macro-F1 で 0.02（20 倍）〜0.08（10 倍）。
  - val では 10 倍 + UNI が 20 倍 + UNI より上だったが、test では逆。val は 312 枚で、この程度の差は誤差の範囲とみるべき。
  - どのモデルも val より test が 0.04〜0.10 低い（UNI 線形も同じ傾向）。
  - 推論コスト: WSM は 1 枚約 4.7 秒（10 ステップ）。UNI 線形は特徴があれば即時。
- 延長 2 本の final の val 評価を依存ジョブで予約。

### 03:05 時点
- 24724（10 倍バッチ 4 の延長、`x-large-creator-extra` で実行）が 02:20 頃に PREEMPTED。直前に自分が通常枠で投入した BCSS 取得ジョブ（24756）が同じノードに入ったためとみられる。`state/step_0005750` まで保存済み。24803 として再投入（110 GB）、val 評価は 24804。
  - **教訓: 横取りされうる枠で長い学習が走っている間に、通常枠の小さなジョブを足すと自分の学習を落とす。** 追加する前に `squeue` でどのジョブが `-extra` にいるかを見る。
- BCSS 取得の 1 回目（24756）は即失敗: `ssh parent_h sbatch --chdir=...` だと `SLURM_SUBMIT_DIR` がホーム（計算ノードに無い）になる。`cd <repo> && sbatch` で投入し直した（24802、空き待ち）。スクリプトは `tools/bcss/fetch_bcss.sh`。
- 実行中: 24723（20 倍バッチ 4、8680 / 9000）→ 24738（val、final）。

### 04:05 時点
- **`bracs_cls_20x__steps9000__bs4`（UNI なし）の final の val（24738）: acc 0.651、macro-F1 0.636。** クラス別 F1: N 0.71、PB 0.52、UDH 0.52、FEA 0.59、ADH 0.34、DCIS 0.85、IC 0.93。
  - 同じ run の推移: 3000 で 0.510 / 0.465 → 5500 で 0.574 / 0.534 → 9000 で 0.651 / 0.636。バッチ 4 なら延長で伸び続けている。
  - val では UNI2-h + 線形（0.657〜0.692 / 0.647〜0.673）、WSM + UNI の 3000 ステップ（0.638 / 0.630）と同水準。
  - 学習は計 5 時間 4 分（5500 → 9000 の再開分）+ それ以前。9000 ステップ全体でおよそ 13 時間。
- test 評価を投入: 24870（`bracs_cls_20x__steps9000__bs4`、final）。
- 公平な比較のため、20 倍 + UNI も 9000 まで延長: 24872（110 GB）→ 24873（val、final）。
- 10 倍バッチ 4 の延長（24803）は 6400 / 9000。→ 24804（val、final）。
- BCSS 取得の 2 回目（24802）も失敗: この版の gdown に `--remaining-ok` が無い。引数を外して 24871 で再投入（空き待ち）。

### 05:05 時点
- **test（570 枚）: `bracs_cls_20x__steps9000__bs4`（UNI なし、24870）acc 0.561、macro-F1 0.545、weighted-F1 0.546。** クラス別 F1: N 0.67、PB 0.43、UDH 0.26、FEA 0.65、ADH 0.39、DCIS 0.59、IC 0.83。
  - test のまとめ（macro-F1）: UNI 線形 0.573（範囲 1024）/ 0.600（範囲 2048）、WSM 20 倍 + UNI（3000）0.557、**WSM 20 倍（9000、UNI なし）0.545**、WSM 10 倍 + UNI（3000）0.525、WSM 10 倍（3000）0.499。
  - val では UNI 線形に並んだが（0.636 対 0.647〜0.673）、test では 0.03〜0.05 下。val → test の落ち幅は WSM（0.09）も UNI 線形（0.05〜0.10）も同程度。
- BCSS 取得（24871、14 分）: `/workspace/filesrv02/honzawa/data/bcss/0_Public-data-Amgad2019_0.25MPP/` に `rgbs_colorNormalized/` 151 枚、`masks/` 150 枚、`meta/`（`gtruth_codes.tsv`、`roiBounds_BaseMagnification.csv` など）。計 4.6 GB。**マスク 1 枚（`TCGA-C8-A12V-DX1_xmin11056_ymin54401_MPP-0.2500.png`）が Google Drive の制限で未取得。** 後で取り直す。toxpatho-kb への追記は未。
- 実行中: 24803（10 倍バッチ 4、7100 / 9000）→ 24804、24872（20 倍 + UNI、3000 → 9000）→ 24873。

### 07:00 時点（BCSS の組織セグメンテーションの準備）
- 実験 `0009_20261123_bcss_wsm_latent_cache`（24886、空き待ち）: BCSS の ROI を 20 倍相当（0.5 µm/px）に縮小 → 512 px タイル（重なりなし、ラベル 50% 以上）→ latent キャッシュ `data/wsm_cache/bcss_seg_20x/`。正解のクラス番号画像は `gt/` に保存。
  - 5 クラス（tumor / stroma / inflammatory / necrosis / other）+ ignore（0、評価から除く）。色は `palette.json`。
  - 分割は施設単位。test は OL、LL、E2、EW、GM、S3（**記憶に基づく指定。公式ベースラインのページで未確認**）、残りのスライドの 15% を val。
  - 取れていなかったマスク 1 枚は、`meta/roiBounds_BaseMagnification.csv` の figshare のリンクからジョブ内で取得を試みる。
- 実験 `0010_20261123_bcss_wsm_train_seg`（0005 の写し、対象モダリティ `BCSS_Tissue_Palette`）: 24888（バッチ 4、6000 ステップ、110 GB、24886 の後）。
- 実験 `0011_20261123_bcss_wsm_eval_seg`（推論は 0006 の写し、評価は画素単位: 最も近いクラス色 → pixel accuracy、mIoU、Dice、混同行列）: val の final を依存ジョブで予約。
- 比較対象（UNI + 軽量デコーダ）は UNI2-h の重みが必要。クラスタ上の場所を利用者に確認中。無ければ DINOv2。
- 実行中: 24803（10 倍バッチ 4、8330 / 9000）→ 24804、24872（20 倍 + UNI、4250 / 9000）→ 24873。

### 08:00 時点
- 0009 の 1 回目（24886）は失敗: figshare から取り直したマスク（`TCGA-C8-A12V-DX1_...`）が 7622×4738 で、画像（6313×3924、0.25 MPP）と大きさが違った。縦横比は同じなので、画像の大きさに最近傍で合わせる処理を足して再実行（24898、成功）。
  - **`afterok` の依存は当てにならない**: テンプレートの seq 実行は失敗しても COMPLETED になるため、24888（学習）が壊れたキャッシュで起動して即失敗した。依存ジョブを組むときは、先頭の結果を確かめてから。
- **`data/wsm_cache/bcss_seg_20x/`（808 MB）**: ROI 151 枚（150 スライド）→ タイル 1591 枚。train 1025（91 スライド）/ val 116（15）/ test 450（44）。
  - 画素の割合: tumor 0.378、stroma 0.342、inflammatory 0.121、necrosis 0.061、other 0.052、ignore 0.047。
- 学習 24899（`bcss_seg_20x__steps6000__bs4`、110 GB）→ 評価 24900（val、final）。
- BRACS: 24803（10 倍バッチ 4、9000 まで）完了、val 評価 24804 実行中。24872（20 倍 + UNI）は 4940 / 9000。

### 09:55 時点
- BRACS `bracs_cls_10x__steps9000__bs4`（UNI なし）: val（24804）acc 0.628、macro-F1 0.598（N 0.70、PB 0.50、UDH 0.51、FEA 0.62、ADH 0.17、DCIS 0.76、IC 0.93）。**test（24914）acc 0.574、macro-F1 0.557、weighted-F1 0.557**（N 0.71、PB 0.47、UDH 0.39、FEA 0.60、ADH 0.29、DCIS 0.52、IC 0.90）。
  - BRACS test のまとめ（macro-F1）: UNI 線形 0.600（範囲 2048）/ 0.573（範囲 1024）、WSM + UNI 20 倍 3000 ステップ 0.557、**WSM 10 倍 9000 ステップ 0.557**、WSM 20 倍 9000 ステップ 0.545、WSM + UNI 10 倍 3000 ステップ 0.525、WSM 10 倍 3000 ステップ 0.499。
  - UNI なし・9000 ステップでは、val は 20 倍が上（0.636 対 0.598）、test は 10 倍が上（0.557 対 0.545）。倍率の差は誤差の範囲。
- BCSS の学習（24899、`x-large-creator-extra`）: train 1025 枚を認識。1250 ステップ時点で検証損失 0.160 → 0.150。
- 実行中: 24872（BRACS 20 倍 + UNI、6370 / 9000）→ 24873。

### 11:05 時点
- **BCSS セグメンテーションの途中評価（24927）: `bcss_seg_20x__steps6000__bs4` の 1750 ステップ時点、val 116 タイル。**
  - pixel accuracy 0.797、mIoU 0.568、平均 Dice 0.712。推論 4.4 秒/タイル。
  - クラス別 IoU: tumor 0.765、stroma 0.675、necrosis 0.574、inflammatory 0.485、other 0.339。
  - ラベルのある画素を ignore（黒）と予測した割合は 0.003。
  - 目視（`outputs/0011_*/.../examples/`、左から 正解 / 予測 / 生成画像そのまま）: 腫瘍の広い領域は合う。細い帯状の領域でクラスを取り違える例あり（正解 other を necrosis と予測）。生成画像には中間色（橙など）が出ることがあり、最も近い色に丸めている。
  - 評価の処理は正常に動作。比較対象（同じ分割・同じタイルでの UNI / DINOv2 + 軽量デコーダ）は未作成なので、この数字の良し悪しはまだ判断できない。
- 実行中: 24899（BCSS 学習、2000 / 6000 を通過）→ 24900（val、final）、24872（BRACS 20 倍 + UNI、7000 / 9000）→ 24873。

### 12:10 時点
- 実験 `0012_20261123_bcss_frozen_encoder_linear_seg`: BCSS の比較対象。凍結 ViT のパッチ特徴（入力 518 px、37×37 トークン）+ 1×1 畳み込み、40 エポック。タイル・分割・正解は 0009 のキャッシュと同じ。投入は `sbatch --export=ALL,ENCODER=dinov2_vitl14 experiments/0012_*/run_slurm.sh`。
  - **DINOv2 ViT-L/14（24936、4 分）**: val pixel accuracy 0.783、mIoU 0.569。**test（450 タイル）pixel accuracy 0.795、mIoU 0.580、平均 Dice 0.727。** test のクラス別 IoU: tumor 0.749、stroma 0.676、inflammatory 0.541、necrosis 0.534、other 0.402。特徴抽出 0.066 秒/タイル。
    - 最良エポックが最終の 40 なので、学習を延ばせばもう少し上がる可能性がある。線形ヘッドのみで、デコーダの工夫は無し。
  - **UNI2-h（24937）は取得できず**: `GatedRepoError 401`。計算ノードの環境に Hugging Face のトークンが無い（`HF_HOME` は `data/hf_cache`）。利用者にトークンか重みの場所を確認中。
- 参考: WSM（`bcss_seg_20x__steps6000__bs4`）は 1750 ステップ時点で val pixel accuracy 0.797、mIoU 0.568。DINOv2 + 線形の val（0.783 / 0.569）と同水準。

### 13:10 時点
- 24872（BRACS 20 倍 + UNI の 9000 への延長）が 8 時間の制限で TIMEOUT（8500 ステップ付近）。**再開用の state は検証損失が改善したときにしか保存されないため、残っていたのは `step_0006750`。** 1750 ステップ分をやり直す形で 24954 として再投入（creator4 指定、6 時間、110 GB）。val 評価は `afterany` で予約。
  - 教訓: 延長するときは制限時間に余裕を持たせる（バッチ 4 は約 5.3 秒/ステップ + 検証。6000 ステップで 9〜10 時間）。`tools/wsm_extend_run.sh` の `--time=08:00:00` は短い。
- BCSS の学習（24899）は 3500 / 6000、検証損失 0.147。`--time=12:00:00` で、現在 5 時間 11 分。

### 17:05 時点
- **BCSS: `bcss_seg_20x__steps6000__bs4` の final の val（24900、116 タイル）: pixel accuracy 0.794、mIoU 0.561、平均 Dice 0.704。** クラス別 IoU: tumor 0.763、stroma 0.678、inflammatory 0.514、necrosis 0.558、other 0.291。
  - 1750 ステップ時点（0.797 / 0.568）から伸びていない。DINOv2 ViT-L + 線形の val（0.783 / 0.569）と同水準で頭打ち。学習は 8 時間 49 分。
- **BRACS: `bracs_cls_20x_uni__steps9000__bs4__uni-attention` の final の val（24955、312 枚）: acc 0.699、macro-F1 0.670。** クラス別 F1: N 0.78、PB 0.60、UDH 0.48、FEA 0.76、ADH 0.23、DCIS 0.87、IC 0.98。
  - val では、UNI2-h + 線形（範囲 1024: 0.692 / 0.673）と同じ、WSM 20 倍 UNI なし 9000 ステップ（0.651 / 0.636）より上。
- test 評価を投入: BCSS の final、BRACS 20 倍 + UNI の final。

### 18:00 時点
- **test の結果がそろった。全体のまとめは `docs/RESULTS.md`。**
  - BCSS（25060、450 タイル）: WSM pixel accuracy 0.808、mIoU 0.611、平均 Dice 0.751（DINOv2 + 線形は 0.795 / 0.580 / 0.727）。推論 4.26 秒/タイル。
  - BRACS 20 倍 + UNI 9000 ステップ（25061、570 枚）: acc 0.588、macro-F1 0.570（UNI + 線形の範囲 1024 は 0.590 / 0.573）。
- 実行中・予約中のジョブは無し。ループを停止。

## 2026-11-25

### 11:45 時点
- 論文を全文で読み直した（`docs/RELATED_WORK.md` に追記）。著者は分類を「生成で領域の形を決め、別の小さな分類器でクラスを決める」構成で扱っている（3.6 節、表 11）。病理は課題の分類表に載っているが実験は無い。
- コードを読んで分かったこと: 分類器（`MaskedLatentClassifier`、`src/pipelines/categorical_refiner.py`）は**入力画像の VAE latent だけ**を見る。生成結果は領域の形にしか使わない。BRACS は画像全体が 1 クラスなので、この構成ではトンネルも生成も結果に関与しない。
- 著者の `categorical_record_split` はキャッシュの `split` を見ずにハッシュで分割し直す。公式分割が混ざるので、`tools/patches/apply_wsm_refiner_split_patch.py` で明示的な分割を優先させた（クラスタに適用済み）。
- 実験 0013（`bracs_wsm_categorical_refiner`）を作成。sidecar（`<cache>/categorical_targets/BRACS_Class_Palette/`）を作り、著者のスクリプトを既定値（3000 ステップ、50 ステップごとに val、balanced accuracy 最良の重み）で学習し、val / test を評価する。10 倍・20 倍 × 3 シード。ジョブ 25402（medium-creator、投入時はノードが埋まっていて待ち）。

### 12:15 時点
- 25402 は creator4 で 18 分で完了（大戸さんの `-extra` 枠のジョブを Preempt して開始。利用者の了承あり、2 ノードまで可）。
- **0013 の結果（3 シード平均、正解率 / macro-F1）: 10 倍 val 0.523 / 0.498、test 0.426 / 0.400。20 倍 val 0.450 / 0.422、test 0.415 / 0.382。** 色の多数決方式（test macro-F1 0.545〜0.570）と UNI + 線形（0.573〜0.600）より大きく下。
  - 学習損失は下限まで下がり val の損失は 1.4〜1.5。過適合。最良ステップは 100〜2300 とばらつく。
  - 各 split の件数は 3658 / 312 / 570 で公式分割どおり（実験コード内で検証）。
- `docs/RESULTS.md` と 0013 の `report.yaml` に反映。

### 次にやること
利用者の判断待ち:
1. UNI2-h の重み（または Hugging Face のトークン）の場所。BCSS で UNI + 軽量デコーダと比べるのに必要（0012 を `ENCODER=uni2h` で再実行するだけ）。
2. コミットと push。クラスタの `experiments/`、`tools/`、`docs/`、`pyproject.toml` などは未コミット。toxpatho-kb の追記も未コミット。
3. 次の方向: (a) BCSS で少数ショット比較、(b) WSI 全体への拡張の設計、(c) ここで区切る。

判断なしで進められること:
- BCSS の分割（test の施設）を公式ベースラインのページで確認する。
- `docs/RELATED_WORK.md` の一次確認。
- DINOv2 + 線形デコーダのエポック数を増やして、比較対象を強くする。
- テンプレートの「seq 実行で子が失敗しても COMPLETED になる」問題の修正。
