
# monkeycar car-app

ラズパイ上でカメラ・ジョイスティック・モーター制御を行う走行アプリ。`uv` で実行します。

## 要件
- Python >= 3.11
- uv
- ラズパイ環境（PiCamera2 / ServoKit など）

## セットアップ
```bash
make install
```

## 実行
```bash
make run
```

### オプション実行
```bash
make autopilot   # 推論モードで起動
make record      # 記録モードで起動
```

## 操作方法（ジョイスティック）
- X: オートパイロット ON/OFF
- B: 記録 ON/OFF
- Y: 終了（例外でループ終了）
- 左スティック X 軸: ステアリング
- RT: スロットル

## データ/モデル
- 記録データ: `./data/session_<timestamp>/`
	- 画像: `image/*.jpg`
	- ラベル: `records.csv`
- 推論モデル: `./output/`
	- `config.json`
	- `params.pkl`

## Make ターゲット
- `make run`: 通常起動
- `make autopilot`: オートパイロットで起動（`--autopilot`）
- `make record`: 記録モードで起動（`--record`）
- `make calibrate`: ESC キャリブレーション
- `make lint`: ruff + mypy（通常）
- `make lint-strict`: mypy strict
- `make clean`: キャッシュ削除
- `make clean-data`: `./data` と `./output` を削除
- `make fclean`: `clean` + `.venv` 削除

## 共有/運用メモ
### Shared
- shared がアップデートされたら `git pull` してラズパイ上にも反映する

### SSH
- `ssh -A team40@team40.local`（`-A` でフォワーディング）
- VS Code の SSH 機能を使うと便利

### rsync
- `rsync -rv [src] [dst]`
- 例: `rsync -rv data team40@team40.local:/home/team40/monkeycar/data`
