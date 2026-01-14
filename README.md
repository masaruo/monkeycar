

### 環境設定
* pythonの環境構築に`uv`
* プロジェクトの実行の`make`
* pc-appについては`Apple Silicon Mac`
* car-appについては`debian bookwarm`で開発

### SSH
* `ssh -A team40@team40.local`.　`-A`オプションでsshのフォワーディングを可能にすることによりラズパイからGITHUBにコミット可。
* vscodeのssh機能を使うとやりやすい

### rsync
* `rsync -rv [src] [dst]`, i.e. `rsync -rv data team40@team40.local:/home/team40/monkeycar/data`

