
スロットルの幅 RT = -1 - 1
steering -1 to 1


### Shared
* sharedがアップデートされたら、`git pull`して変更をラズパイ上にも反映すること

### SSH
* `ssh -A team40@team40.local`.　`-A`オプションでsshのフォワーディングを可能にすることによりラズパイからGITHUBにコミット可。
* vscodeのssh機能を使うとやりやすい

### rsync
* `rsync -rv [src] [dst]`, i.e. `rsync -rv data team40@team40.local:/home/team40/monkeycar/data`
