# 更新回放图库

把实录和海报放进对应的实验目录；在 `experiments.json` 加一项。从仓库根目录运行：

```sh
python3 gallery/build.py . gallery/experiments.json
python3 -m http.server 8000
```

打开本机的 8000 端口检查播放器和排版。提交数据、媒体和生成的 `index.html`。GitHub Pages 从 `main` 根目录发布；视频只存一份。
