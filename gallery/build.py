from pathlib import Path
from html import escape as e
import json,sys
root=Path(sys.argv[1])
cfg=json.load(open(sys.argv[2]))
github='https://github.com/siddhartha-yz/embodied-weight-growth'
def a(link):
    name,target=link
    if not target.startswith('https://'):
        if target.endswith('.md'):
            assert (root/target).is_file(),target
            target=f'{github}/blob/main/{target}'
        else:
            assert (root/target).is_file(),target
    return f'<a href="{e(target)}">{e(name)}</a>'
def media(d):
    img=d['poster']
    assert (root/img).is_file(), img
    if 'video' in d:
        v=d['video']; assert (root/v).is_file(),v
        return f'<video controls playsinline preload="none" poster="{e(img)}" aria-label="{e(d["title"])}录像"><source src="{e(v)}" type="video/mp4"><a href="{e(v)}">下载录像</a></video>'
    return f'<img class="preview figure" loading="lazy" src="{e(img)}" alt="{e(d["title"])}实验图">'
def caption(d,level):
    return f'<div class="caption"><p class="kicker">{e(d["label"])}</p><h{level}>{e(d["title"])}</h{level}><p>{e(d["copy"])}</p>{" ".join(a(l) for l in d["links"])}</div>'
feature=cfg['feature']
groups='\n'.join(f'<section class="section" id="{e(g["id"])}"><h2>{e(g["title"])}</h2><p>{e(g["copy"])}</p><div class="grid">'+'\n'.join(f'<article class="card">{media(d)}{caption(d,3)}</article>' for d in g['cards'])+'</div></section>' for g in cfg['groups'])
(root/'index.html').write_text(f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="具身神经生长实验：从机械臂写 W 到 H1 受推恢复和 LIBERO 抓放。看真实回放、测量结果和对照。">
  <title>具身神经生长｜实验回放</title>
  <link rel="stylesheet" href="gallery/style.css">
</head>
<body>
  <header><nav class="shell nav" aria-label="项目导航"><strong>Embodied Weight Growth</strong><a href="{github}">GitHub 仓库 ↗</a></nav></header>
  <main class="shell">
    <section class="intro"><p class="kicker">EXPERIMENT GALLERY</p><h1>具身神经生长</h1><p>同样的交互预算，把试错写进权重，还是写成代码？</p></section>
    <article class="feature" id="libero">{media(feature)}{caption(feature,2)}</article>
    {groups}
  </main>
  <footer class="shell"><p>仿真实录与按记录生成的回放在各实验页分别标注。<a href="{github}/blob/main/RESEARCH.md">研究问题与对照设计</a></p></footer>
</body></html>
""",encoding='utf-8')
