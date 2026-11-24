# 离线盲评表

| 表单 | 取样 | 填写内容 |
| --- | --- | --- |
| `initial` | 历史 500 张，四层各抽 20 张 | 80 张的逐句证据与句间关系 |
| `audit` | 150 张确认图中，原图 Qwen 对基准困难句答 Yes 的全部 42 条 | 126 句逐句核查，可优先查看 18 张对象互换 |
| `pair` | What'sUp 实拍 5 对，10 张图 | 每对两句、四格判断，以及每张图的可见依据 |

使用 Python 3.11+。以下命令从**仓库根目录**执行，生成的页面在本地浏览器中可放大图片、保存进度、导出和恢复标注。首轮只想用一张表时，可将 `all` 改为表中的英文名称。`--fetch` 表示允许从公开数据源取得缺少的输入；复用有权访问的本地数据时可去掉该参数。

```console
python -m pip install Pillow==12.3.0
python results/m0-grounding-20261122/review/build_forms.py all --cache data/local/review-cache --out data/local/review-forms --fetch
```

缓存目录内，`annotations/<编辑类型>.json` 对应固定版本 SugarCrepe++ 注释，`coco/<12 位图号>.jpg` 对应原始 COCO 图片，`whatsup/<行号>.jpg` 对应登记的 What’sUp 图片响应字节或已校验的缩小预览。重建逐句校验文本 SHA256，逐图校验来源或展示字节，最后校验整张 HTML 的 SHA256。上游内容变化将导致校验失败。表单身份与最初交付给评阅者的离线版本相同。照片、基准原句及个人回传原件均保存在本地。[SugarCrepe++ 固定版本](https://github.com/Sri-Harsha/scpp/tree/c083bbf039a5ca215acaf25e7ee736d2ea6f3864) · [What’sUp 作者入口](https://github.com/amitakamath/whatsup_vlms) · [实拍副本固定版本](https://huggingface.co/datasets/joaomonteirof/whatsup_subset_a_inference/tree/3c1b68d412c6b3cd9de2bef13f0420942bc1ce5e)

**给评阅者。** 打开生成目录中的 HTML。能指出画中哪一处推翻陈述时选“冲突”；关键对象在画外、模糊、被挡住，或单帧无法辨别动作方向时选“证据不足”。只把离线表发给评阅者即可。

**评完再解盲。** 研究者可以按题号和显示列 A/B/C 连接回传和审计主表。主表包含基准身份、句子散列、模型 Yes−No 分差、两条参考句是否均严格领先，以及助手初判和依据。两项统计可以离线用标准库复核。

```console
python results/m0-grounding-20261122/review/audit_counts.py
```

[42 张解盲主表（含分数和判断）](unblinding-42.csv) · [取样索引（含呈现顺序）](sources.json) · [首轮 8 张人工试标匿名汇总](initial-80-summary.json) · [研究报告](../../../docs/m0-grounding/README.md)
