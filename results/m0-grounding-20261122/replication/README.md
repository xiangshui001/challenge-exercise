# 视觉证据与措辞偏好

## 用封存分数复算

解压到可写目录，在包根目录执行。Python 3.11 或更新版本和 NumPy 即可完成以下步骤。确认统计用了 NumPy 的分层重抽样；不同库版本或平台的小数末位可能有差异。

```console
python confirmation/confirm_stats.py
python confirmation/absolute_check.py controls confirmation
python controls/analyze.py
python deeper/diagnose_offsets.py
python deeper/audit_grounding.py
```

第一条命令复算 150 张确认图的预先指定主比较、三张随机错图扣分、近远错图对照与逐图审计表。输出在 confirmation/confirm_summary.json 和 confirmation/confirm_audit.csv。confirm_stats.py 会校验匿名清单与冻结清单的绑定、输入和分数文件指纹、复现用代码指纹、各图对应关系及完整性。第二条命令是追加的探索性阈值诊断，使用旧批 200 图选绝对打分阈值，在新批 150 图上固定应用，并写入 confirmation/absolute_summary.json。

第三条命令复算 200 张旧图的每个句对在原图、三次随机错图、纯文本条件下的错误。后两条命令使用同一旧批次，复算 Qwen 无图分数诊断、打分差分及错图后排序重现时的回答倾向。每张图有两个预期正确句和一个困难负句；数据文件按这个固定顺序保存三项分数。

## 词面相似度与跨模型重合

deeper/cross_model.json 对应旧批次，confirmation/replication_patterns.json 对应新批次。两个脚本同时计算词面与跨模型统计，其中词面部分需要原始英文。程序可从作者仓库的固定版本重建两份研究清单。在包根目录指定一个尚不存在、位于解压目录外的临时目录。脚本只读取固定 Git 对象校验通过的上游注释，再按逐句话的 SHA256、图片身份和冻结清单的 SHA256 核对。

```console
python rebuild_texts.py --out ../lexical-rerun
python ../lexical-rerun/deeper/cross_model.py
python ../lexical-rerun/confirmation/replicate_patterns.py
```

已有上游五份原始 JSON 时可在第一条命令后加上 `--annotations-dir 路径`，从而离线重建。该目录应直接包含五个以编辑类型命名的 .json 文件。确认出包时的词面复核结果保存在上述两份 JSON 中。confirmation/caption_overlap.json 记录新图与此前共 820 组记录的逐句散列去重结果，完整重建该历史审计还需要更早两轮的来源记录；check_overlap.py 记录了原审计操作。

## 包内布局和来源

- confirmation/ 含新批冻结协议、抽样与错配代码、匿名清单与绑定、图像间相似度、三模型逐项分数及运行元数据、主结果表和追加分析。manifest.phash.json 仅保存还原冻结清单所需的图像感知散列。
- controls/ 含旧批冻结协议、准备与推理代码、匿名清单、三模型分数及运行元数据、按句对的初始分析。
- deeper/ 含对旧批封存分数的追加诊断脚本和结果。各脚本使用相邻的 controls/ 数据。
- SHA256SUMS.txt 记录包内其他文件未压缩内容的 SHA256。数据使用公开的样本编号、图片编号和散列识别。GPU 设备标识只随运行记录保留为 SHA256。

GPU 推理脚本保留实验执行时的代码及服务器目录约定，用于审核实现；从原始图片重跑还需要原始数据、私有图像缓存、固定模型权重、此前运行代码，以及 PyTorch、Transformers、Pillow、SciPy 等环境。CPU 数值复算直接使用本包的封存预测。

数据与前序工作见 [SugarCrepe++ 固定注释版本](https://github.com/Sri-Harsha/scpp/tree/c083bbf039a5ca215acaf25e7ee736d2ea6f3864) 和 [challenge-exercise 前序研究](https://github.com/xiangshui001/challenge-exercise/pull/9)。
