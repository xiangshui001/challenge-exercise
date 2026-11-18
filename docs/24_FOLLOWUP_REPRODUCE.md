# 三项冻结实验的复算与复现

从仓库根目录运行，保留原 M0 分支中的结果目录。逐项复算只需 Python 和 NumPy。主结果目录中的图像以指纹表示，完整输入仅存放在受控环境。

```bash
python -m followup.summarize results/followup-20261119
python -m unittest discover -s tests -p 'test_m0*.py' -q
```

## 重建输入

自然图像沿用 M0 的官方来源和固定版本。[来源说明](21_M0_DATA_AND_ASSETS.md)。CLEVR 使用 Stanford 官方发布的 [CLEVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/)；验证场景 JSON 的 SHA256 为 f53e27a9463d00534a857aac37601b7166749f811c566ce0dca8d32bf77e0585。程序按 HTTP Range 读取 ZIP 目录和选中成员并核对 ZIP CRC，无需下载完整归档。请在私有运行工作区中操作，将仓库根目录加入 PYTHONPATH，并提供 results/m0-20261102 原证据目录。

```bash
python -m followup.prepare --scpp /data/scpp/data --esnli /data/e-ViL/data/esnlive_test.csv --old results/m0-20261102/manifest.jsonl --out natural.jsonl
python -m followup.acquire scenes scenes.json
python -m followup.prepare --scenes scenes.json --out spatial.jsonl
python -m followup.acquire clevr images --manifest spatial.jsonl
python -m followup.acquire coco images --manifest natural.jsonl
# images/flickr 指向按原文件名组织的 Flickr30K 原图目录
python -m followup.audit
```

原复核队列及 13 个 AI 选义输入可用 M0 的私有清单、原图和完整候选知识重建。此处的 knowledge.private.json 是原 M0 run_joint 生成的按样本 ID 索引的 WordNet 候选表。

```bash
python -m followup.reference --old /data/m0/manifest.private.jsonl --images /data/m0/images --knowledge /data/m0/knowledge.private.json --evidence /repo/results/followup-20261119 --out /private/rebuilt
```

## 运行

模型文件身份沿用 [M0 复现说明](22_M0_REPRODUCE.md)。程序会核验权重 SHA256。实跑环境见各 run.json 的 packages。请为每次运行提供全新 out 路径以及当前被授权且空闲的 GPU 索引。每个模型和清单组合先运行 --limit 2 的接口小试，再去掉该参数正式运行。小试记录单列。

```bash
python -m followup.run --model siglip --manifest natural.jsonl --model-dir models/siglip --images images --gpu 0 --out runs/siglip-natural-full
python -m followup.run --model clip --manifest natural.jsonl --model-dir models/clip --images images --gpu 0 --out runs/clip-natural-full
python -m followup.run --model qwen --manifest natural.jsonl --model-dir models/qwen --images images --gpu 0 --out runs/qwen-natural-full
python -m followup.run --model qwen --manifest spatial.jsonl --model-dir models/qwen --images images --gpu 0 --out runs/qwen-spatial-full
python -m followup.run --model qwen --manifest /private/rebuilt/reference.jsonl --model-dir models/qwen --images images --gpu 0 --out runs/qwen-reference-full
```

## 复核与配对

人工填写时打开私有 review/review.html，可逐项看图、选择标签或候选义项、填写可见证据、导出及重新导入。文件夹须随图片一起保存。向评审只提供该文件夹，避免在提交前阅读公开模型答案。AI 选义条件只传入选中的词义和这些义项的既有上位关系；AI 的图像判断、理由及人工字段都没有进入提示。该条件沿用旧队列的已知类别和难例选择，所以只用于探索性对照。

```bash
python -m followup.review --human human-review.json --reference results/followup-20261119/ai-reference.jsonl
python -m followup.verify_inputs /private/manifests results/followup-20261119
```

verify_inputs 要求 natural.jsonl、spatial.jsonl 和 reference.jsonl 位于同一私有目录。公开审核可用 summarize 检查完整覆盖、标签、输入图像身份、模型散列及臂间配对，并从逐项预测重新计算结果。

