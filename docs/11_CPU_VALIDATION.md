# 11｜CPU 验证记录

2026-09-30，在助手的临时 Linux CPU 环境实际运行：

```bash
python -m unittest discover -s tests -v
python -m exp5 --help
python scripts/on_gpu.py --help
```

首次完整测试：**24 tests，12.936s，OK**；sklearn 交叉核对实际执行，没有跳过。没有下载真实数据或权重，没有运行 GPU 或收费 API。

| 验证项 | 覆盖 |
|---|---|
| 人工指标 | 正例第 1/7 名、五正例只命中一个、理想 NDCG |
| sklearn oracle | 随机单/五正例、确定性并列排序交叉核对 |
| 错误输入 | 重复 ID、缺正例、形状、NaN/Inf |
| 排序融合 | ID 字典序、RRF 端点、反向独立排名 |
| 数据 | 坏图、跨划分重复、ID/正例映射、sentids、路径越界 |
| 缓存 | FP32/范数/顺序、篡改检测、manifest 变更失败 |
| 分组统计 | 同图五查询整体采样、配对零差值、修复/退化 |
| 完整流程 | 合成 1000 val + 1000 test 图、每图五描述；审计→选择→确认→冻结→测试→复算 |
| 运行约束 | 小样本无法冻结、无 freeze 拒绝测试编码、模型和配置篡改失败、单卡显式选择 |

完整流程使用测试生成的小图和圆周特征，保存在 TemporaryDirectory 并自动清理，没有将合成指标写入 results。**这些不是 CLIP/SigLIP 2 成绩**。真实 processor 加载、特征接口、BF16/SDPA 和服务器 CUDA 兼容性仍未测试。

服务器剩余验收：真实数据审计、两模型 20 图/100 描述小试、实际锁文件/显存、全量验证/冻结、独立测试、人工错误分析与报告。Issue #1 仍未完成，PR 不自动合并。
