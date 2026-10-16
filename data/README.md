# data：数据处理代码与本地数据

统一项目根目录下的数据入口；服务器与 GitHub 使用相同相对路径。

- 当前可用的数据审计实现：`exp5/data.py` 与 `python -m exp5 prepare`，处理 Karpathy val/test，每图保留五条描述。
- 下一步在本目录实现训练集 Dataset/DataLoader，读取原始 `train` 划分，按图像采样训练描述；不使用 val/test 做梯度更新。
- 原始图片、划分 JSON 与生成 manifest 放 `data/local/`，已被 Git 忽略。
- 不复制第二份指标或审计逻辑；新训练数据入口复用已有校验，并显式补充训练划分支持。

当前训练数据读取器未实现。本目录不是数据集的公开下载地址。训练数量由实际固定划分审计决定，不沿用聊天中的约数。
