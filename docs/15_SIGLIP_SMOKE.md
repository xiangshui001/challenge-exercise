# 15｜SigLIP-base 短小试

P0 已通过用户服务器日志确认。用户于 2026-10-01 通过 HF-Mirror 完成 SigLIP-base 下载，revision 为 `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`。此入口使用真实权重，默认创建一张合成红色方块和两段英文描述，仅检查加载、原生预处理、特征与相似度接口。合成输入不验收语义质量，不计算 Flickr30K 指标，不用于选择训练配置。

## 同步代码

PR #3 尚未合并时，从工作分支 clone 到一个不存在的新目录：

```bash
git clone --branch feature/siglip-training-layout --single-branch \
  https://github.com/xiangshui001/challenge-exercise.git ~/challenge-exercise
cd ~/challenge-exercise
python -m pip install -e .
python -m pip check
git rev-parse HEAD
```

已存在 checkout 时先查看分支与 `git status --short`，工作区干净且处于对应分支才 `git pull --ff-only`；不向非空目录重复 clone。PR 合并后可使用 main；不能把 PR 上传当成 main 已更新。

## 运行

在激活的 siglip 环境和仓库根目录中执行：

```bash
python scripts/siglip_smoke.py --gpu-index 2 \
  --revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed \
  --out runs/p1-smoke-001
```

`2` 是本次物理 nvidia-smi 索引，可按实际空闲资源更换。脚本先以 UUID 隔离这一张卡，再加载模型；它不会改驱动或其他进程。processor/权重在 CPU 完成下载与加载后才移到 GPU，短小试结束进程退出释放自己的显存。默认 FP32 便于检查；后续可在新输出目录试 `--precision bfloat16`。此入口为单卡，多卡训练尚未实现。

上面的命令使用已下载的固定 SHA 和 `local_files_only=True`，不连接原站、不重新下载权重。禁止远程自定义代码，仅加载 safetensors。其他环境首次下载时可显式添加 `--allow-download`；需要镜像时，在该命令前加 `HF_ENDPOINT=https://hf-mirror.com`，仅作用于本次进程。未指定 SHA 的在线下载会先解析为不可变 revision，processor/权重均使用同一 SHA；不要用文字占位符冒充 SHA。

已有本地图片时可指定图片与两段英文描述：

```bash
python scripts/siglip_smoke.py --gpu-index 2 \
  --revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed \
  --image data/local/my-photo.jpg \
  --text 'a photo of a cat' 'a photo of a dog' \
  --out runs/p1-photo-001
```

把图片路径/描述换成自己的真实输入。不要在本阶段使用保留的 Flickr30K test 图选择方法。每次使用新输出目录，不覆盖旧日志。下载失败时提供错误末尾；不改共享驱动、不绕过 TLS 检查。

不依赖未经核验的外链测试图片，也不预设某段描述一定分数最高。外部聊天给出的独立 test_siglip.py 不作为第二份项目实现；已创建时先保留并查看文件，以本仓库 checkout 内的脚本继续，避免覆盖用户改动。

## 验收与输出

- `smoke.json`：真实 model_revision、1×D / 2×D 的 FP32 归一化特征形状、有限的余弦相似度、预处理形状、GPU/依赖/代码版本、时间和峰值显存。
- `parameter_names.json`：实际参数名称、形状和数量，后续据此选择部分解冻层；不假定有与 CLIP 同名的 projection。
- `requirements-lock.txt`：成功运行环境的实际 pip freeze。输出留在被忽略的 runs 目录，不默认公开原始环境日志。
- `synthetic-red.png`：仅默认合成输入时生成；真实输入记录文件 SHA。失败记录 failed.json，不能算成功。

相似度是归一化特征的余弦，不是概率或原 sigmoid logits。文本按模型要求小写、padding=max_length、64 token、截断；图片使用同一 revision 的原生 processor。参考 [4.57.3 SigLIP 文档](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/siglip)、[模型卡](https://huggingface.co/google/siglip-base-patch16-224)和 [Hub model_info](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info)。

此阶段的云端检查覆盖余弦数值与形状/有限性/范数拒绝、离线 revision 约束、物理索引到 UUID 映射、输出不覆盖和 CLI 预检。云端未安装 torch/transformers，未下载权重或执行真实模型；真实小试以用户服务器输出为准。`python -m unittest discover -s tests -v` 实际通过 **28 tests，13.449s，OK**（原 24 项 + 新增 4 项）；CLI help、compileall、JSON/相对链接/忽略规则检查和 diff --check 通过。
