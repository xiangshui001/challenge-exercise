# 12｜服务器数据准备与原始划分核对

## 已确认的环境

用户 2026-10-11 上传的终端记录确认：仓库已克隆至 `/workspace/challenge-exp5/repo`；独立 `challenge-exp5` 环境为 torch 2.8.0+cu126、torchvision 0.23.0+cu126、transformers 4.57.3、huggingface_hub 0.36.2；`pip check` 通过，四张约 80GB A100 的 BF16 小矩阵检查均通过。最新记录确认修复分支的 26 项 CPU 测试通过（41.807s），划分压缩包下载完成；图片下载因 Hugging Face 连接不可达而失败。这不是 CLIP/SigLIP 2 模型验收或真实检索成绩。

**2026-10-19 更新**：图片下载已通过代理恢复，真实数据审计 `passed`、`standard_protocol=true`，图像 val/test=1014/1000，描述=5070/5000；图片内容重复组为空，4 组重复描述按原始标注保留。下面的网络排障保留为恢复步骤；当前下一步是 [20 图模型小试](10_RUNNING_BASELINE.md)。

## 数据数量更正

2026-10-11 实际读取[作者下载包](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)（36,745,453 字节）内 `dataset_flickr30k.json`，计数如下：

| 划分 | 图像 | 描述 | 用途 |
|---|---:|---:|---|
| train | 29000 | 本批不用 | 不开展本地训练 |
| val | 1014 | 5070 | 参数选择和确认 |
| test | 1000 | 5000 | 冻结后正式测试 |

验证组按图像分成 507/507，两组均面对完整 1014 图像或 5070 描述候选库。不要任意删掉 14 张验证图片。程序新增 `--expected-val-images 1014`，将声明纳入来源记录和数据指纹，并在冻结配置写出实际规模。默认 1000 仅保留已有明确来源的 1000 验证图模式；不能按测试效果选择规模。

图片使用 [nlphuji/flickr30k 镜像](https://huggingface.co/datasets/nlphuji/flickr30k/tree/2b239befc81b6e3f035ce6bd52f5f4d60f5625f7) 的 `flickr30k-images.zip`，约 4.39GB。镜像外层 `test` 包含约 31k 图，不是本实验 1000 图正式测试划分；评测划分和描述来自作者 JSON。图片来源按镜像记录，数量与文件匹配由审计验证，不声称已核实与原始作者图片逐字节一致。

## 1. 保存依赖记录

```bash
conda activate challenge-exp5
cd /workspace/challenge-exp5/repo
mkdir -p data/local/raw runs/environment-20261011
python -m pip freeze > runs/environment-20261011/requirements-lock.txt
```

安装命令、当前服务器检查和下载命令分阶段执行。安装本身不会下载模型或启动 GPU 推理。

## 2. 下载两个文件

```bash
curl -fL --retry 3 --connect-timeout 20 \
  -o data/local/raw/caption_datasets.zip \
  https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

hf download nlphuji/flickr30k flickr30k-images.zip \
  --repo-type dataset \
  --revision 2b239befc81b6e3f035ce6bd52f5f4d60f5625f7 \
  --local-dir data/local/raw
```

这些参数适用于已安装的 huggingface_hub 0.36.2。不要直接 `load_dataset` 执行镜像中的旧加载脚本，也无需为下载额外安装或升级 `datasets`。连接失败时先查看错误；HTTP 代理隧道可通过单条命令的 `http_proxy`/`https_proxy` 指定，不关闭 TLS 校验。

### 2.1 Hugging Face 直连失败时复用电脑代理

最新报错发生在 Hugging Face 的 HEAD 请求，原因为 `[Errno 101] Network is unreachable`；划分包已下载，无需重做安装或测试。先沿用已建立的隧道；若尚未建立，在电脑 PowerShell 中运行以下命令。示例假定代理软件的 HTTP/混合端口为 7890；将 `SERVER_HOST` 和 SSH 端口换成实际值，7890 换成代理软件显示的端口：

```powershell
curl.exe -I --proxy http://127.0.0.1:7890 --connect-timeout 10 --max-time 30 https://huggingface.co
ssh -p 38 -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:17890:127.0.0.1:7890 root@SERVER_HOST
```

先确认电脑 curl 得到站点成功响应，再启动 SSH；按提示输入密码，保持 SSH 窗口打开。`-R` 将服务器回环端口 17890 转发到电脑的代理端口，语义见 [OpenSSH 手册](https://man.openbsd.org/ssh.1#R)。然后在服务器终端验证整条代理路径：

```bash
curl --proxy http://127.0.0.1:17890 --noproxy '' \
  --silent --show-error --head --output /dev/null \
  --write-out 'HTTP %{http_code}\n' \
  --connect-timeout 10 --max-time 30 https://huggingface.co
```

应输出 `HTTP 200`。若 17890 连接被拒绝，先检查隧道；若 SSH 提示转发端口占用，先检查已有隧道，不重复启动或结束其他人的进程。代理验证通过后，只重试图片下载：

```bash
env http_proxy=http://127.0.0.1:17890 \
    https_proxy=http://127.0.0.1:17890 \
    no_proxy=localhost,127.0.0.1 \
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=120 \
    hf download nlphuji/flickr30k flickr30k-images.zip \
      --repo-type dataset \
      --revision 2b239befc81b6e3f035ce6bd52f5f4d60f5625f7 \
      --local-dir data/local/raw
```

这些环境变量只作用于该下载进程。关闭 Xet 和 hf_transfer，使用普通 HTTP 下载；提高的是请求等待时间，不是允许跳过校验。依据 [Hub 0.36 环境变量文档](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/environment_variables)及 [Requests 代理文档](https://requests.readthedocs.io/en/latest/user/advanced/#proxies)。这里只验证了命令参数依据，实际电脑代理和 SSH 隧道仍须由用户终端确认；首页连通后若文件下载失败，继续保留该次错误以检查重定向下载端点。

## 3. 解压与审计

先完成两个下载，再运行：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
raw = Path('data/local/raw')
with ZipFile(raw / 'caption_datasets.zip') as archive:
    archive.extract('dataset_flickr30k.json', raw)
with ZipFile(raw / 'flickr30k-images.zip') as archive:
    archive.extractall(raw)
print('解压完成')
PY

python -m exp5 prepare \
  --annotations data/local/raw/dataset_flickr30k.json \
  --image-root data/local/raw/flickr30k-images \
  --expected-val-images 1014 \
  --source-revision 'Karpathy author caption_datasets.zip retrieved 2026-10-11; images nlphuji/flickr30k@2b239befc81b6e3f035ce6bd52f5f4d60f5625f7' \
  --out data/local/manifests
```

输出应为 `status=passed`、`standard_protocol=true`，图像 val/test=1014/1000，描述=5070/5000。审计会检查图片可解码、ID 和正例、每图五描述、重复及跨划分相同字节、路径和固定分组；缺图或坏图会停止，不静默跳过。每次审计使用新的输出目录，失败记录也保留。

## 本批验收边界

本批 CPU **26 项测试通过**，覆盖旧 1000/1000 模式及 1014/1000 模式的审计、冻结和测试；2026-10-19 用户真实图片审计也已通过，证据见 [11](11_CPU_VALIDATION.md)。真实模型加载和 GPU 检索仍待小试验收。数据、压缩包、依赖锁和原始运行记录均保留在本地忽略目录，公开 PR 只包含代码、测试及必要的小型说明。
