# 目标对象属性报告的机制识别

本地做理论验证，真实网络训练与干预实验在单张 RTX 5090 32GB 服务器运行。
使用已安装的 Python 环境依赖；不创建额外的依赖目录。

## 目录

```text
measure_distinguish/
  config.py             公共配置与模型/数据默认值
  data.py               合成场景与配对干预
  coco.py               COCO 实例读取
  models.py             SmallViT、ResNet-18、DINOv2-S/14
  train.py              训练、checkpoint、预测
  run_experiment.py     真实实验入口，与 train.py 同级
configs/
  experiment.json       唯一公共实验配置
local_theory/
  mechanisms.py         理论机制与条件诊断
  run.py                理论恒等式、反例与有限样本模拟
  results/
    REPORT.md           理论结果统一汇总
    summary.json        对应数值
scripts/
  download_models.py    模型权重下载
  download_data.py      COCO 数据下载
weights/                模型权重，不入库
outputs/                仅保存真实实验输出，不入库
```

## 模型与数据选择

首轮使用从头训练的 SmallViT（6 blocks、width=192、patch=8）与 ResNet-18。
后续用 DINOv2-S/14 进行预训练迁移，并保留同架构随机初始化对照。
主数据是受控的目标方向报告场景；扩展数据是 COCO 非 crowd 实例的相对旋转报告。
旧 object-binding 实验只作为设计参考，不复用它的外部 probe 作为本任务输出。

合成训练中目标与干扰方向来自相同的 [-45°,45°] 分布，蓝环提示目标。
机制干预包含方向差扫描、分别改变目标/干扰物、同时改变、属性绑定交换与 cue 交换。
校准方向差是 0、±5°、±10°，留出条件是 ±20°、±30°。
同 family 固定布局和背景，按 family 隔离训练、开发与干预 split。

COCO 按源图像隔离五个 split。同一场景使用同一源实例的旋转副本，左上角给出固定的零旋转参考；任务报告相对该参考的角度，不是自然物体的真实姿态。
所有模型只接收 RGB 图像。真实方向和 mask 只用于监督与分析。

## 本地理论验证

```powershell
& 'H:\anaconda3\envs\py312\python.exe' local_theory/run.py
```

结果统一写入 `local_theory/results/REPORT.md` 和 `summary.json`，覆盖既有汇总，不另建时间戳或重复结果目录。
默认检查多对象方差恒等式、one-hot 等价边界、有界噪声漂移和上下文污染反例，
再运行 N=128/512/2048、8 种机制、各 100 重复、500 bootstrap 的模拟。
仅重建已有汇总及复验恒等式：

```powershell
& 'H:\anaconda3\envs\py312\python.exe' local_theory/run.py --report-only
```

这是理论验证。模型前后向、短训练和保存恢复不作为理论研究结果。
诊断区分“均值模型失败”“假设未验证”“无法区分”和“有条件拒绝干净选择”。
无法从输出区分上下文污染后选择与直接聚合，也不能把模拟中的噪声预算当作真实网络证据。

## 单独下载

下载 DINOv2-S/14，只保存权重，不生成新配置：

```bash
python scripts/download_models.py
# 可选公共镜像：
python scripts/download_models.py --endpoint https://hf-mirror.com
```

SmallViT 与 ResNet-18 从头训练，不需要下载权重。
COCO 扩展使用 train2017 和 instances_train2017.json：

```bash
python scripts/download_data.py --root datasets/coco
# 只下载标注：
python scripts/download_data.py --root datasets/coco --annotations-only
```

主合成数据在线生成，不需下载。数据脚本支持中断续传，解压后移除 zip，不保存无关 COCO 标注。

## 服务器真实实验

在项目根目录运行。使用匹配服务器 GPU 的 CUDA PyTorch/torchvision 环境，然后安装 `requirements-server.txt`。

```bash
python -m measure_distinguish.run_experiment --stage plan
python -m measure_distinguish.run_experiment --stage train
python -m measure_distinguish.run_experiment --model resnet18 --stage train
python -m measure_distinguish.run_experiment --model dinov2_small --stage train
python -m measure_distinguish.run_experiment --model dinov2_small --initialization random --stage train
python -m measure_distinguish.run_experiment --dataset coco_rotation --data-root datasets/coco --stage train
```

也支持直接执行 `python measure_distinguish/run_experiment.py`。默认 stage=plan，不启动训练。
一份公共配置控制数据量、epoch、seed、精度等；选择模型/数据时自动应用相应的分辨率、batch 和学习率默认值，plan 显示最终参数。
SmallViT/ResNet 的合成图像为 128，DINOv2 或 COCO 为 224；DINOv2 batch=32，其余为 64；预训练适配 lr=1e-5，其余为 3e-4。
如需手动设置，直接修改同一份公共配置里的 data.size、training.batch_size、training.lr。
不同模型的配方不完全相同，不能将结果差异唯一归因于预训练或 attention。

默认输出为 `outputs/<模型>_<数据>_s<seed>/`，DINOv2 另区分预训练/随机初始化。
每个 run 仅保存 `last.pt`、`best.pt` 和 `metrics.json`；配置随 checkpoint 保存，不再复制配置、源码摘要和审计文件。
last 保存 optimizer、scheduler、RNG，支持从完整 epoch 恢复；best 只保存任务模型及其配置。
恢复时只比较实际训练配置，不计算源码/产物哈希。

```bash
python -m measure_distinguish.run_experiment --stage train --resume
python -m measure_distinguish.run_experiment --stage evaluate --checkpoint outputs/small_vit_synthetic_s0/best.pt --split calibration
python -m measure_distinguish.run_experiment --stage evaluate --checkpoint outputs/small_vit_synthetic_s0/best.pt --split validation
```

评估的逐样本响应留在同一 run 的 `predictions_<split>.jsonl`，指标汇入该 run 的 `metrics.json`。
默认训练只使用 train/dev；确定诊断规则后再显式评估 test。程序不自动给网络贴机制标签。

完整研究设计见 `EXPERIMENT_PLAN.md`。目前尚未执行服务器正式实验。
