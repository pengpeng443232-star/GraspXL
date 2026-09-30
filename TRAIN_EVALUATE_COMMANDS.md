# GraspXL 训练与评估命令

本文档统一从 GraspXL 仓库根目录或 `raisimGymTorch` 目录启动程序，避免相对路径解析错误。

## 1. 环境准备

原版 GraspXL（RaiSim）：

```bash
conda activate graspxl5080
cd /home/robot/raisim/GraspXL/raisimGymTorch
```

当前 JDF lightweight/Isaac Sim 实验：

```bash
cd /home/robot/raisim/GraspXL
PYTHON=/home/robot/simtoolreal/.venv_isaacsim/bin/python
SIMTOOLREAL_ROOT=/home/robot/simtoolreal
```

> 原版 GraspXL 依赖 RaiSim license；`experiments/jdf_lift` 使用 simtoolreal 中的 lightweight validation 环境，不依赖 RaiSim license。

## 2. JDF Lift：单杯分阶段训练

三个阶段应按顺序训练：抓准位置、稳定抓取、抬升 10 cm。

### 配置与网络构建检查

```bash
$PYTHON experiments/jdf_lift/train.py \
  --simtoolreal-root "$SIMTOOLREAL_ROOT" \
  --config experiments/jdf_lift/cfg_stage1_cup_accuracy.yaml \
  --num-envs 64 \
  --dry-run
```

### Stage 1：杯子固定，学习抓准

```bash
$PYTHON experiments/jdf_lift/train.py \
  --simtoolreal-root "$SIMTOOLREAL_ROOT" \
  --config experiments/jdf_lift/cfg_stage1_cup_accuracy.yaml \
  --num-envs 64
```

### Stage 2：杯子解锁，学习稳定抓取

```bash
$PYTHON experiments/jdf_lift/train.py \
  --simtoolreal-root "$SIMTOOLREAL_ROOT" \
  --config experiments/jdf_lift/cfg_stage2_stable_grasp.yaml \
  --num-envs 64
```

### Stage 3：稳定抬升 10 cm

```bash
$PYTHON experiments/jdf_lift/train.py \
  --simtoolreal-root "$SIMTOOLREAL_ROOT" \
  --config experiments/jdf_lift/cfg_stage3_lift_10cm.yaml \
  --num-envs 64
```

### 默认综合配置

```bash
$PYTHON experiments/jdf_lift/train.py \
  --simtoolreal-root "$SIMTOOLREAL_ROOT" \
  --config experiments/jdf_lift/cfg.yaml \
  --num-envs 64
```

### Joint Distance Feature 单元检查

```bash
$PYTHON experiments/jdf_lift/test_joint_distance.py
```

> 当前 `jdf_lift` 目录没有独立的 evaluation 入口。不要把 `--dry-run` 当成策略评估：它只检查环境、配置及算法对象能否成功构建。正式 checkpoint 评估需要后续增加专用 evaluate 脚本或在 `train.py` 中增加 checkpoint/test 模式。

## 3. 原版 GraspXL 训练

以下命令均在 `/home/robot/raisim/GraspXL/raisimGymTorch` 下执行。

### Allegro Hand

固定手腕：

```bash
python raisimGymTorch/env/envs/allegro_fixed/runner.py
```

浮动手腕：

```bash
python raisimGymTorch/env/envs/allegro_floating/runner.py
```

原始备份入口（仅用于复现实验或对照，不建议作为默认入口）：

```bash
python raisimGymTorch/env/envs/allegro_fixed/runner_original.py
python raisimGymTorch/env/envs/allegro_floating/runner_orinial.py
```

### Ours / MANO Hand

固定手腕：

```bash
python raisimGymTorch/env/envs/ours_fixed/runner.py
```

浮动手腕：

```bash
python raisimGymTorch/env/envs/ours_floating/runner.py
```

### Shadow Hand

固定手腕：

```bash
python raisimGymTorch/env/envs/shadow_fixed/runner.py
```

浮动手腕：

```bash
python raisimGymTorch/env/envs/shadow_floating/runner.py
```

每个 runner 默认读取同目录 `cfgs/cfg_reg.yaml`。运行前应在对应配置或 runner 中确认数据集路径、实验名、并行环境数、保存目录和恢复 checkpoint 设置。

## 4. 原版 GraspXL Evaluation

### Allegro Hand

```bash
python raisimGymTorch/env/envs/allegro_test/mini_unseen_eval.py
python raisimGymTorch/env/envs/allegro_test/shapenet_eval.py
python raisimGymTorch/env/envs/allegro_test/partnet_eval.py
```

生成 mini-unseen 测试标签：

```bash
python raisimGymTorch/env/envs/allegro_test/generate_mini_test_labels.py
```

单物体可视化 demo：

```bash
python raisimGymTorch/env/envs/allegro_demo/demo.py
```

### Ours / MANO Hand

```bash
python raisimGymTorch/env/envs/ours_test/shapenet_eval.py
python raisimGymTorch/env/envs/ours_test/partnet_eval.py
python raisimGymTorch/env/envs/ours_test/objaverse_eval.py
python raisimGymTorch/env/envs/ours_test/gen_recon_eval.py
```

可视化 demo：

```bash
python raisimGymTorch/env/envs/ours_demo/demo.py
python raisimGymTorch/env/envs/ours_demo/objaverse_demo.py
```

### Shadow Hand

```bash
python raisimGymTorch/env/envs/shadow_test/shapenet_eval.py
python raisimGymTorch/env/envs/shadow_test/partnet_eval.py
```

可视化 demo：

```bash
python raisimGymTorch/env/envs/shadow_demo/demo.py
```

## 5. RaiSimUnity 可视化

先启动 Unity 客户端：

```bash
mkdir -p /tmp/raisim-unity-libs
ln -sf /lib/x86_64-linux-gnu/libdl.so.2 /tmp/raisim-unity-libs/libdl.so

cd /home/robot/raisim/GraspXL
LD_LIBRARY_PATH=/tmp/raisim-unity-libs:$LD_LIBRARY_PATH \
  ./raisimUnity/linux/raisimUnity.x86_64
```

若提示缺少 `libminizip.so.1`：

```bash
sudo apt update
sudo apt install libminizip1
```

随后在另一个终端运行相应 demo。一个 RaiSimUnity 实例只连接一个正在可视化的 Python 仿真进程；训练与 demo 同时抢占连接时可能出现 `Connection reset by peer`。

## 6. 使用 checkpoint 前的检查

原版 demo/evaluation 脚本中的 checkpoint 路径通常直接写在脚本或配置中。运行前检查：

```bash
rg -n "weight_path|checkpoint|full_.*\.pt|load_param" \
  raisimGymTorch/env/envs
```

确保打印出的 `loading from the checkpoint:` 路径真实存在。例如 Allegro demo 当前使用的权重应位于其配置的 `data_all/allegro_floating` 实验目录中。

