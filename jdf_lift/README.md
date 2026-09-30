# GraspXL JDF on lightweight_validation_rl

This version uses the Isaac Lab multi-object environment underlying
`train_ppo_multiobject.py` and does not import RaiSim.

It retains the 29D KUKA–Sharpa direct-joint action, five fingertip contact
sensors, and 10 cm goal. The lateral shake-check state machine and its hard
gate are intentionally not used. For initial pipeline validation, every
environment uses the same GraspXL mug. Its nearly locked two-link URDF is
copied temporarily and merged into one rigid body; source assets are unchanged.

The mug handle is the positive/graspable region and its body is the negative
region. For each region, 256 deterministic surface points are sampled. Policy
input is the 282D GraspXL feature: finger angles, joint-to-PD-target error,
hand/object velocity, positive/negative contact flags and forces, heading,
midpoint and wrist errors, and 21-link positive/negative JDF vectors. Only the
five fingertip contacts are currently measured by the base environment; other
link contact/force entries are zero until 21-link sensors are added.

The 29D policy action specifies incremental arm and absolute finger joint PD
targets. Isaac Lab's implicit actuators convert target error to joint torque.
In stage 3, midpoint and wrist objective errors are mapped through the arm
Jacobian and added as a PD-target bias (objective-driven hand guidance).

The copied lightweight folder lacks the complete SimToolReal robot assets/base
module, so pass the full checkout (the default is `/home/robot/simtoolreal`):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
cd /home/robot/raisim/GraspXL
/home/robot/simtoolreal/.venv_isaacsim/bin/python experiments/jdf_lift/train.py \
  --simtoolreal-root /home/robot/simtoolreal --headless --dry-run --num-envs 10
```

Remove `--dry-run` to train.

## Three-stage curriculum

All stages use the same 282D observation and 29D action, so checkpoints transfer
without changing the network. Stage 1 spawns only the mug, holds both object and
arm fixed, and emphasizes JDF accuracy. Stage 2 restores a dynamic mug while
keeping the arm fixed; it emphasizes sustained real multi-finger contact. Stage
3 keeps the same mug, unlocks the arm, and adds the ordinary 10 cm lift and
pose-goal reward.

Run each command after the preceding stage converges, passing its latest `.pth`
checkpoint to the next command with `--checkpoint ... --checkpoint-load-mode weights`:

```bash
# Stage 1: fixed cup, grasp accurately
.../python experiments/jdf_lift/train.py --config experiments/jdf_lift/cfg_stage1_cup_accuracy.yaml --headless

# Stage 2: dynamic cup, stable grasp
.../python experiments/jdf_lift/train.py --config experiments/jdf_lift/cfg_stage2_stable_grasp.yaml --checkpoint STAGE1.pth --checkpoint-load-mode weights --headless

# Stage 3: stable grasp and 10 cm lift
.../python experiments/jdf_lift/train.py --config experiments/jdf_lift/cfg_stage3_lift_10cm.yaml --checkpoint STAGE2.pth --checkpoint-load-mode weights --headless
```
