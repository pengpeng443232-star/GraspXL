#!/usr/bin/env python3
"""Train GraspXL JDF with the lightweight multi-object lift environment."""
from __future__ import annotations
import argparse, copy, math, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; GRASPXL_ROOT=HERE.parents[1]

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simtoolreal-root",type=Path,default=Path(os.environ.get("SIMTOOLREAL_ROOT","/home/robot/simtoolreal")))
    p.add_argument("--config",type=Path,default=HERE/"cfg.yaml"); p.add_argument("--checkpoint")
    p.add_argument("--checkpoint-load-mode",choices=("resume","weights"),default="weights")
    p.add_argument("--rl-device",default="cuda:0"); p.add_argument("--train-dir",default=str(GRASPXL_ROOT/"experiments/runs/jdf_lift"))
    p.add_argument("--num-envs",type=int); p.add_argument("--seed",type=int); p.add_argument("--dry-run",action="store_true")
    return p

def main():
    pre=argparse.ArgumentParser(add_help=False); pre.add_argument("--simtoolreal-root",type=Path,default=Path(os.environ.get("SIMTOOLREAL_ROOT","/home/robot/simtoolreal")))
    root=pre.parse_known_args()[0].simtoolreal_root.resolve()
    if not (root/"experiments/lightweight_validation/assets.py").is_file(): raise FileNotFoundError(f"incomplete SimToolReal checkout: {root}")
    sys.path[:0]=[str(HERE),str(root)]
    from isaaclab.app import AppLauncher
    p=parser(); AppLauncher.add_app_launcher_args(p); args=p.parse_args(); app=AppLauncher(args).app
    from rl_games.torch_runner import Runner
    from isaacsimenvs.utils.rlgames_utils import EnvStatsAlgoObserver,register_rlgames_env
    from experiments.lightweight_validation_rl.train_ppo import _apply_shovel_ppo_config,_load_yaml
    from env_graspxl_jdf import GraspXLJDFEnv
    from env_graspxl_jdf_cfg import GraspXLJDFEnvCfg
    exp=_load_yaml(args.config); raw=dict(exp["environment"])
    extra={k:raw.pop(k) for k in list(raw) if hasattr(GraspXLJDFEnvCfg(),k) and k not in {
      "num_envs","episode_length_s","decimation","arm_action_scale","hand_stiffness","goal_height_offset","goal_epsilon",
      "z_drop_threshold","max_palm_x_env","max_palm_y_env","joint_velocity_limit","object_escape_distance","transition_budget"}}
    cfg=_apply_shovel_ppo_config(GraspXLJDFEnvCfg(),raw)
    for k,v in extra.items(): setattr(cfg,k,type(getattr(cfg,k))(v))
    if args.num_envs: cfg.scene.num_envs=args.num_envs
    base=root/"experiments/lightweight_validation_rl/cfg"
    agent=copy.deepcopy(_load_yaml(base/"rl_games_ppo_shovel_22D.yaml")); agent["params"]["config"].update(exp.get("ppo",{}))
    # This SimToolReal RL-Games fork parses the leading integer as policy_idx.
    ac=agent["params"]["config"]; ac["name"]=ac["full_experiment_name"]=f"0_graspxl_jdf_stage{cfg.curriculum_stage}"; ac["num_actors"]=cfg.scene.num_envs
    ac["max_epochs"]=math.ceil(cfg.transition_budget/(cfg.scene.num_envs*ac["horizon_length"]))
    cfg.seed=agent["params"]["seed"]=args.seed if args.seed is not None else int(agent["params"]["seed"]); cfg.sim.device=args.device
    print(f"[graspxl_jdf] stage={cfg.curriculum_stage} envs={cfg.scene.num_envs} obs={cfg.observation_space} act={cfg.action_space} object=mug goal=0.10m")
    env=GraspXLJDFEnv(cfg); register_rlgames_env(env,rl_device=args.rl_device,clip_obs=float(agent["params"]["env"]["clip_observations"]),clip_actions=float(agent["params"]["env"]["clip_actions"]))
    ac["device"]=ac["device_name"]=args.rl_device; ac["train_dir"]=str(Path(args.train_dir).resolve())
    run=Runner(algo_observer=EnvStatsAlgoObserver()); run.load(agent); run.reset()
    if args.dry_run: run.algo_factory.create(run.algo_name,base_name="run",params=run.params); print("[graspxl_jdf] dry-run PASS")
    else: run.run({"train":True,"play":False,"checkpoint":args.checkpoint,"checkpoint_load_mode":args.checkpoint_load_mode})
    env.close(); del app
if __name__=="__main__": main()
