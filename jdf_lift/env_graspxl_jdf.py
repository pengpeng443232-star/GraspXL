"""GraspXL JDF reproduction on the lightweight Isaac Lab environment."""
from __future__ import annotations
import shutil, tempfile
from pathlib import Path
import torch
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse
from experiments.lightweight_validation_rl import env_PPO_multiobject as multi
from experiments.lightweight_validation_rl.env_PPO_multiobject import ShovelMultiObjectEnv
from env_graspxl_jdf_cfg import GraspXLJDFEnvCfg

GRASPXL_ROOT = Path(__file__).resolve().parents[2]
OBJECT_NAMES = (
    "Mug_6a9b31e1298ca1109c515ccf0f61e75f_handle",
)
_BOUNDS = (
    ((-.073699,-.048051,-.051545),(.061720,.046434,.054749)),
)
_PLATFORM_TOP_Z = 0.624

def _rigid_urdf_copy(name: str, root: Path) -> Path:
    src = GRASPXL_ROOT / "rsc/mixed_train" / name
    dst = root / name; dst.mkdir(parents=True)
    for f in ("top_watertight_tiny.obj","top_watertight_tiny.stl","bottom_watertight_tiny.obj","bottom_watertight_tiny.stl"):
        shutil.copy2(src / f, dst / f)
    xml = (src / f"{name}.urdf").read_text()
    xml = xml.replace('<joint name="rotation" type="revolute">','<joint name="rotation" type="fixed">')
    start = xml.find("      <dynamics", xml.find('name="rotation"'))
    if start >= 0:
        xml = xml[:start] + xml[xml.find("   </joint>", start):]
    out = dst / f"{name}_rigid.urdf"; out.write_text(xml); return out

def _surface_and_centroid(name: str, count: int, seed: int):
    import numpy as np, trimesh
    src = GRASPXL_ROOT / "rsc/mixed_train" / name
    parts = [trimesh.load(src/f"{p}_watertight_tiny.obj", force="mesh") for p in ("top","bottom")]
    rng = np.random.default_rng(seed)
    def sample(mesh):
        faces=rng.choice(len(mesh.faces),count,p=mesh.area_faces/mesh.area_faces.sum())
        tri=mesh.triangles[faces]; uv=rng.random((count,2)); uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
        return (tri[:,0]+uv[:,:1]*(tri[:,1]-tri[:,0])+uv[:,1:]*(tri[:,2]-tri[:,0])).astype("float32")
    return sample(parts[0]),sample(parts[1]),parts[0].centroid.astype("float32"),parts[1].centroid.astype("float32")

def _pool(root, centers, indices):
    result=[]
    for i in indices:
        name=OBJECT_NAMES[i]
        lo,hi=_BOUNDS[i]; size=tuple(hi[j]-lo[j] for j in range(3)); kp=tuple(min(max(v,.03),.14) for v in size)
        result.append(multi._ObjectSpec(_rigid_urdf_copy(name,root),(-.04,.004,_PLATFORM_TOP_Z-lo[2]+.002),(1.,0.,0.,0.),tuple(centers[i]),size,kp))
    return tuple(result)

class GraspXLJDFEnv(ShovelMultiObjectEnv):
    cfg: GraspXLJDFEnvCfg
    def __init__(self,cfg,render_mode=None,**kwargs):
        self._rigid_dir=Path(tempfile.mkdtemp(prefix="graspxl_rigid_"))
        if cfg.curriculum_stage not in (1,2,3): raise ValueError("curriculum_stage must be 1, 2, or 3")
        # Pipeline-validation mode: keep the same mug in all three stages.
        # JDF size is object-count independent, so multi-object training can
        # later be restored without changing the policy architecture.
        pool_indices=(0,)
        data={i:_surface_and_centroid(OBJECT_NAMES[i],cfg.jdf_surface_samples,cfg.seed+i) for i in pool_indices}
        centers={i:data[i][2] for i in pool_indices}
        old=multi.OBJECT_POOL; multi.OBJECT_POOL=_pool(self._rigid_dir,centers,pool_indices)
        try: super().__init__(cfg,render_mode,**kwargs)
        finally: multi.OBJECT_POOL=old
        import numpy as np
        self._surface_positive=torch.as_tensor(np.stack([data[i][0] for i in pool_indices]),device=self.device)[self._object_index]
        self._surface_negative=torch.as_tensor(np.stack([data[i][1] for i in pool_indices]),device=self.device)[self._object_index]
        self._target_midpoint_local=torch.as_tensor(np.stack([data[i][2] for i in pool_indices]),device=self.device)[self._object_index]
        names=["iiwa14_link_7","left_thumb_MC","left_thumb_MCP_VL","left_thumb_PP","left_thumb_DP",
          "left_index_MCP_VL","left_index_PP","left_index_MP","left_index_DP",
          "left_middle_MCP_VL","left_middle_PP","left_middle_MP","left_middle_DP",
          "left_ring_MCP_VL","left_ring_PP","left_ring_MP","left_ring_DP",
          "left_pinky_MC","left_pinky_MCP_VL","left_pinky_PP","left_pinky_DP"]
        ids={n:i for i,n in enumerate(self.robot.body_names)}; missing=[n for n in names if n not in ids]
        if missing: raise RuntimeError(f"JDF hand bodies missing: {missing}")
        self._jdf_ids=torch.tensor([ids[n] for n in names],device=self.device)
        self._jdf_pos=torch.zeros(self.num_envs,21,3,device=self.device); self._jdf_neg=torch.zeros_like(self._jdf_pos)
        self._dist_pos=torch.zeros(self.num_envs,21,device=self.device); self._dist_neg=torch.zeros_like(self._dist_pos)
        self._desired_palm_quat=self.palm_quat_w.clone(); self._tip_link_ids=torch.tensor([4,8,12,16,20],device=self.device)
    def _pre_physics_step(self,actions):
        # Objective-driven wrist guidance: Cartesian goal error is mapped by
        # damped least-squares Jacobian into the arm's PD-target action.
        if self.cfg.curriculum_stage==3 and hasattr(self,"_jdf_ids"):
            self._task_features()
            jac=self.robot.root_physx_view.get_jacobians()[:,self.palm_id-1,:,self.arm_ids_t]
            twist=torch.cat((self.cfg.guidance_position_gain*self._midpoint_error,
                             self.cfg.guidance_rotation_gain*self._wrist_error),1)
            eye=torch.eye(6,device=self.device).expand(self.num_envs,-1,-1)
            dq=jac.transpose(1,2)@torch.linalg.solve(jac@jac.transpose(1,2)+1e-3*eye,twist.unsqueeze(2))
            actions=actions.clone(); actions[:,:7]+=dq.squeeze(2)/self.cfg.arm_action_scale
        if self.cfg.curriculum_stage in (1,2):
            actions=actions.clone(); actions[:,:7]=0.0
        super()._pre_physics_step(actions)
    def _apply_action(self):
        super()._apply_action()
        if self.cfg.curriculum_stage==1 and hasattr(self,"initial_object_pos"):
            pose=torch.cat((self.initial_object_pos,self.initial_object_quat),dim=1)
            self.shovel.write_root_pose_to_sim(pose)
            self.shovel.write_root_velocity_to_sim(torch.zeros(self.num_envs,6,device=self.device))
    def _compute_jdf(self):
        rel=self.robot.data.body_pos_w[:,self._jdf_ids]-self.object_pos_w[:,None]
        q=self.object_quat_w[:,None].expand(-1,21,-1).reshape(-1,4)
        joints=quat_apply_inverse(q,rel.reshape(-1,3)).reshape(-1,21,3)
        for surface,vectors,distances in ((self._surface_positive,self._jdf_pos,self._dist_pos),(self._surface_negative,self._jdf_neg,self._dist_neg)):
            delta=surface[:,None]-joints[:,:,None]; sq=(delta*delta).sum(-1); idx=sq.argmin(-1)
            vectors.copy_(delta.gather(2,idx[...,None,None].expand(-1,-1,1,3)).squeeze(2))
            distances.copy_(sq.gather(2,idx[...,None]).squeeze(2).sqrt())
    def _task_features(self):
        bodies=self.robot.data.body_pos_w[:,self._jdf_ids]
        self._hand_midpoint=0.5*(bodies[:,4]+bodies[:,11])
        target=quat_apply(self.object_quat_w,self._target_midpoint_local)+self.object_pos_w
        self._midpoint_error=target-self._hand_midpoint
        self._heading=(self._hand_midpoint-self.palm_pos_w); self._heading=self._heading/self._heading.norm(dim=1,keepdim=True).clamp_min(1e-6)
        desired=target-self.object_pos_w; desired=desired/desired.norm(dim=1,keepdim=True).clamp_min(1e-6)
        self._heading_error=desired-self._heading
        _,self._wrist_error=compute_pose_error(self.palm_pos_w,self.palm_quat_w,self.palm_pos_w,self._desired_palm_quat,rot_error_type="axis_angle")
        forces=torch.stack([torch.linalg.vector_norm(s.data.force_matrix_w[:,0,0],dim=-1) for s in self.contact_sensors.values()],1)
        loaded=forces>self.cfg.contact_force_threshold
        pos_closer=self._dist_pos[:,self._tip_link_ids]<=self._dist_neg[:,self._tip_link_ids]
        self._contact_pos=torch.zeros(self.num_envs,21,device=self.device); self._contact_neg=torch.zeros_like(self._contact_pos)
        self._force_pos=torch.zeros_like(self._contact_pos); self._force_neg=torch.zeros_like(self._contact_pos)
        self._contact_pos[:,self._tip_link_ids]=(loaded&pos_closer).float(); self._contact_neg[:,self._tip_link_ids]=(loaded&~pos_closer).float()
        clipped=forces.clamp_max(self.cfg.force_clip)
        self._force_pos[:,self._tip_link_ids]=clipped*pos_closer; self._force_neg[:,self._tip_link_ids]=clipped*(~pos_closer)
    def _get_observations(self):
        if not hasattr(self,"_jdf_ids"): return {"policy":torch.zeros(self.num_envs,self.cfg.observation_space,device=self.device)}
        self._compute_jdf(); self._task_features()
        q=self.robot.data.joint_pos[:,self.hand_ids_t]
        d=self.full_targets-self.robot.data.joint_pos
        uh=torch.cat((self.robot.data.body_lin_vel_w[:,self.palm_id],self.robot.data.body_ang_vel_w[:,self.palm_id]),1)
        uo=torch.cat((self.shovel.data.root_lin_vel_w,self.shovel.data.root_ang_vel_w),1)
        obs=torch.cat((q,d,uh,uo,self._contact_pos,self._contact_neg,self._force_pos,self._force_neg,
                       self._heading_error,self._midpoint_error,self._wrist_error,self._jdf_pos.flatten(1),self._jdf_neg.flatten(1)),1)
        if obs.shape[1]!=self.cfg.observation_space: raise RuntimeError(f"GraspXL feature {obs.shape[1]} != {self.cfg.observation_space}")
        return {"policy":obs}
    def _get_rewards(self):
        lift_reward=super()._get_rewards(); self._compute_jdf(); self._task_features(); c=self.cfg
        rv=-(self._heading_error.square().sum(1))*c.heading_rew_scale
        rw=-(self._wrist_error.square().sum(1))*c.wrist_rew_scale
        rm=-(self._midpoint_error.square().sum(1))*c.midpoint_rew_scale
        rdis=-c.goal_distance_rew_scale*self._dist_pos.square().mean(1)+c.non_grasp_distance_rew_scale*self._dist_neg.clamp_max(.15).square().mean(1)
        rc=c.positive_contact_rew_scale*self._contact_pos.sum(1)-c.negative_contact_rew_scale*self._contact_neg.sum(1)
        rf=c.positive_force_rew_scale*self._force_pos.sum(1)-c.negative_force_rew_scale*self._force_neg.sum(1)
        uh=torch.cat((self.robot.data.body_lin_vel_w[:,self.palm_id],self.robot.data.body_ang_vel_w[:,self.palm_id]),1)
        uo=torch.cat((self.shovel.data.root_lin_vel_w,self.shovel.data.root_ang_vel_w),1)
        reg=-c.hand_velocity_rew_scale*uh.square().sum(1)-c.object_velocity_rew_scale*uo.square().sum(1)
        goal=rdis+rv+rw+rm; grasp=rc+rf+reg
        reward=goal if c.curriculum_stage==1 else goal+grasp
        if c.curriculum_stage==3: reward=reward+lift_reward
        # Do not latch an earlier transient lift as final success: the object
        # must still be at the 10 cm goal with a valid grasp at sequence end.
        self.success_latched.copy_(self._is_success & self.valid_contact)
        self.extras["successes"]=self.success_latched.float()
        self._reward_terms.update(r_distance=rdis,r_heading=rv,r_wrist=rw,r_midpoint=rm,r_contact=rc,r_force=rf,r_regularization=reg,total_reward=reward)
        self.extras["episode_cumulative"]=self._reward_terms
        return reward
    def close(self):
        try: super().close()
        finally: shutil.rmtree(self._rigid_dir,ignore_errors=True)
