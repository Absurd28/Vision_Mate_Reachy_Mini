import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import queue
import os

# Robust asset loading
try:
    import reachy_mini
    HAS_REACHY_SDK = True
except ImportError:
    HAS_REACHY_SDK = False

class ReachyRobot:
    """
    Stabilized Robot Controller for Reachy Mini.
    Features: Surgical Actuator Sync, Environment Isolation, and Pre-flight Stability Checks.
    """
    
    def __init__(self, cmd_queue):
        # 1. ASSET SELECTION
        self.model_path = "custom_reachy.xml"
        if HAS_REACHY_SDK:
            try:
                package_path = os.path.dirname(reachy_mini.__file__)
                official_path = os.path.join(package_path, 'daemon', 'assets', 'reachy_mini.xml')
                if os.path.exists(official_path):
                    self.model_path = official_path
            except: pass
            
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        
        # 2. PHYSICS INITIALIZATION
        self.model.opt.timestep = 0.002
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_PGS
        self.model.opt.iterations = 100
        
        # Lift the base body to Z=0.8 to ensure isolation from any floor/table clipping
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "reachy_base")
        if base_id != -1:
            self.model.body_pos[base_id][2] = 0.8

        self.data = mujoco.MjData(self.model)

        # --- CRITICAL FIX: SURGICAL ACTUATOR SYNC ---
        # Stop all initial momentum
        self.data.qvel[:] = 0 
        
        # Compute kinematics for current safe spawn pose
        mujoco.mj_forward(self.model, self.data)
        
        # Map each actuator strictly to its corresponding joint position
        # This prevents array mismatch crashes (NaN/Inf)
        for i in range(self.model.nu):
            # trnid[i, 0] gives the ID of the joint this actuator is attached to
            joint_id = self.model.actuator_trnid[i, 0]
            # jnt_qposadr gives the starting index in the qpos array for that joint
            qpos_adr = self.model.jnt_qposadr[joint_id]
            # Set the control target to the current joint position
            self.data.ctrl[i] = self.data.qpos[qpos_adr]
        
        # --- PRE-FLIGHT STABILITY CHECK ---
        try:
            # Test step exactly once
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise ValueError("NaN detected in joint positions during warm-up.")
            print("ROBOT: Physics stabilized. Pre-flight check PASSED.")
        except Exception as e:
            print(f"CRITICAL PHYSICS ERROR: {e}")
            print("Ensure joint limits and actuator gains are balanced.")

        # 3. WORKSPACE & COMPONENT MAPPING
        self.WORKSPACE_COORDS = {
            "bottle_loc":    [0.45, 0.0, 0.42],  
            "drop_bin_loc":  [0.30, 0.4, 0.45],  
            "home_pose":     [0.30, 0.0, 0.50],  
            "user_hand_loc": [0.55, -0.2, 0.55], 
            "idle":          [0.30, 0.0, 0.50]
        }

        # Arm and Site Mapping
        arm_joints = ["r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll"]
        self.arm_actuators = []
        self.arm_jnt_dof_indices = []
        for n in arm_joints:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"a_{n}")
            if aid == -1: aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if aid != -1 and jid != -1:
                self.arm_actuators.append(aid)
                self.arm_jnt_dof_indices.append(self.model.jnt_dofadr[jid])

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_end_effector")

        # --- TASK ENGINE STATE ---
        self.cmd_queue = cmd_queue 
        self.task_sequence = []
        self.current_step = None
        self.wait_timer = 0
        self.is_moving = False
        self.start_pos = np.zeros(3)
        self.goal_pos = np.array(self.WORKSPACE_COORDS["idle"])
        self.move_start_time = 0
        self.move_duration = 3.0
        self.current_target = self.goal_pos.copy()

    def solve_ik(self, target_pos):
        """Jacobian-based Inverse Kinematics."""
        step_size, damping = 0.3, 0.1
        for _ in range(25):
            mujoco.mj_fwdPosition(self.model, self.data)
            current_pos = self.data.site_xpos[self.site_id]
            error = target_pos - current_pos
            if np.linalg.norm(error) < 0.005: break
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, self.site_id)
            jac_arm = jac[:, self.arm_jnt_dof_indices]
            n = jac_arm.shape[0]
            dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + damping * np.eye(n)) @ error
            for i, dof_idx in enumerate(self.arm_jnt_dof_indices):
                q_idx = self.model.jnt_qposadr[self.model.dof_jntid[dof_idx]]
                self.data.qpos[q_idx] += dq[i] * step_size
        return [self.data.qpos[self.model.jnt_qposadr[self.model.dof_jntid[d]]] for d in self.arm_jnt_dof_indices]

    def run_physics(self, log_callback):
        self.running = True
        log_callback("Robot: Physics thread active and stabilized.")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while self.running and viewer.is_running():
                step_start = time.time()

                # Process task sequence
                if not self.current_step and self.task_sequence:
                    self.current_step = self.task_sequence.pop(0)
                    self.initiate_step(self.current_step, log_callback)

                if self.current_step and self.is_step_done():
                    self.current_step = None 

                # Trajectory updates
                if self.is_moving:
                    t_elapsed = time.time() - self.move_start_time
                    alpha = np.clip(t_elapsed / self.move_duration, 0, 1)
                    smooth_alpha = 3*(alpha**2) - 2*(alpha**3)
                    self.current_target = self.start_pos + (self.goal_pos - self.start_pos) * smooth_alpha
                    if alpha >= 1.0: self.is_moving = False

                # Control actuation
                target_q = self.solve_ik(self.current_target)
                for i, aid in enumerate(self.arm_actuators):
                    if i < len(target_q): self.data.ctrl[aid] = target_q[i]

                # Command polling
                try:
                    intent = self.cmd_queue.get_nowait()
                    self.load_sequence(intent, log_callback)
                except queue.Empty: pass

                # Simulation update
                for _ in range(5): mujoco.mj_step(self.model, self.data)
                viewer.sync()

                elapsed = time.time() - step_start
                if elapsed < 0.01: time.sleep(0.01 - elapsed)

    def initiate_step(self, step, log_callback):
        if step["type"] == "move": self.trigger_move(step["val"])
        elif step["type"] == "wait": self.wait_timer = time.time() + step["val"]

    def is_step_done(self):
        if not self.current_step: return True
        if self.current_step["type"] == "move": return not self.is_moving
        if self.current_step["type"] == "wait": return time.time() >= self.wait_timer
        return True

    def load_sequence(self, intent, log_callback):
        self.task_sequence = []
        if intent == "fetch_object" or intent == "fetch_bottle":
            self.task_sequence = [
                {"type": "move", "val": [0.4, 0.0, 0.6]},
                {"type": "move", "val": self.WORKSPACE_COORDS["bottle_loc"]},
                {"type": "wait", "val": 1.0},
                {"type": "move", "val": self.WORKSPACE_COORDS["user_hand_loc"]}
            ]
        elif intent == "stop_robot":
            self.is_moving = False
            self.task_sequence = []

    def trigger_move(self, new_goal):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.start_pos = self.data.site_xpos[self.site_id].copy()
        self.goal_pos = np.array(new_goal)
        self.move_start_time = time.time()
        self.is_moving = True

    def start(self, log_callback):
        threading.Thread(target=self.run_physics, args=(log_callback,), daemon=True).start()

    def stop(self): self.running = False
