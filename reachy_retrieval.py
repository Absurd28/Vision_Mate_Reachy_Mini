import mujoco
import mujoco.viewer
import numpy as np
import time

# --- CONFIGURATION ---
MODEL_PATH = "custom_reachy.xml"
TABLE_HEIGHT = 0.42  # Z-coordinate of the table surface
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

class ReachyVisionBridge:
    """
    Simulates the interface between a CV pipeline (eye-tracking) 
    and the robot's control system.
    """
    def __init__(self):
        self.last_input = (0.5, 0.5)  # Normalized coordinates (x, y)

    def get_eye_cursor_coords(self):
        """
        PLACEHOLDER: In a live system, this would read from a socket 
        or a shared memory buffer populated by an external eye-tracking script.
        """
        # For simulation, we could use mouse position or a predefined path
        # Returning normalized coordinates [0, 1]
        return self.last_input

    def update_input(self, x, y):
        """Method to simulate moving the eye cursor."""
        self.last_input = (x, y)

def map_2d_to_3d(normalized_coords, table_bounds):
    """
    Maps 2D screen/eye coordinates to 3D workspace coordinates.
    Assumes the target is on the table surface (fixed Z).
    
    table_bounds: [x_min, x_max, y_min, y_max]
    """
    nx, ny = normalized_coords
    x_min, x_max, y_min, y_max = table_bounds
    
    # Map normalized X [0, 1] to table X [x_min, x_max]
    target_x = x_min + nx * (x_max - x_min)
    # Map normalized Y [0, 1] to table Y [y_min, y_max]
    target_y = y_min + ny * (y_max - y_min)
    
    return np.array([target_x, target_y, TABLE_HEIGHT])

def solve_ik(model, data, target_pos, site_name="r_end_effector", tolerance=0.01, max_steps=100):
    """
    Numerical Inverse Kinematics using the Jacobian transpose/pseudo-inverse method.
    Maps a 3D target position to joint angles for the right arm.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_names = [
        "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", 
        "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll"
    ]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    qpos_indices = [model.jnt_qposadr[jid] for jid in joint_ids]
    dof_indices = [model.jnt_dofadr[jid] for jid in joint_ids]

    # Jacobian and step parameters
    step_size = 0.5
    damping = 0.01

    for _ in range(max_steps):
        # Forward kinematics to get current end-effector position
        mujoco.mj_fwdPosition(model, data)
        current_pos = data.site_xpos[site_id]
        
        # Error vector
        error = target_pos - current_pos
        if np.linalg.norm(error) < tolerance:
            return True, data.qpos[qpos_indices]

        # Calculate Jacobian
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, site_id)
        
        # Isolate arm joints Jacobian
        jac_arm = jac[:, dof_indices]
        
        # Solve for joint velocity: dq = J^T * (J * J^T + lambda*I)^-1 * error
        # Using pseudo-inverse for stability
        n = jac_arm.shape[0]
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + damping * np.eye(n)) @ error
        
        # Update joint positions
        data.qpos[qpos_indices] += dq * step_size
        
        # Clamp to joint limits
        for i, idx in enumerate(qpos_indices):
            limit = model.jnt_range[joint_ids[i]]
            data.qpos[idx] = np.clip(data.qpos[idx], limit[0], limit[1])

    return False, data.qpos[qpos_indices]

def control_gripper(data, open_amount):
    """
    Actuates the gripper. 0.0 is closed, 0.02 is open in our XML.
    """
    data.ctrl[7] = open_amount  # r_gripper_left
    data.ctrl[8] = open_amount  # r_gripper_right

def main():
    # Load the model
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize vision bridge
    vision_bridge = ReachyVisionBridge()
    
    # Table bounds for 2D->3D mapping (X, Y range in world coordinates)
    # Match the table pos (0.4, 0) and size (0.3, 0.5) in XML
    table_bounds = [0.1, 0.7, -0.5, 0.5]

    # Retrieval State Machine
    # 0: Idle/Tracking, 1: Moving to object, 2: Grasping, 3: Lifting
    state = 0
    target_3d = None
    
    print("Starting Reachy Mini Object Retrieval Simulation...")
    print("Use the 'Vision Bridge' to guide the arm.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # 1. Vision Input Simulation
            # In a real app, inject eye-tracking data here:
            # normalized_input = eye_tracker.get_gaze_point()
            # For demo, we just use a fixed target or dummy input
            if state == 0:
                # Simulate user looking at a specific point after 2 seconds
                if time.time() - start_time > 2.0:
                    vision_bridge.update_input(0.5, 0.5) # Look at center of table
                    state = 1
            
            # 2. Coordinate Mapping
            normalized_coords = vision_bridge.get_eye_cursor_coords()
            target_3d = map_2d_to_3d(normalized_coords, table_bounds)
            
            # 3. Task Execution
            if state == 1:
                # Move to Target
                success, target_qpos = solve_ik(model, data, target_3d)
                if success:
                    # Apply control to reach the solved joint angles
                    # Note: In this simple position-control setup, we map IK directly to actuators
                    data.ctrl[:7] = target_qpos
                    
                    # Check if close enough to start grasping
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_end_effector")
                    dist = np.linalg.norm(data.site_xpos[site_id] - target_3d)
                    if dist < 0.02:
                        state = 2
                        print("Target reached. Grasping...")
                else:
                    # Graceful error handling
                    print("Warning: Target position out of physical reach.")
            
            elif state == 2:
                # Close Gripper
                control_gripper(data, 0.0) # 0.0 is closed/tight
                # Wait a bit for grasp to secure
                if time.time() - start_time > 8.0:
                    state = 3
                    print("Object secured. Lifting...")
            
            elif state == 3:
                # Lift object
                lift_target = target_3d + np.array([0, 0, 0.2])
                success, target_qpos = solve_ik(model, data, lift_target)
                if success:
                    data.ctrl[:7] = target_qpos

            # --- EXTERNAL INJECTION POINT ---
            # Inject live camera feed processing or external tracking script logic here
            # e.g., camera_frame = cam.read(); tracking_data = process(camera_frame)
            
            # Step the physics
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Real-time synchronization
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()
