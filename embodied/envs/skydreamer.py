"""
DreamerQuadEnv: Vision-based drone racing environment for DreamerV3

Combines:
- Physics simulation from ~/Projects/drone/quad_race_env.py
- Optimized 64x64 binary segmentation rendering (no textures, no drone body)
- Embodied framework interface for DreamerV3 integration

Observation space:
    Dict({
        'image': 64x64 binary gate segmentation mask (uint8): 0 or 255
        'state': Full physics state vector (for future privileged learning)
        'reward': Scalar reward
        'is_first': Episode start indicator
        'is_last': Episode end indicator
        'is_terminal': Terminal state indicator
    })
"""

import sys
import os
import numpy as np
import mujoco
import elements
import embodied

# Add drone project to path for imports
sys.path.insert(0, os.path.expanduser('~/Projects/drone'))
from quad_race_env import Quadcopter3DGates
from randomization import skydreamer as skydreamer_randomization


class DreamerQuadEnv(embodied.Env):
    """
    Vision-based drone racing environment for DreamerV3.

    Features:
    - Fast 64x64 binary gate segmentation (no textures/materials)
    - Minimal rendering (camera only, no drone body)
    - Pre-generated static XML for A2RL track
    - Embodied framework interface
    """

    def __init__(self,
                 task='quad',  # Required by embodied framework
                 randomization=None,
                 max_steps=2400,
                 disable_max_steps=False,
                 gate_size=1.0,
                 camera_reward_weight=0.0,
                 seed=None,
                 **kwargs):
        """
        Initialize environment with physics and rendering.

        Args:
            task: Task name (required by embodied framework, ignored)
            randomization: Domain randomization function (default: skydreamer)
            max_steps: Maximum steps per episode
            disable_max_steps: Disable max steps limit
            gate_size: Gate size for collision detection (meters)
            camera_reward_weight: Weight for camera alignment reward
            seed: Random seed
        """
        super().__init__()

        # Use skydreamer randomization by default
        if randomization is None:
            randomization = skydreamer_randomization

        # A2RL track configuration (hardcoded for static XML)
        self.gates_pos = np.array([
            [12.5, 2, 1.25],
            [6.5, 6, 1.25],
            [5.5, 14, 1.25],
            [2.5, 24, 1.25],
            [7.5, 30, 1.25],
            [12.2, 22, 1.25],
            [17.5, 30, (1.25 + 2.7)],
            [17.5, 30, 1.25],
            [18.5, 22, 1.25],
            [20.5, 14, 1.25],
            [18.5, 6, (1.25 + 2.7)],
            [20, 4.5, 1.9],
            [18.5, 6, 1.25],
        ], dtype=np.float32)

        self.gate_yaw = np.array([180, 135, 120, 90, 350, 0, 80, 260, 280, 260, 225, 45, 225],
                                 dtype=np.float32) * np.pi / 180

        self.start_pos = self.gates_pos[0] + np.array([0, -1., 0])

        # Initialize physics environment (vectorized with num_envs=1)
        print("Initializing physics environment...")
        self.physics_env = Quadcopter3DGates(
            num_envs=1,
            randomization=randomization,
            gates_pos=self.gates_pos,
            gate_yaw=self.gate_yaw,
            start_pos=self.start_pos,
            gates_ahead=0,
            num_state_history=0,
            num_action_history=0,
            history_step_size=1,
            param_input=False,
            param_input_noise=0.0,
            max_steps=max_steps,
            disable_max_steps=disable_max_steps,
            gate_size=gate_size,
            track_x_size=25.0,
            track_y_size=35.0,
            track_z_size=7.0,
            camera_reward_weight=camera_reward_weight,
            camera_x_offset=-0.02,
            camera_y_offset=0.0,
            camera_z_offset=0.05,
            camera_pitch_deg=40.0,
            motor_smoothness_weight=0.01,
            seed=seed
        )

        # Load static MuJoCo model for rendering
        print("Loading MuJoCo model for rendering...")
        xml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'dreamer_quad_minimal.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        print("MuJoCo model loaded successfully")

        # Setup optimized segmentation renderer (64x64, no textures)
        self.renderer = mujoco.Renderer(self.mj_model, height=64, width=64)
        self.renderer.enable_segmentation_rendering()

        # Get FPV camera ID
        self.fpv_cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")

        # Find gate geom IDs for segmentation
        self.gate_geom_ids = set()
        for i in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and "gate_mesh" in geom_name:
                self.gate_geom_ids.add(i)
        print(f"Found {len(self.gate_geom_ids)} gate geoms for segmentation")

        # State tracking
        self._done = True
        self._state_dim = self.physics_env.observation_space.shape[0]

        print(f"\nDreamerQuadEnv initialized:")
        print(f"  Image obs: 64x64 binary segmentation (optimized)")
        print(f"  State obs: {self._state_dim}D")
        print(f"  Action space: 4D continuous")
        print(f"  Rendering: No textures, minimal geometry")

    @property
    def obs_space(self):
        """Define observation space using embodied framework format"""
        return {
            'image': elements.Space(np.uint8, (64, 64)),
            'state': elements.Space(np.float32, (self._state_dim,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        """Define action space using embodied framework format"""
        return {
            'action': elements.Space(np.float32, (4,), -1.0, 1.0),
            'reset': elements.Space(bool),
        }

    def _update_mujoco_state(self):
        """Update camera position/orientation in MuJoCo from physics state"""
        # Get drone state from physics env
        pos = self.physics_env.world_states[0, :3]  # x, y, z
        quat = self.physics_env.world_states[0, 6:10]  # qw, qx, qy, qz

        # Update MuJoCo state (freejoint qpos: [x, y, z, qw, qx, qy, qz])
        self.mj_data.qpos[0:3] = pos
        self.mj_data.qpos[3:7] = quat

        # Forward kinematics (fast, no dynamics)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _render_image(self):
        """Render 64x64 binary gate segmentation mask (optimized)"""
        # Update scene with current state
        self.renderer.update_scene(self.mj_data, camera=self.fpv_cam_id)

        # Render segmentation
        seg_pixels = self.renderer.render()

        # Extract segmentation IDs (subtract 1 to get geom IDs)
        seg_ids = seg_pixels[:, :, 0].astype(np.int32) - 1

        # Create binary gate mask
        gate_mask = np.isin(seg_ids, list(self.gate_geom_ids))

        # Convert to uint8: 255 for gates, 0 for background
        return gate_mask.astype(np.uint8) * 255

    def _obs(self, image, state, reward, is_first=False, is_last=False, is_terminal=False):
        """Format observation for embodied framework"""
        return {
            'image': image,
            'state': state,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def step(self, action):
        """Step environment and return observation"""
        # Handle reset
        if action['reset'] or self._done:
            self._done = False
            state = self.physics_env.reset()
            self._update_mujoco_state()
            image = self._render_image()
            return self._obs(image, state[0], 0.0, is_first=True)

        # Step physics
        action_array = action['action']
        action_batched = action_array[None] if action_array.ndim == 1 else action_array
        states, rewards, dones, infos = self.physics_env.step(action_batched)

        # Update MuJoCo visualization
        self._update_mujoco_state()

        # Render image
        image = self._render_image()

        # Update done state
        self._done = bool(dones[0])

        # Return observation
        return self._obs(
            image=image,
            state=states[0],
            reward=float(rewards[0]),
            is_last=self._done,
            is_terminal=self._done  # For now, treat all terminations as terminal
        )

    def render(self):
        """Render RGB image for visualization (optional)"""
        return self._render_image()

    def close(self):
        """Clean up resources"""
        self.physics_env.close()
        self.renderer.close()


# Convenience function (for backwards compatibility)
def make_dreamer_quad_env(**kwargs):
    """
    Create DreamerQuadEnv with default configuration.

    Usage:
        env = make_dreamer_quad_env()
        env = make_dreamer_quad_env(max_steps=3600, camera_reward_weight=0.1)
    """
    return DreamerQuadEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    print("\n" + "="*80)
    print("Testing DreamerQuadEnv...")
    print("="*80 + "\n")

    env = make_dreamer_quad_env(disable_max_steps=True)

    print(f"\nObservation space: {env.obs_space}")
    print(f"Action space: {env.act_space}")

    # Test reset
    obs = env.step({'reset': True})
    print(f"\nReset observation:")
    print(f"  Image shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
    print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
    print(f"  State shape: {obs['state'].shape}")
    print(f"  Gate pixels: {np.sum(obs['image'] > 0)} / {64*64}")
    print(f"  is_first: {obs['is_first']}")

    # Test a few steps
    print(f"\nRunning 10 test steps...")
    for i in range(10):
        action = {'action': np.random.uniform(-1, 1, size=4).astype(np.float32), 'reset': False}
        obs = env.step(action)
        gate_pixels = np.sum(obs['image'] > 0)
        print(f"Step {i+1}: reward={obs['reward']:.3f}, is_last={obs['is_last']}, gate_pixels={gate_pixels:4d}")

        if obs['is_last']:
            print("  Episode ended, resetting...")
            obs = env.step({'reset': True})

    env.close()
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
