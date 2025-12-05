"""
SkyDreamer: Vision-based drone racing environment for DreamerV3

Self-contained implementation with inlined physics simulation and rendering.
Uses SkyDreamer dynamics model with domain randomization.

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

import os
import numpy as np
from numba import njit
import elements
import embodied
from gymnasium import spaces

# Configure MuJoCo for headless rendering (must be set before importing mujoco)
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import mujoco


# ============================================================================
# SkyDreamer Domain Randomization
# ============================================================================

# SkyDreamer parameters from Table II (page 6)
SKYDREAMER_PARAMS = {
    'k_w': 1.55e-06,
    'k_x': 5.37e-05,
    'k_y': 5.37e-05,
    'k_x2': 4.10e-03,
    'k_y2': 1.51e-02,
    'k_angle': 3.145,
    'k_hor': 7.245,
    'k_v2': 0.00,
    'k_p1': 4.99e-05, 'k_p2': 3.78e-05, 'k_p3': 4.82e-05, 'k_p4': 3.83e-05,
    'k_q1': 2.05e-05, 'k_q2': 2.46e-05, 'k_q3': 2.02e-05, 'k_q4': 2.57e-05,
    'k_r1': 3.38e-03, 'k_r2': 3.38e-03, 'k_r3': 3.38e-03, 'k_r4': 3.38e-03,
    'k_r5': 3.24e-04, 'k_r6': 3.24e-04, 'k_r7': 3.24e-04, 'k_r8': 3.24e-04,
    'J_x': -0.89,
    'J_y': 0.96,
    'J_z': -0.34,
    'w_min': 341.75,
    'w_max': 3100.00,
    'k': 0.50,
    'tau': 0.03,
    'r_prop': 0.0635
}


def skydreamer_randomization(num):
    """
    SkyDreamer domain randomization from Table III (page 9).
    Motor limits: ±20%, all other parameters: ±30%.
    """
    w_min = np.random.uniform(0.8, 1.2, size=num) * SKYDREAMER_PARAMS['w_min']
    w_max = np.random.uniform(0.8, 1.2, size=num) * SKYDREAMER_PARAMS['w_max']

    k = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k']
    k = np.clip(k, 0.0, 1.0)
    tau = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['tau']

    k_w = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_w']
    k_x = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_x']
    k_y = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_y']
    k_x2 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_x2']
    k_y2 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_y2']

    k_angle = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_angle']
    k_hor = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_hor']
    k_v2 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_v2']

    k_p1 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_p1']
    k_p2 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_p2']
    k_p3 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_p3']
    k_p4 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_p4']

    k_q1 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_q1']
    k_q2 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_q2']
    k_q3 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_q3']
    k_q4 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_q4']

    k_r1 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r1']
    k_r2 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r2']
    k_r3 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r3']
    k_r4 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r4']

    k_r5 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r5']
    k_r6 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r6']
    k_r7 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r7']
    k_r8 = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['k_r8']

    J_x = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['J_x']
    J_y = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['J_y']
    J_z = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['J_z']

    r_prop = np.random.uniform(0.7, 1.3, size=num) * SKYDREAMER_PARAMS['r_prop']

    return {
        'k_w': k_w, 'k_x': k_x, 'k_y': k_y,
        'k_x2': k_x2, 'k_y2': k_y2,
        'k_angle': k_angle, 'k_hor': k_hor, 'k_v2': k_v2,
        'k_p1': k_p1, 'k_p2': k_p2, 'k_p3': k_p3, 'k_p4': k_p4,
        'k_q1': k_q1, 'k_q2': k_q2, 'k_q3': k_q3, 'k_q4': k_q4,
        'k_r1': k_r1, 'k_r2': k_r2, 'k_r3': k_r3, 'k_r4': k_r4,
        'k_r5': k_r5, 'k_r6': k_r6, 'k_r7': k_r7, 'k_r8': k_r8,
        'J_x': J_x, 'J_y': J_y, 'J_z': J_z,
        'tau': tau, 'k': k,
        'w_min': w_min, 'w_max': w_max,
        'r_prop': r_prop
    }


# ============================================================================
# Quaternion Helper Functions (Numba JIT)
# ============================================================================

@njit(fastmath=True, cache=True)
def quat_normalize(q):
    """Normalize a quaternion [qw, qx, qy, qz]"""
    norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


@njit(fastmath=True, cache=True)
def normalize_quaternions_batch(states):
    """Vectorized quaternion normalization for all environments in-place"""
    num_envs = states.shape[0]
    for i in range(num_envs):
        norm = np.sqrt(states[i,6]**2 + states[i,7]**2 + states[i,8]**2 + states[i,9]**2)
        if norm < 1e-10:
            states[i, 6] = 1.0
            states[i, 7:10] = 0.0
        else:
            inv_norm = 1.0 / norm
            states[i, 6:10] *= inv_norm
    return states


@njit(fastmath=True, cache=True)
def quat_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix"""
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    R00 = 1 - 2*(qy**2 + qz**2)
    R01 = 2*(qx*qy - qw*qz)
    R02 = 2*(qx*qz + qw*qy)
    R10 = 2*(qx*qy + qw*qz)
    R11 = 1 - 2*(qx**2 + qz**2)
    R12 = 2*(qy*qz - qw*qx)
    R20 = 2*(qx*qz - qw*qy)
    R21 = 2*(qy*qz + qw*qx)
    R22 = 1 - 2*(qx**2 + qy**2)

    return R00, R01, R02, R10, R11, R12, R20, R21, R22


@njit(fastmath=True, cache=True)
def euler_to_quat(phi, theta, psi):
    """Convert Euler angles (roll, pitch, yaw) to quaternion [qw, qx, qy, qz]"""
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz], dtype=np.float32)


@njit(fastmath=True, cache=True)
def quat_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z], dtype=np.float32)


# ============================================================================
# Quadcopter Dynamics (Numba JIT)
# ============================================================================

@njit(fastmath=True, cache=True)
def compute_dynamics_jit(states, actions, params):
    """
    Vectorized JIT-compiled quadcopter dynamics with quaternions.
    Uses ROS convention: ENU (East-North-Up)
    Implements SkyDreamer dynamics model with aerodynamic corrections.

    Args:
        states: (num_envs, 17) - [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r, w1, w2, w3, w4]
        actions: (num_envs, 4) - [u1, u2, u3, u4] normalized motor commands in [-1, 1]
        params: (num_envs, 32) - physics parameters

    Returns:
        state_derivatives: (num_envs, 17)
    """
    num_envs = states.shape[0]
    derivatives = np.zeros((num_envs, 17), dtype=np.float32)

    g = 9.81
    w_min_n = 0.0
    w_max_n = 3000.0

    for i in range(num_envs):
        x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r, w1, w2, w3, w4 = states[i]
        u1, u2, u3, u4 = actions[i]

        k_x, k_y, k_w = params[i, 0], params[i, 1], params[i, 2]
        k_x2, k_y2 = params[i, 3], params[i, 4]
        k_angle, k_hor, k_v2 = params[i, 5], params[i, 6], params[i, 7]
        k_p1, k_p2, k_p3, k_p4 = params[i, 8], params[i, 9], params[i, 10], params[i, 11]
        k_q1, k_q2, k_q3, k_q4 = params[i, 12], params[i, 13], params[i, 14], params[i, 15]
        k_r1, k_r2, k_r3, k_r4 = params[i, 16], params[i, 17], params[i, 18], params[i, 19]
        k_r5, k_r6, k_r7, k_r8 = params[i, 20], params[i, 21], params[i, 22], params[i, 23]
        J_x, J_y, J_z = params[i, 24], params[i, 25], params[i, 26]
        tau, k, w_min, w_max = params[i, 27], params[i, 28], params[i, 29], params[i, 30]
        r_prop = params[i, 31]

        W1 = (w1 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
        W2 = (w2 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
        W3 = (w3 + 1) / 2 * (w_max_n - w_min_n) + w_min_n
        W4 = (w4 + 1) / 2 * (w_max_n - w_min_n) + w_min_n

        U1 = (u1 + 1) / 2
        U2 = (u2 + 1) / 2
        U3 = (u3 + 1) / 2
        U4 = (u4 + 1) / 2

        Wc1 = (w_max - w_min) * np.sqrt(k * U1**2 + (1 - k) * U1) + w_min
        Wc2 = (w_max - w_min) * np.sqrt(k * U2**2 + (1 - k) * U2) + w_min
        Wc3 = (w_max - w_min) * np.sqrt(k * U3**2 + (1 - k) * U3) + w_min
        Wc4 = (w_max - w_min) * np.sqrt(k * U4**2 + (1 - k) * U4) + w_min

        d_W1 = (Wc1 - W1) / tau
        d_W2 = (Wc2 - W2) / tau
        d_W3 = (Wc3 - W3) / tau
        d_W4 = (Wc4 - W4) / tau

        quat = np.array([qw, qx, qy, qz], dtype=np.float32)
        R00, R01, R02, R10, R11, R12, R20, R21, R22 = quat_to_rotation_matrix(quat)

        vbx = R00 * vx + R10 * vy + R20 * vz
        vby = R01 * vx + R11 * vy + R21 * vz
        vbz = R02 * vx + R12 * vy + R22 * vz

        w_bar = (W1 + W2 + W3 + W4) / 4.0
        w_sum = W1 + W2 + W3 + W4
        w_sum_sq = W1**2 + W2**2 + W3**2 + W4**2

        alpha = np.arctan2(vbz, r_prop * w_bar + 1e-6)
        v_hor = np.sqrt(vbx**2 + vby**2)
        mu_hor = np.arctan2(v_hor, r_prop * w_bar + 1e-6)

        Dx = -k_x * vbx * w_sum - k_x2 * vbx * abs(vbx)
        Dy = -k_y * vby * w_sum - k_y2 * vby * abs(vby)
        T = k_w * (1.0 + k_angle * alpha + k_hor * mu_hor) * w_sum_sq - k_v2 * vbz * abs(vbz)

        Mx = -k_p1 * W1**2 - k_p2 * W2**2 + k_p3 * W3**2 + k_p4 * W4**2 + J_x * q * r
        My = -k_q1 * W1**2 + k_q2 * W2**2 - k_q3 * W3**2 + k_q4 * W4**2 + J_y * p * r
        Mz = (-k_r1 * W1 + k_r2 * W2 + k_r3 * W3 - k_r4 * W4
              - k_r5 * d_W1 + k_r6 * d_W2 + k_r7 * d_W3 - k_r8 * d_W4 + J_z * p * q)

        derivatives[i, 0] = vx
        derivatives[i, 1] = vy
        derivatives[i, 2] = vz

        derivatives[i, 3] = R00 * Dx + R01 * Dy + R02 * T
        derivatives[i, 4] = R10 * Dx + R11 * Dy + R12 * T
        derivatives[i, 5] = R20 * Dx + R21 * Dy + R22 * T - g

        omega_quat = np.array([0.0, p, q, r], dtype=np.float32)
        dq = quat_multiply(quat, omega_quat) * 0.5
        derivatives[i, 6] = dq[0]
        derivatives[i, 7] = dq[1]
        derivatives[i, 8] = dq[2]
        derivatives[i, 9] = dq[3]

        derivatives[i, 10] = Mx
        derivatives[i, 11] = My
        derivatives[i, 12] = Mz

        derivatives[i, 13] = d_W1 / (w_max_n - w_min_n) * 2
        derivatives[i, 14] = d_W2 / (w_max_n - w_min_n) * 2
        derivatives[i, 15] = d_W3 / (w_max_n - w_min_n) * 2
        derivatives[i, 16] = d_W4 / (w_max_n - w_min_n) * 2

    return derivatives


@njit(fastmath=True, cache=True)
def compute_step_logic_jit(pos_old, pos_new, new_states, actions, prev_actions,
                           gate_pos, gate_yaw, step_counts, final_gate_passed,
                           max_steps, disable_max_steps, gate_size, x_min, x_max, y_min, y_max, z_min, z_max,
                           camera_offset, camera_pitch, camera_reward_weight, motor_smoothness_weight):
    """
    JIT-compiled reward and collision detection logic.
    Returns: rewards, gate_passed, gate_collision, ground_collision, out_of_bounds, max_steps_reached
    """
    num_envs = pos_old.shape[0]
    rewards = np.zeros(num_envs, dtype=np.float32)
    gate_passed = np.zeros(num_envs, dtype=np.bool_)
    gate_collision = np.zeros(num_envs, dtype=np.bool_)
    ground_collision = np.zeros(num_envs, dtype=np.bool_)
    out_of_bounds = np.zeros(num_envs, dtype=np.bool_)
    max_steps_reached = np.zeros(num_envs, dtype=np.bool_)

    for i in range(num_envs):
        d2g_old = np.sqrt((pos_old[i, 0] - gate_pos[i, 0])**2 +
                          (pos_old[i, 1] - gate_pos[i, 1])**2 +
                          (pos_old[i, 2] - gate_pos[i, 2])**2)
        d2g_new = np.sqrt((pos_new[i, 0] - gate_pos[i, 0])**2 +
                          (pos_new[i, 1] - gate_pos[i, 1])**2 +
                          (pos_new[i, 2] - gate_pos[i, 2])**2)

        rate_l1_norm = abs(new_states[i, 10]) + abs(new_states[i, 11]) + abs(new_states[i, 12])
        rat_penalty = (1.0 / (2.0 * (1.0/0.01) * 1e5)) * (np.exp(min(rate_l1_norm, 17.0)) - 1.0)

        qw, qx, qy, qz = new_states[i, 6], new_states[i, 7], new_states[i, 8], new_states[i, 9]
        R00, R01, R02, R10, R11, R12, R20, R21, R22 = quat_to_rotation_matrix(
            np.array([qw, qx, qy, qz], dtype=np.float32)
        )

        cam_axis_body_x = np.cos(camera_pitch)
        cam_axis_body_y = 0.0
        cam_axis_body_z = np.sin(camera_pitch)

        cam_offset_world_x = R00 * camera_offset[0] + R01 * camera_offset[1] + R02 * camera_offset[2]
        cam_offset_world_y = R10 * camera_offset[0] + R11 * camera_offset[1] + R12 * camera_offset[2]
        cam_offset_world_z = R20 * camera_offset[0] + R21 * camera_offset[1] + R22 * camera_offset[2]

        cam_x = new_states[i, 0] + cam_offset_world_x
        cam_y = new_states[i, 1] + cam_offset_world_y
        cam_z = new_states[i, 2] + cam_offset_world_z

        cam_axis_world_x = R00 * cam_axis_body_x + R01 * cam_axis_body_y + R02 * cam_axis_body_z
        cam_axis_world_y = R10 * cam_axis_body_x + R11 * cam_axis_body_y + R12 * cam_axis_body_z
        cam_axis_world_z = R20 * cam_axis_body_x + R21 * cam_axis_body_y + R22 * cam_axis_body_z

        gate_dir_x = gate_pos[i, 0] - cam_x
        gate_dir_y = gate_pos[i, 1] - cam_y
        gate_dir_z = gate_pos[i, 2] - cam_z

        gate_dir_norm = np.sqrt(gate_dir_x**2 + gate_dir_y**2 + gate_dir_z**2)
        camera_alignment_reward = 0.0
        if gate_dir_norm > 1e-6:
            gate_dir_x /= gate_dir_norm
            gate_dir_y /= gate_dir_norm
            gate_dir_z /= gate_dir_norm

            alignment = (cam_axis_world_x * gate_dir_x +
                        cam_axis_world_y * gate_dir_y +
                        cam_axis_world_z * gate_dir_z)
            camera_alignment_reward = camera_reward_weight * alignment

        prog_reward = d2g_old - d2g_new

        normal_x = np.cos(gate_yaw[i])
        normal_y = np.sin(gate_yaw[i])

        pos_old_projected = (pos_old[i, 0] - gate_pos[i, 0]) * normal_x + (pos_old[i, 1] - gate_pos[i, 1]) * normal_y
        pos_new_projected = (pos_new[i, 0] - gate_pos[i, 0]) * normal_x + (pos_new[i, 1] - gate_pos[i, 1]) * normal_y

        passed_gate_plane = (pos_old_projected < 0) and (pos_new_projected > 0)

        gate_reward = 0.0
        if passed_gate_plane:
            dy = abs(pos_new[i, 1] - gate_pos[i, 1])
            dz = abs(pos_new[i, 2] - gate_pos[i, 2])
            max_offset = max(dy, dz)

            if max_offset < gate_size / 2:
                gate_passed[i] = True
                gate_reward = 1.0 - max_offset / (gate_size / 2)
            else:
                gate_collision[i] = True

        if prog_reward > 0:
            rewards[i] = 5.0 * prog_reward - rat_penalty + 30.0 * gate_reward + camera_alignment_reward
        else:
            rewards[i] = 5.0 * prog_reward - rat_penalty + 30.0 * gate_reward

        if new_states[i, 2] < z_min:
            ground_collision[i] = True
            rewards[i] = 0.0

        if new_states[i, 0] < x_min or new_states[i, 0] > x_max:
            out_of_bounds[i] = True
            rewards[i] = 0.0
        if new_states[i, 1] < y_min or new_states[i, 1] > y_max:
            out_of_bounds[i] = True
            rewards[i] = 0.0
        if new_states[i, 2] > z_max:
            out_of_bounds[i] = True
            rewards[i] = 0.0
        if (abs(new_states[i, 10]) > 1000 or abs(new_states[i, 11]) > 1000 or abs(new_states[i, 12]) > 1000):
            out_of_bounds[i] = True
            rewards[i] = 0.0

        if not disable_max_steps and step_counts[i] >= max_steps:
            max_steps_reached[i] = True

        if final_gate_passed[i]:
            rewards[i] = 10.0

    return rewards, gate_passed, gate_collision, ground_collision, out_of_bounds, max_steps_reached


# ============================================================================
# Quadcopter Physics Environment
# ============================================================================

class Quadcopter3DGates:
    """Vectorized quadcopter racing environment with quaternion dynamics"""

    def __init__(self,
                 num_envs,
                 randomization,
                 gates_pos,
                 gate_yaw,
                 start_pos,
                 max_steps=2400,
                 disable_max_steps=False,
                 gate_size=1.0,
                 track_x_size=25.0,
                 track_y_size=35.0,
                 track_z_size=7.0,
                 camera_reward_weight=0.0,
                 camera_x_offset=-0.02,
                 camera_y_offset=0.0,
                 camera_z_offset=0.05,
                 camera_pitch_deg=40.0,
                 motor_smoothness_weight=0.01,
                 seed=None,
                 **kwargs):

        if seed is not None:
            np.random.seed(seed)

        self.num_envs = num_envs
        self.camera_reward_weight = camera_reward_weight
        self.camera_offset = np.array([camera_x_offset, camera_y_offset, camera_z_offset], dtype=np.float32)
        self.camera_pitch = np.radians(camera_pitch_deg)
        self.motor_smoothness_weight = motor_smoothness_weight

        self.track_x_size = track_x_size
        self.track_y_size = track_y_size
        self.track_z_size = track_z_size
        self.x_min = 0.0
        self.x_max = track_x_size
        self.y_min = 0.0
        self.y_max = track_y_size
        self.z_min = 0.0
        self.z_max = track_z_size

        self.gate_size = gate_size
        self.start_pos = start_pos.astype(np.float32)
        self.gate_pos = gates_pos.astype(np.float32)
        self.gate_yaw = gate_yaw.astype(np.float32)
        self.num_gates = gates_pos.shape[0]

        # Domain randomization
        def rand_f(n):
            param_dict = randomization(n)
            param_keys = ['k_x', 'k_y', 'k_w', 'k_x2', 'k_y2', 'k_angle', 'k_hor', 'k_v2',
                         'k_p1', 'k_p2', 'k_p3', 'k_p4', 'k_q1', 'k_q2', 'k_q3', 'k_q4',
                         'k_r1', 'k_r2', 'k_r3', 'k_r4', 'k_r5', 'k_r6', 'k_r7', 'k_r8',
                         'J_x', 'J_y', 'J_z', 'tau', 'k', 'w_min', 'w_max', 'r_prop']
            return np.array([param_dict[k] for k in param_keys]).T

        self.randomization = rand_f
        self.params = self.randomization(num_envs)

        self.target_gates = np.zeros(num_envs, dtype=int)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*17),
            high=np.array([np.inf]*17)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))

        self.world_states = np.zeros((num_envs,17), dtype=np.float32)
        self.world_states[:, 6] = 1.0  # Identity quaternion

        self.max_steps = max_steps
        self.disable_max_steps = disable_max_steps
        self.dt = np.float32(0.01)

        self.step_counts = np.zeros(num_envs, dtype=int)
        self.actions = np.zeros((num_envs,4), dtype=np.float32)
        self.prev_actions = np.zeros((num_envs,4), dtype=np.float32)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.final_gate_passed = np.zeros(num_envs, dtype=bool)

    def reset_(self, dones):
        num_reset = dones.sum()

        # Random gate initialization: uniformly sample from all gates (0 to num_gates-1)
        self.target_gates[dones] = np.random.randint(0, self.num_gates, size=num_reset)

        x0 = np.full(num_reset, self.start_pos[0])
        y0 = np.full(num_reset, self.start_pos[1])
        z0 = np.full(num_reset, self.start_pos[2])

        vx0 = np.random.uniform(-0.1, 0.1, size=(num_reset,))
        vy0 = np.random.uniform(-0.1, 0.1, size=(num_reset,))
        vz0 = np.random.uniform(-0.1, 0.1, size=(num_reset,))

        gate_yaw = self.gate_yaw[0]
        phi0 = np.random.uniform(-np.pi/36, np.pi/36, size=(num_reset,))
        theta0 = np.random.uniform(-np.pi/36, np.pi/36, size=(num_reset,))
        psi0 = gate_yaw + np.random.uniform(-np.pi/18, np.pi/18, size=(num_reset,))

        p0 = np.random.uniform(-0.05, 0.05, size=(num_reset,))
        q0 = np.random.uniform(-0.05, 0.05, size=(num_reset,))
        r0 = np.random.uniform(-0.05, 0.05, size=(num_reset,))

        w10 = np.random.uniform(-0.2, 0.2, size=(num_reset,))
        w20 = np.random.uniform(-0.2, 0.2, size=(num_reset,))
        w30 = np.random.uniform(-0.2, 0.2, size=(num_reset,))
        w40 = np.random.uniform(-0.2, 0.2, size=(num_reset,))

        quaternions = np.zeros((num_reset, 4), dtype=np.float32)
        for idx in range(num_reset):
            quaternions[idx] = euler_to_quat(phi0[idx], theta0[idx], psi0[idx])

        self.world_states[dones] = np.stack([x0, y0, z0, vx0, vy0, vz0,
                                              quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3],
                                              p0, q0, r0, w10, w20, w30, w40], axis=1)

        self.step_counts[dones] = np.zeros(num_reset)
        self.params[dones] = self.randomization(num_reset)

        return self.world_states[dones if num_reset < self.num_envs else slice(None)]

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step(self, actions):
        self.prev_actions = self.actions
        self.actions = actions

        new_states = self.world_states + self.dt * compute_dynamics_jit(self.world_states, self.actions, self.params)
        new_states = normalize_quaternions_batch(new_states)

        self.step_counts += 1

        pos_old = self.world_states[:,0:3]
        pos_new = new_states[:,0:3]
        pos_gate = self.gate_pos[self.target_gates%self.num_gates]
        yaw_gate = self.gate_yaw[self.target_gates%self.num_gates]

        rewards, gate_passed, gate_collision, ground_collision, out_of_bounds, max_steps_reached = \
            compute_step_logic_jit(
                pos_old, pos_new, new_states, self.actions, self.prev_actions,
                pos_gate, yaw_gate, self.step_counts, self.final_gate_passed,
                self.max_steps, self.disable_max_steps, self.gate_size,
                self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max,
                self.camera_offset, self.camera_pitch, self.camera_reward_weight,
                self.motor_smoothness_weight
            )

        self.target_gates[gate_passed] += 1
        self.target_gates[gate_passed] %= self.num_gates

        rewards[self.final_gate_passed] = 10

        dones = max_steps_reached | ground_collision | out_of_bounds | gate_collision
        self.dones = dones

        self.world_states = new_states
        self.reset_(dones)

        infos = [{}] * self.num_envs
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self.world_states[i]
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True
            infos[i]["ground_collision"] = ground_collision[i]
            infos[i]["out_of_bounds"] = out_of_bounds[i]
            infos[i]["gate_collision"] = gate_collision[i]
            infos[i]["gate_passed"] = gate_passed[i]

        return self.world_states, rewards, dones, infos

    def close(self):
        pass


# ============================================================================
# DreamerV3 Environment Wrapper
# ============================================================================

class DreamerQuadEnv(embodied.Env):
    """Vision-based drone racing environment for DreamerV3"""

    def __init__(self,
                 task='quad',
                 randomization=None,
                 max_steps=2400,
                 disable_max_steps=False,
                 gate_size=1.0,
                 camera_reward_weight=0.0,
                 seed=None,
                 **kwargs):

        super().__init__()

        if randomization is None:
            randomization = skydreamer_randomization

        # A2RL track configuration
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

        # Initialize physics environment
        self.physics_env = Quadcopter3DGates(
            num_envs=1,
            randomization=randomization,
            gates_pos=self.gates_pos,
            gate_yaw=self.gate_yaw,
            start_pos=self.start_pos,
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

        # Load MuJoCo model for rendering (EGL configured at module level for headless)
        xml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'dreamer_quad_minimal.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.renderer = mujoco.Renderer(self.mj_model, height=64, width=64)
        self.renderer.enable_segmentation_rendering()

        self.fpv_cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")

        self.gate_geom_ids = set()
        for i in range(self.mj_model.ngeom):
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and "gate_mesh" in geom_name:
                self.gate_geom_ids.add(i)

        self._done = True
        self._state_dim = self.physics_env.observation_space.shape[0]

    @property
    def obs_space(self):
        return {
            'image': elements.Space(np.uint8, (64, 64, 1)),
            'state': elements.Space(np.float32, (self._state_dim,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.float32, (4,), -1.0, 1.0),
            'reset': elements.Space(bool),
        }

    def _update_mujoco_state(self):
        pos = self.physics_env.world_states[0, :3]
        quat = self.physics_env.world_states[0, 6:10]
        self.mj_data.qpos[0:3] = pos
        self.mj_data.qpos[3:7] = quat
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _render_image(self):
        self.renderer.update_scene(self.mj_data, camera=self.fpv_cam_id)
        seg_pixels = self.renderer.render()
        seg_ids = seg_pixels[:, :, 0].astype(np.int32) - 1
        gate_mask = np.isin(seg_ids, list(self.gate_geom_ids))
        # Add channel dimension for DreamerV3 (64, 64) -> (64, 64, 1)
        return (gate_mask.astype(np.uint8) * 255)[:, :, None]

    def _obs(self, image, state, reward, is_first=False, is_last=False, is_terminal=False):
        return {
            'image': image,
            'state': state,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            state = self.physics_env.reset()
            self._update_mujoco_state()
            image = self._render_image()
            return self._obs(image, state[0], 0.0, is_first=True)

        action_array = action['action']
        action_batched = action_array[None] if action_array.ndim == 1 else action_array
        states, rewards, dones, infos = self.physics_env.step(action_batched)

        self._update_mujoco_state()
        image = self._render_image()

        self._done = bool(dones[0])

        return self._obs(
            image=image,
            state=states[0],
            reward=float(rewards[0]),
            is_last=self._done,
            is_terminal=self._done
        )

    def render(self):
        return self._render_image()

    def close(self):
        self.physics_env.close()
        self.renderer.close()
