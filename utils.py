import torch

@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_from_angle_axis(angle_axis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert axis-angle rotation vectors to quaternions.

    Args:
        angle_axis: Rotation vectors of shape (..., 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).
    """
    angle = torch.norm(angle_axis, p=2, dim=-1, keepdim=True)
    axis = angle_axis / angle.clamp(min=eps)
    theta = angle / 2
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

@torch.jit.script
def quat_box_minus(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """The box-minus operator (quaternion difference) between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (N, 4).
        q2: The second quaternion in (w, x, y, z). Shape is (N, 4).

    Returns:
        The difference between the two quaternions. Shape is (N, 3).

    Reference:
        https://github.com/ANYbotics/kindr/blob/master/doc/cheatsheet/cheatsheet_latest.pdf
    """
    quat_diff = quat_mul(q1, quat_conjugate(q2))  # q1 * q2^-1
    return axis_angle_from_quat(quat_diff)  # log(qd)

@torch.jit.script
def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Computes the rotation difference between two quaternions.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Angular error between input quaternions in radians.
    """
    axis_angle_error = quat_box_minus(q1, q2)
    return torch.norm(axis_angle_error, dim=-1)


def prepare_original_motion(
    batch: dict,
    body_model,
    device: torch.device,
    max_body_joints: int = 22,
):
    """
    Prepares original motion data with original betas and scale=1.
    
    Args:
        batch: Dictionary containing 'poses', 'trans', 'betas'
        body_model: SMPL body model
        device: torch device
        max_body_joints: Maximum number of body joints to use
    
    Returns:
        Dictionary with:
            - 'betas': original betas [B, 10]
            - 'scale': original scale [B, 1] (always 1.0)
            - 'global_orient': original beta's global_orient [B, T, 4]
            - 'joints': original beta's joints [B, T, 24, 3]
    """
    poses = batch["poses"].to(device)  # [B, T, 156]
    trans = batch["trans"].to(device)  # [B, T, 3]
    betas = batch["betas"].to(device)  # [B, 10]

    B, T, _ = poses.shape
    poses = poses.view(B, T, -1, 3)[:, :, :max_body_joints]

    global_orient = poses[:, :, 0]
    body_pose = poses[:, :, 1:]
    hand_pose = torch.zeros(B, T, 2, 3, device=device)
    body_pose = torch.cat([body_pose, hand_pose], dim=2)

    smpl_body_pose = body_pose.reshape(B * T, -1)
    smpl_global_orient = global_orient.reshape(B * T, 3)
    smpl_trans = trans.reshape(B * T, 3)
    
    # Prepare original betas (scale = 1.0)
    smpl_betas_original = betas[:, None, :].expand(-1, T, -1).reshape(B * T, -1)
    scale_original = torch.ones(B, 1, device=device)  # [B, 1]

    with torch.no_grad():
        smpl_output_original = body_model(
            betas=smpl_betas_original,
            body_pose=smpl_body_pose,
            global_orient=smpl_global_orient,
            transl=smpl_trans,
        )

    # Process original beta outputs (for encoder input, scale=1)
    vertices_original = smpl_output_original.vertices.reshape(B, T, -1, 3)
    joints_original = smpl_output_original.joints[:, :24, :].reshape(B, T, 24, 3)
    height_offset_original = vertices_original[..., 2].min()
    joints_original[..., 2] -= height_offset_original

    global_orient_quat_original = quat_from_angle_axis(
        smpl_output_original.global_orient.reshape(B, T, 3)
    )

    return {
        "betas": betas,
        "scale": scale_original,
        "global_orient": global_orient_quat_original,
        "joints": joints_original,
        # Keep intermediate data for reuse
        "_smpl_body_pose": smpl_body_pose,
        "_smpl_global_orient": smpl_global_orient,
        "_smpl_trans": smpl_trans,
        "_B": B,
        "_T": T,
    }


def apply_beta_augmentation(
    betas: torch.Tensor,
    smpl_body_pose: torch.Tensor,
    smpl_global_orient: torch.Tensor,
    smpl_trans: torch.Tensor,
    body_model,
    beta_augment_std: float = 0.1,
):
    """
    Applies beta augmentation to motion data.
    
    Args:
        betas: Original betas [B, 10]
        smpl_body_pose: Body pose [B*T, ...]
        smpl_global_orient: Global orientation [B*T, 3]
        smpl_trans: Translation [B*T, 3]
        body_model: SMPL body model
        beta_augment_std: Standard deviation for beta augmentation (Gaussian noise)
    
    Returns:
        Dictionary with:
            - 'betas_augmented': augmented betas [B, 10]
            - 'vertices': augmented beta's vertices [B, T, V, 3]
            - 'joints': augmented beta's joints [B, T, 24, 3]
            - 'global_orient': augmented beta's global_orient [B, T, 3] (axis-angle)
    """
    B = betas.shape[0]
    T = smpl_body_pose.shape[0] // B
    
    # Augment betas with Gaussian noise
    betas_augmented = betas + torch.randn_like(betas) * beta_augment_std
    smpl_betas_augmented = betas_augmented[:, None, :].expand(-1, T, -1).reshape(B * T, -1)

    with torch.no_grad():
        smpl_output_augmented = body_model(
            betas=smpl_betas_augmented,
            body_pose=smpl_body_pose,
            global_orient=smpl_global_orient,
            transl=smpl_trans,
        )

    vertices_augmented = smpl_output_augmented.vertices.reshape(B, T, -1, 3)
    joints_augmented = smpl_output_augmented.joints[:, :24, :].reshape(B, T, 24, 3)
    global_orient_augmented = smpl_output_augmented.global_orient.reshape(B, T, 3)

    return {
        "betas_augmented": betas_augmented,
        "vertices": vertices_augmented,
        "joints": joints_augmented,
        "global_orient": global_orient_augmented,
    }


def apply_scale_augmentation(
    vertices: torch.Tensor,
    joints: torch.Tensor,
    global_orient: torch.Tensor,
    scale_augmented: torch.Tensor,
):
    """
    Applies scale augmentation to motion data in root coordinate system.
    
    Args:
        vertices: Vertices [B, T, V, 3]
        joints: Joints [B, T, 24, 3]
        global_orient: Global orientation in axis-angle [B, T, 3]
        scale_augmented: Augmented scale [B, 1]
    
    Returns:
        Dictionary with:
            - 'joints': scaled joints [B, T, 24, 3]
            - 'global_orient': global_orient quaternion [B, T, 4] (not affected by scale)
    """
    B, T = joints.shape[:2]
    
    # Apply scale augmentation in root coordinate system
    # Root joint is typically the first joint (index 0)
    root_positions = joints[:, :, 0, :]  # [B, T, 3] - root joint positions
    
    # Convert to root coordinate system
    vertices_local = vertices - root_positions[:, :, None, :]  # [B, T, V, 3]
    joints_local = joints - root_positions[:, :, None, :]  # [B, T, 24, 3]
    
    # Apply scale in root coordinate system
    scale_augmented_expanded = scale_augmented.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 1]
    vertices_local_scaled = vertices_local * scale_augmented_expanded
    joints_local_scaled = joints_local * scale_augmented_expanded
    
    # Convert back to world coordinate system
    vertices_scaled = vertices_local_scaled + root_positions[:, :, None, :]
    joints_scaled = joints_local_scaled + root_positions[:, :, None, :]
    
    # Adjust height offset to maintain foot-ground contact
    height_offset = vertices_scaled[..., 2].min()
    vertices_scaled[..., 2] -= height_offset
    joints_scaled[..., 2] -= height_offset

    # Global orient is not affected by scale (rotation is scale-invariant)
    global_orient_quat = quat_from_angle_axis(global_orient)

    return {
        "joints": joints_scaled,
        "global_orient": global_orient_quat,
    }


def prepare_motion_batch(
    batch: dict,
    body_model,
    device: torch.device,
    max_body_joints: int = 22,
    beta_augment_std: float = 0.1,
    scale_augment_std: float = 0.1,
):
    """
    Converts raw AMASS data into betas, global orientations (quaternions),
    and joints suitable for the AutoEncoder. Heavy SMPL computations stay
    outside the dataset and run here on the chosen device.
    
    Now supports beta and scale augmentation:
    - Encoder input: original beta's global_orient and joints (scale=1)
    - Decoder modulation: augmented beta + augmented scale
    - Loss target: augmented beta + augmented scale's global_orient and joints
    
    Args:
        batch: Dictionary containing 'poses', 'trans', 'betas'
        body_model: SMPL body model
        device: torch device
        max_body_joints: Maximum number of body joints to use
        beta_augment_std: Standard deviation for beta augmentation (Gaussian noise)
        scale_augment_std: Standard deviation for scale augmentation (Gaussian noise, applied to log scale)
    
    Returns:
        Dictionary with:
            - 'betas': original betas [B, 10] (for encoder input context, not used directly)
            - 'betas_augmented': augmented betas [B, 10] (for decoder modulation)
            - 'scale': original scale [B, 1] (always 1.0, for encoder input context)
            - 'scale_augmented': augmented scale [B, 1] (for decoder modulation)
            - 'global_orient': original beta's global_orient [B, T, 4] (for encoder input, scale=1)
            - 'joints': original beta's joints [B, T, 24, 3] (for encoder input, scale=1)
            - 'global_orient_target': augmented beta+scale's global_orient [B, T, 4] (for loss)
            - 'joints_target': augmented beta+scale's joints [B, T, 24, 3] (for loss)
    """
    # Prepare original motion
    original_motion = prepare_original_motion(batch, body_model, device, max_body_joints)
    
    # Apply beta augmentation
    beta_augmented_motion = apply_beta_augmentation(
        original_motion["betas"],
        original_motion["_smpl_body_pose"],
        original_motion["_smpl_global_orient"],
        original_motion["_smpl_trans"],
        body_model,
        beta_augment_std,
    )
    
    # Augment scale: use log-normal distribution to ensure scale > 0
    # scale_augmented = exp(log(1) + noise) = exp(noise)
    log_scale_noise = torch.randn(original_motion["_B"], 1, device=device) * scale_augment_std
    scale_augmented = torch.exp(log_scale_noise)  # [B, 1], always > 0
    
    # Apply scale augmentation
    scale_augmented_motion = apply_scale_augmentation(
        beta_augmented_motion["vertices"],
        beta_augmented_motion["joints"],
        beta_augmented_motion["global_orient"],
        scale_augmented,
    )

    return {
        "betas": original_motion["betas"],
        "betas_augmented": beta_augmented_motion["betas_augmented"],
        "scale": original_motion["scale"],
        "scale_augmented": scale_augmented,
        "global_orient": original_motion["global_orient"],
        "joints": original_motion["joints"],
        "global_orient_target": scale_augmented_motion["global_orient"],
        "joints_target": scale_augmented_motion["joints"],
    }