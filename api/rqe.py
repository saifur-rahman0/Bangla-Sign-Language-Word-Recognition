import numpy as np

def apply_rqe(sequence, quantization_step=0.05, shoulder_fixing=True):
    """
    Applies Relative Quantization Encoding (RQE) and optional Shoulder Fixing (RQE-SF)
    to a sequence of raw landmarks.
    
    Parameters:
      sequence: np.ndarray of shape (T, 387) representing T frames of 129 3D landmarks.
      quantization_step: float, step size for trajectory discretization. None to disable quantization.
      shoulder_fixing: bool, if True, uses RQE-SF (shoulder landmarks fixed to frame 0).
      
    Returns:
      np.ndarray of shape (T, 387) containing RQE-encoded landmarks.
    """
    T = sequence.shape[0]
    # Reshape sequence to (T, 129, 3)
    seq_reshaped = sequence.copy().reshape(T, 129, 3)
    
    FACE_LM = 54
    HAND_LM = 21
    POSE_LM = 33
    
    FACE_START = 0
    LEFT_HAND_START = FACE_LM
    RIGHT_HAND_START = LEFT_HAND_START + HAND_LM
    POSE_START = RIGHT_HAND_START + HAND_LM
    
    # 1. First frame calculations for Pose anchoring
    # Pose indices: left shoulder (11), right shoulder (12)
    p_left_shoulder_idx = POSE_START + 11
    p_right_shoulder_idx = POSE_START + 12
    
    frame0_pose = seq_reshaped[0]
    l_shoulder0 = frame0_pose[p_left_shoulder_idx]
    r_shoulder0 = frame0_pose[p_right_shoulder_idx]
    
    # Mid shoulder at frame 0
    if np.any(l_shoulder0 != 0) and np.any(r_shoulder0 != 0):
        mid_shoulder_ref = (l_shoulder0 + r_shoulder0) / 2
        scale_pose = np.linalg.norm(l_shoulder0 - r_shoulder0)
    else:
        mid_shoulder_ref = np.zeros(3)
        scale_pose = 1.0
        
    scale_pose = max(scale_pose, 1e-6)
    
    # 2. Process frame-by-frame
    rqe_seq = []
    for t in range(T):
        frame = seq_reshaped[t].copy()
        
        # Face (indices: 0 to 53)
        # Face anchor: nose tip (index 22 in IMPORTANT_FACE_IDX)
        nose_idx = FACE_START + 22
        nose = frame[nose_idx]
        face_active = np.any(nose != 0)
        
        # Calculate local face scale: eye-to-eye distance (left eye 4, right eye 10)
        l_eye = frame[FACE_START + 4]
        r_eye = frame[FACE_START + 10]
        if np.any(l_eye != 0) and np.any(r_eye != 0):
            scale_face = np.linalg.norm(l_eye - r_eye)
        else:
            scale_face = 1.0
        scale_face = max(scale_face, 1e-6)
        
        # Normalize Face
        face_mask = np.any(frame[FACE_START:LEFT_HAND_START] != 0, axis=1)
        if face_active:
            # Anchor face joints to nose tip
            frame[FACE_START:LEFT_HAND_START][face_mask] = (
                frame[FACE_START:LEFT_HAND_START][face_mask] - nose
            ) / scale_face
        else:
            frame[FACE_START:LEFT_HAND_START] = 0.0
            
        # Left Hand (indices: 54 to 74)
        # Anchor: Wrist (index 0 of left hand)
        l_wrist_idx = LEFT_HAND_START
        l_wrist = frame[l_wrist_idx]
        lh_active = np.any(l_wrist != 0)
        
        # Scale: wrist to middle finger MCP (index 9 of left hand)
        l_mcp = frame[LEFT_HAND_START + 9]
        if np.any(l_mcp != 0) and lh_active:
            scale_lh = np.linalg.norm(l_mcp - l_wrist)
        else:
            scale_lh = 1.0
        scale_lh = max(scale_lh, 1e-6)
        
        lh_mask = np.any(frame[LEFT_HAND_START:RIGHT_HAND_START] != 0, axis=1)
        if lh_active:
            frame[LEFT_HAND_START:RIGHT_HAND_START][lh_mask] = (
                frame[LEFT_HAND_START:RIGHT_HAND_START][lh_mask] - l_wrist
            ) / scale_lh
        else:
            frame[LEFT_HAND_START:RIGHT_HAND_START] = 0.0
            
        # Right Hand (indices: 75 to 95)
        # Anchor: Wrist (index 0 of right hand)
        r_wrist_idx = RIGHT_HAND_START
        r_wrist = frame[r_wrist_idx]
        rh_active = np.any(r_wrist != 0)
        
        # Scale: wrist to middle finger MCP (index 9 of right hand)
        r_mcp = frame[RIGHT_HAND_START + 9]
        if np.any(r_mcp != 0) and rh_active:
            scale_rh = np.linalg.norm(r_mcp - r_wrist)
        else:
            scale_rh = 1.0
        scale_rh = max(scale_rh, 1e-6)
        
        rh_mask = np.any(frame[RIGHT_HAND_START:POSE_START] != 0, axis=1)
        if rh_active:
            frame[RIGHT_HAND_START:POSE_START][rh_mask] = (
                frame[RIGHT_HAND_START:POSE_START][rh_mask] - r_wrist
            ) / scale_rh
        else:
            frame[RIGHT_HAND_START:POSE_START] = 0.0
            
        # Pose (indices: 96 to 128)
        # Anchor: mid-shoulder
        if shoulder_fixing:
            # Anchor to mid-shoulder of frame 0 (RQE-SF)
            pose_anchor = mid_shoulder_ref
            pose_scale = scale_pose
        else:
            # Anchor to mid-shoulder of current frame t
            l_shoulder = frame[p_left_shoulder_idx]
            r_shoulder = frame[p_right_shoulder_idx]
            if np.any(l_shoulder != 0) and np.any(r_shoulder != 0):
                pose_anchor = (l_shoulder + r_shoulder) / 2
                pose_scale = np.linalg.norm(l_shoulder - r_shoulder)
            else:
                pose_anchor = np.zeros(3)
                pose_scale = 1.0
        
        pose_scale = max(pose_scale, 1e-6)
        pose_mask = np.any(frame[POSE_START:] != 0, axis=1)
        
        if np.any(pose_anchor != 0):
            frame[POSE_START:][pose_mask] = (
                frame[POSE_START:][pose_mask] - pose_anchor
            ) / pose_scale
        else:
            frame[POSE_START:] = 0.0
            
        # 3. Quantization step
        if quantization_step is not None:
            # Mask active landmarks to avoid quantizing zero vectors into non-zero bins
            all_mask = np.any(frame != 0, axis=1)
            frame[all_mask] = np.round(frame[all_mask] / quantization_step) * quantization_step
            
        rqe_seq.append(frame.flatten())
        
    return np.array(rqe_seq)
