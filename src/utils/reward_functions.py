def calculate_ad_reward(state, collision, off_road=False):
    """
    Calculate reward for autonomous driving agent
    based on equation (3) from the paper
    """
    # Constants from the paper
    c1 = 1.0   # Velocity reward weight
    c2 = 10.0  # Collision penalty weight
    c3 = 0.1   # Right lane preference weight
    c4 = 0.5   # On-road reward weight
    
    # Current state
    ego_speed = state['ego']['speed']
    ego_lane = state['ego']['lane']
    vmin = 0.0
    vmax = 33.33
    
    # Component 1: Speed reward (normalized)
    speed_reward = c1 * (ego_speed - vmin) / (vmax - vmin)
    
    # Component 2: Collision penalty
    collision_penalty = -c2 if collision else 0
    
    # Component 3: Right lane preference
    # Higher reward for being in rightmost lane
    right_lane_reward = c3 if ego_lane == 0 else 0
    
    # Component 4: On-road reward
    on_road_reward = 0 if off_road else c4
    
    # Total reward
    total_reward = speed_reward + collision_penalty + right_lane_reward + on_road_reward
    
    return total_reward

def calculate_v2i_reward(v2i_state, action, handover_occurred):
    """
    Calculate reward for V2I communication
    Based on equation (5) from the paper
    
    Args:
        v2i_state: Current V2I state
        action: Selected action (0=stay, 1=switch_rf, 2=switch_thz)
        handover_occurred: Whether handover happened
    
    Returns:
        reward: Scalar reward value
    """
    # Constants
    c1 = 1.0   # Base connectivity reward
    c2_rf = 0.3  # RF handover penalty
    c2_thz = 0.5  # THz handover penalty (higher)
    c3 = 0.1   # Connectivity bonus per BS
    
    # Base reward for maintaining connection
    base_reward = c1
    
    # Handover penalty (higher for THz)
    handover_penalty = 0.0
    if handover_occurred:
        if action == 2:  # THz handover
            handover_penalty = -c2_thz
        elif action == 1:  # RF handover
            handover_penalty = -c2_rf
    
    # Bonus for having more reachable base stations
    connectivity_bonus = c3 * (v2i_state['reachable_rf'] + v2i_state['reachable_thz'])
    
    total_reward = base_reward + handover_penalty + connectivity_bonus
    
    return total_reward
