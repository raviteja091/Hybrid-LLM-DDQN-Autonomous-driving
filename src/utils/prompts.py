def create_ad_prompt(current_state, good_examples=None, bad_examples=None):
    prompt = """Task: Drive aggressively but safely on a 4-lane highway.

Task Goal:
1) MAXIMIZE vehicle velocity (aim for 30+ m/s).
2) Overtake slower vehicles by changing lanes.
3) Use FASTER action frequently to increase speed.
4) Avoid collisions at all costs.

Current Situation:
Ego Vehicle: Speed={speed:.1f} m/s (Target: 33 m/s), Lane={lane}
{surrounding_info}

IMPORTANT: 
- If speed < 30 m/s, prefer FASTER action.
- If vehicle ahead is slower, prefer LANE_LEFT or FASTER.
- Only use IDLE if already at max speed.

{good_examples_text}
{bad_examples_text}

Available Actions: FASTER, SLOWER, LANE_LEFT, LANE_RIGHT, IDLE
Choose the MOST AGGRESSIVE safe action.

Action:"""

    # Format ego vehicle info
    ego_speed = current_state['ego']['speed']
    ego_lane = current_state['ego']['lane']
    
    # Format surrounding vehicles
    surrounding_info = "No other vehicles detected nearby." if not current_state['surrounding'] else "Surrounding Vehicles:"
    for i, veh in enumerate(current_state['surrounding'][:3]):  # Show up to 3 closest
        surrounding_info += f"\n  Vehicle {i+1}: x={veh['x']:.1f}m, y={veh['y']:.1f}m, vx={veh['vx']:.1f}m/s, lane={veh['lane']}, dist={veh['distance']:.1f}m"
    
    # Format good examples
    good_examples_text = "Good driving examples (consider these successful actions):"
    if good_examples and len(good_examples) > 0:
        for i, ex in enumerate(good_examples[:2]):
            good_examples_text += f"\n  Example {i+1}: State=Speed {ex['state']['ego']['speed']:.1f}m/s, Lane {ex['state']['ego']['lane']}; Action={ex['action']}; Reward={ex['reward']:.2f}"
    else:
        good_examples_text += " No good examples yet."
    
    # Format bad examples
    bad_examples_text = "Bad driving examples (avoid these actions):"
    if bad_examples and len(bad_examples) > 0:
        for i, ex in enumerate(bad_examples[:2]):
            bad_examples_text += f"\n  Example {i+1}: State=Speed {ex['state']['ego']['speed']:.1f}m/s, Lane {ex['state']['ego']['lane']}; Action={ex['action']}; Reward={ex['reward']:.2f}"
    else:
        bad_examples_text += " No bad examples yet."
    
    return prompt.format(
        speed=ego_speed,
        lane=ego_lane,
        surrounding_info=surrounding_info,
        good_examples_text=good_examples_text,
        bad_examples_text=bad_examples_text
    )
