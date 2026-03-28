import time
from src.environment.sumo_env import SUMOEnvironment
from src.agents.llm_agent import LLMAgent
from src.utils.reward_functions import calculate_ad_reward

print("Testing LLM-Based Autonomous Driving Agent...")

# Initialize environment and agent
env = SUMOEnvironment("sumo_scenarios/highway.sumocfg", gui=True)
agent = LLMAgent(model_name="llama3.1:8b")

# Action names
action_names = ['FASTER', 'SLOWER', 'LANE_LEFT', 'LANE_RIGHT', 'IDLE']

# Start simulation
env.start()
print("Simulation started. Watch the red car in SUMO GUI!")

# Run for a set number of steps
for episode in range(2):
    print(f"\n--- Starting Episode {episode+1} ---")
    
    # Reset environment
    state = env.get_state()
    total_reward = 0
    step_count = 0
    
    for step in range(100):
        if state is None:
            print("⚠ Ego vehicle not available")
            break
            
        # Get action from LLM
        action = agent.select_action(state)
        
        # Apply action
        env.apply_action(action)
        env.step()
        
        # Get new state
        next_state = env.get_state()
        if next_state is None:
            break
            
        # Check for collision
        collision = env.check_collision()
        
        # Calculate reward
        reward = calculate_ad_reward(state, collision)
        total_reward += reward
        
        # Store experience
        agent.store_experience(state, action, reward, collision)
        
        # Print every 20 steps
        if step % 20 == 0:
            ego = state['ego']
            print(f"Step {step:3d} | Spd: {ego['speed']:5.1f} | Ln: {ego['lane']} | Action: {action_names[action]} | Reward: {reward:5.2f}")
        
        # End if collision
        if collision:
            print("⚠ COLLISION OCCURRED!")
            break
            
        # Update state
        state = next_state
        step_count = step
        
        # Slow down for visualization
        time.sleep(0.1)
    
    print(f"Episode {episode+1} completed: {step_count+1} steps, Total Reward: {total_reward:.2f}")
    
    # Reset simulation
    if episode == 0:
        time.sleep(2)

print("\n✅ LLM Agent Test Completed!")
print(f"Experience Buffer: {len(agent.experience_buffer.good_experiences)} good, {len(agent.experience_buffer.bad_experiences)} bad experiences")
env.close()
