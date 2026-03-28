from src.environment.sumo_env import SUMOEnvironment
from src.agents.ddqn_agent import DDQNAgent
from src.utils.reward_functions import calculate_v2i_reward

print("Testing DDQN Agent for V2I Communication...")

# Initialize
env = SUMOEnvironment("sumo_scenarios/highway.sumocfg", gui=False)
agent = DDQNAgent(state_size=3, action_size=3)

action_names = ['STAY', 'SWITCH_RF', 'SWITCH_THZ']

# Training loop
num_episodes = 10
for episode in range(num_episodes):
    env.start()
    env.reset_handovers()
    
    episode_reward = 0
    episode_handovers = 0
    
    for step in range(100):
        # Get V2I state (use dummy AD action for now)
        v2i_state = env.get_v2i_state(ad_action=4)
        
        if v2i_state is None:
            break
        
        # Select action
        action = agent.select_action(v2i_state)
        
        # Simulate step
        env.step()
        
        # Get next state
        next_v2i_state = env.get_v2i_state(ad_action=4)
        
        if next_v2i_state is None:
            break
        
        # Calculate reward
        handover = v2i_state['handover']
        reward = calculate_v2i_reward(v2i_state, action, handover)
        episode_reward += reward
        
        if handover:
            episode_handovers += 1
        
        # Store and train
        done = (step == 99)
        agent.store_transition(v2i_state, action, reward, next_v2i_state, done)
        loss = agent.train_step()
        
    env.close()
    
    print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Handovers: {episode_handovers} | Epsilon: {agent.epsilon:.3f}")

print("\n✅ DDQN Agent Test Completed!")
print(f"Memory size: {len(agent.memory)}")
