"""
Hybrid LLM-DDQN Training Loop
Combines LLM-based Autonomous Driving with DDQN-based V2I Communication
"""

from src.environment.sumo_env import SUMOEnvironment
from src.agents.llm_agent import LLMAgent
from src.agents.ddqn_agent import DDQNAgent
from src.utils.reward_functions import calculate_ad_reward, calculate_v2i_reward
import matplotlib.pyplot as plt
import time
import os

# =====================================================
# CONFIGURATION
# =====================================================
EPISODES = 10
STEPS_PER_EPISODE = 1000
LLM_QUERY_INTERVAL = 50  # Query LLM every N steps (to speed up training)
GUI_MODE = True
SAVE_RESULTS = True

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# =====================================================
# INITIALIZE AGENTS AND ENVIRONMENT
# =====================================================
print("="*60)
print("Hybrid LLM-DDQN Training System")
print("="*60)

env = SUMOEnvironment("sumo_scenarios/highway.sumocfg", gui=GUI_MODE)
llm_agent = LLMAgent(model_name="llama3.1:8b")
ddqn_agent = DDQNAgent(state_size=3, action_size=3)

# Metrics storage
ad_rewards_history = []
v2i_rewards_history = []
collisions_history = []
handovers_history = []
episode_times = []

# Action name mapping for logging
action_names = ['FASTER', 'SLOWER', 'LANE_LEFT', 'LANE_RIGHT', 'IDLE']

# =====================================================
# TRAINING LOOP
# =====================================================
print(f"\nStarting training: {EPISODES} episodes, {STEPS_PER_EPISODE} steps each")
print(f"LLM query interval: every {LLM_QUERY_INTERVAL} steps")
print("="*60 + "\n")

total_start_time = time.time()

for ep in range(EPISODES):
    episode_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"EPISODE {ep+1}/{EPISODES}")
    print(f"{'='*60}")
    
    # Reset environment and counters
    state = env.reset()
    env.reset_handovers()
    
    ad_total_reward = 0
    v2i_total_reward = 0
    collisions = 0
    handovers = 0
    ad_action = 4  # Initialize with IDLE action
    
    # Episode loop
    for t in range(STEPS_PER_EPISODE):
        if state is None:
            print(f"  ⚠ State is None at step {t}, ending episode")
            break
        
        # Progress indicator
        if t % 5 == 0:
            print(f"\n  Step {t}/{STEPS_PER_EPISODE}")
        
        # ---- LLM-based Autonomous Driving ----
        # Query LLM periodically to reduce computation time
        if t % LLM_QUERY_INTERVAL == 0:
            print(f"    Querying LLM...", end=" ", flush=True)
            llm_start = time.time()
            ad_action = llm_agent.select_action(state)
            llm_time = time.time() - llm_start
            print(f"Done ({llm_time:.2f}s) -> Action: {action_names[ad_action]}")
        
        # Apply driving action
        env.apply_action(ad_action)
        env.step()
        next_state = env.get_state()
        
        if next_state is None:
            print(f"  ⚠ Next state is None at step {t}, ending episode")
            break
        
        # Check collision
        collision = env.check_collision()
        
        # Calculate AD reward
        ad_reward = calculate_ad_reward(state, collision)
        ad_total_reward += ad_reward
        
        # Store experience for LLM agent
        llm_agent.store_experience(state, ad_action, ad_reward, collision)
        
        if collision:
            collisions += 1
            print(f"    ⚠ COLLISION DETECTED at step {t}! Ending episode.")
            break
        
        # ---- DDQN-based V2I Communication ----
        v2i_state = env.get_v2i_state(ad_action)
        
        if v2i_state is None:
            print(f"  ⚠ V2I state is None at step {t}, ending episode")
            break
        
        # Select V2I action
        v2i_action = ddqn_agent.select_action(v2i_state, training=True)
        
        # Calculate V2I reward
        v2i_reward = calculate_v2i_reward(v2i_state, v2i_action, v2i_state['handover'])
        v2i_total_reward += v2i_reward
        
        # Get next V2I state
        next_v2i_state = env.get_v2i_state(ad_action)
        
        if next_v2i_state is None:
            next_v2i_state = v2i_state  # Use current state as fallback
        
        # Store transition and train DDQN
        done = (t == STEPS_PER_EPISODE - 1)
        ddqn_agent.store_transition(v2i_state, v2i_action, v2i_reward, next_v2i_state, done)
        
        loss = ddqn_agent.train_step()
        
        # Track handovers
        if v2i_state['handover']:
            handovers += 1
            print(f"    → Handover occurred (Total: {handovers})")
        
        # Log progress every 5 steps
        if t % 5 == 0:
            print(f"    AD Reward: {ad_reward:.2f} | V2I Reward: {v2i_reward:.2f}")
            if loss is not None:
                print(f"    DDQN Loss: {loss:.4f} | Epsilon: {ddqn_agent.epsilon:.3f}")
        
        # Update state for next iteration
        state = next_state
    
    # Episode complete - record metrics
    episode_time = time.time() - episode_start_time
    episode_times.append(episode_time)
    
    ad_rewards_history.append(ad_total_reward)
    v2i_rewards_history.append(v2i_total_reward)
    collisions_history.append(collisions)
    handovers_history.append(handovers)
    
    # Close environment
    env.close()
    
    # Print episode summary
    print(f"\n{'='*60}")
    print(f"✓ EPISODE {ep+1} COMPLETE")
    print(f"{'='*60}")
    print(f"  Time: {episode_time:.2f}s")
    print(f"  AD Reward: {ad_total_reward:.2f}")
    print(f"  V2I Reward: {v2i_total_reward:.2f}")
    print(f"  Collisions: {collisions}")
    print(f"  Handovers: {handovers}")
    print(f"  LLM Experience Buffer: {len(llm_agent.experience_buffer.good_experiences)} good, "
          f"{len(llm_agent.experience_buffer.bad_experiences)} bad")
    print(f"  DDQN Memory: {len(ddqn_agent.memory)}")
    print(f"  DDQN Epsilon: {ddqn_agent.epsilon:.3f}")
    print(f"{'='*60}")

# =====================================================
# TRAINING COMPLETE
# =====================================================
total_time = time.time() - total_start_time

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Average Episode Time: {sum(episode_times)/len(episode_times):.2f}s")
print(f"Total Episodes: {EPISODES}")
print(f"Final DDQN Epsilon: {ddqn_agent.epsilon:.3f}")
print(f"{'='*60}\n")

# =====================================================
# SAVE RESULTS
# =====================================================
if SAVE_RESULTS:
    # Save metrics to file
    with open('results/training_metrics.txt', 'w') as f:
        f.write("Episode,AD_Reward,V2I_Reward,Collisions,Handovers,Time\n")
        for i in range(EPISODES):
            f.write(f"{i+1},{ad_rewards_history[i]:.2f},{v2i_rewards_history[i]:.2f},"
                   f"{collisions_history[i]},{handovers_history[i]},{episode_times[i]:.2f}\n")
    
    print("✓ Metrics saved to results/training_metrics.txt")
    
    # Save model
    ddqn_agent.save('results/ddqn_model.pth')

# =====================================================
# VISUALIZATION
# =====================================================
print("\nGenerating plots...")

plt.figure(figsize=(14, 8))

# Plot 1: Rewards
plt.subplot(2, 2, 1)
plt.plot(range(1, EPISODES+1), ad_rewards_history, 'b-o', label="AD Reward", linewidth=2)
plt.plot(range(1, EPISODES+1), v2i_rewards_history, 'g-s', label="V2I Reward", linewidth=2)
plt.title("Rewards per Episode", fontsize=14, fontweight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Collisions
plt.subplot(2, 2, 2)
plt.plot(range(1, EPISODES+1), collisions_history, 'r-^', label="Collisions", linewidth=2)
plt.title("Collisions per Episode", fontsize=14, fontweight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Number of Collisions", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: Handovers
plt.subplot(2, 2, 3)
plt.plot(range(1, EPISODES+1), handovers_history, 'm-d', label="Handovers", linewidth=2)
plt.title("Handovers per Episode", fontsize=14, fontweight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Number of Handovers", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 4: Episode Time
plt.subplot(2, 2, 4)
plt.plot(range(1, EPISODES+1), episode_times, 'c-*', label="Episode Time", linewidth=2)
plt.title("Training Time per Episode", fontsize=14, fontweight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_summary.png", dpi=300, bbox_inches='tight')
print("✓ Plot saved to results/training_summary.png")

plt.show()

print("\n" + "="*60)
print("ALL TASKS COMPLETE")
print("="*60)
