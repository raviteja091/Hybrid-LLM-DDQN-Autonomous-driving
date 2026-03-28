from src.environment.sumo_env import SUMOEnvironment
import time

print("Testing SUMO Environment Wrapper...")

# Create environment with GUI
env = SUMOEnvironment("sumo_scenarios/highway.sumocfg", gui=True)
env.start()

print("\nRunning 100 simulation steps with random actions...")
print("Watch the RED car in SUMO GUI!\n")

action_names = ['FASTER', 'SLOWER', 'LANE_LEFT', 'LANE_RIGHT', 'IDLE']

for step in range(100):
    # Get current state
    state = env.get_state()
    
    if state is None:
        print("Ego vehicle not available")
        break
    
    # Print every 10 steps
    if step % 10 == 0:
        ego = state['ego']
        print(f"Step {step:3d} | Speed: {ego['speed']:5.1f} m/s | Lane: {ego['lane']} | Surrounding vehicles: {len(state['surrounding'])}")
    
    # Apply random action every 20 steps
    if step % 20 == 0:
        import random
        action = random.randint(0, 4)
        env.apply_action(action)
        print(f"  → Action: {action_names[action]}")
    
    # Check collision
    if env.check_collision():
        print("⚠ COLLISION DETECTED!")
        break
    
    # Step simulation
    env.step()
    time.sleep(0.05)  # Slow down for visualization

print("\n✓ Test completed!")
env.close()
