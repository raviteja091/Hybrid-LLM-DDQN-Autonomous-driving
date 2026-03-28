import os
import sys
import traci
import numpy as np

class SUMOEnvironment:
    def __init__(self, sumo_cfg_file, gui=False):
        """
        Initialize SUMO Environment
        Args:
            sumo_cfg_file: Path to .sumocfg file
            gui: Whether to use GUI (True) or headless (False)
        """
        self.sumo_cfg = sumo_cfg_file
        self.use_gui = gui
        self.ego_id = "ego"
        self.step_length = 0.1
        self.is_running = False

        # NEW: Base station configuration
        self.num_rf_bs = 5   # RF base stations (long range)
        self.num_thz_bs = 20  # THz base stations (short range)
        self.base_stations = []
        self.current_bs = None
        self.handover_count = 0
        self._setup_base_stations()
        
    def start(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary, 
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--collision.action", "warn",
            "--no-warnings", "true",
            "--no-step-log", "true"
        ]
        traci.start(sumo_cmd)
        self.is_running = True
        # <— Perform first step so vehicles depart
        traci.simulationStep()
        print("✓ SUMO simulation started")
        
    def reset(self):
        """Reset environment to initial state"""
        if traci.isLoaded():
            traci.close()
        self.start()
        # <— Add this line to actually spawn vehicles
        traci.simulationStep()
        return self.get_state()
    
    def get_state(self):
        """
        Get current state of ego vehicle and surroundings
        Returns: Dictionary with ego and surrounding vehicle info
        """
        if not self.is_running or self.ego_id not in traci.vehicle.getIDList():
            return None
            
        # Ego vehicle state
        ego_pos = traci.vehicle.getPosition(self.ego_id)
        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        ego_lane = traci.vehicle.getLaneIndex(self.ego_id)
        ego_x = traci.vehicle.getRoadID(self.ego_id)
        
        # Get surrounding vehicles within 100m range
        all_vehicles = traci.vehicle.getIDList()
        surrounding_vehicles = []
        
        for veh_id in all_vehicles:
            if veh_id == self.ego_id:
                continue
                
            try:
                veh_pos = traci.vehicle.getPosition(veh_id)
                veh_speed = traci.vehicle.getSpeed(veh_id)
                veh_lane = traci.vehicle.getLaneIndex(veh_id)
                
                # Calculate relative position
                x_rel = veh_pos[0] - ego_pos[0]
                y_rel = veh_pos[1] - ego_pos[1]
                
                # Only consider vehicles within 100m
                distance = np.sqrt(x_rel**2 + y_rel**2)
                if distance < 100:
                    vx_rel = veh_speed - ego_speed
                    
                    surrounding_vehicles.append({
                        'id': veh_id,
                        'x': x_rel,
                        'y': y_rel,
                        'vx': vx_rel,
                        'speed': veh_speed,
                        'lane': veh_lane,
                        'distance': distance
                    })
            except:
                continue
        
        # Sort by distance (closest first)
        surrounding_vehicles.sort(key=lambda v: v['distance'])
        
        state = {
            'ego': {
                'x': ego_pos[0],
                'y': ego_pos[1],
                'speed': ego_speed,
                'lane': ego_lane
            },
            'surrounding': surrounding_vehicles[:5]  # Keep only 5 closest
        }
        
        return state
    
    def apply_action(self, action):
        """
        Apply autonomous driving action to ego vehicle
        Actions:
            0 = FASTER (accelerate)
            1 = SLOWER (decelerate)
            2 = LANE_LEFT (change to left lane)
            3 = LANE_RIGHT (change to right lane)
            4 = IDLE (maintain current speed)
        """
        if not self.is_running or self.ego_id not in traci.vehicle.getIDList():
            return False
            
        current_speed = traci.vehicle.getSpeed(self.ego_id)
        current_lane = traci.vehicle.getLaneIndex(self.ego_id)
        max_speed = 33.33
        
        try:
            if action == 0:  # FASTER
                new_speed = min(current_speed + 3.0, max_speed)
                traci.vehicle.setSpeed(self.ego_id, new_speed)
                
            elif action == 1:  # SLOWER
                new_speed = max(current_speed - 3.0, 0)
                traci.vehicle.setSpeed(self.ego_id, new_speed)
                
            elif action == 2:  # LANE_LEFT
                if current_lane < 3:  # Can go left
                    traci.vehicle.changeLane(self.ego_id, current_lane + 1, 3.0)
                    
            elif action == 3:  # LANE_RIGHT
                if current_lane > 0:  # Can go right
                    traci.vehicle.changeLane(self.ego_id, current_lane - 1, 3.0)
                    
            # action == 4 is IDLE (do nothing)
            
            return True
        except Exception as e:
            print(f"Error applying action: {e}")
            return False
    
    def step(self):
        """Advance simulation by one step"""
        if self.is_running:
            traci.simulationStep()
    
    def check_collision(self):
        """Check if ego vehicle has collided"""
        if not self.is_running:
            return False
        try:
            colliding = traci.simulation.getCollidingVehiclesIDList()
            return self.ego_id in colliding
        except:
            return False
    
    def get_ego_position(self):
        """Get ego vehicle position"""
        if self.ego_id in traci.vehicle.getIDList():
            return traci.vehicle.getPosition(self.ego_id)
        return None
    
    def is_ego_active(self):
        """Check if ego vehicle is still in simulation"""
        return self.ego_id in traci.vehicle.getIDList()
    
    def close(self):
        """Close SUMO simulation"""
        if traci.isLoaded():
            traci.close()
            self.is_running = False
            print("✓ SUMO simulation closed")
    
    def _setup_base_stations(self):
        """Setup RF and THz base stations along the highway"""
        self.base_stations = []
    
        # RF Base Stations (longer range, every 600m)
        for i in range(self.num_rf_bs):
            self.base_stations.append({
                'id': f'RF_BS_{i}',
                'type': 'RF',
                'position': (i * 800, 0),  # Evenly spaced along highway
                'range': 400,  
                'capacity': 10,
                'users': 0
            })
    
        # THz Base Stations (shorter range, every 150m)
        for i in range(self.num_thz_bs):
            self.base_stations.append({
                'id': f'THz_BS_{i}',
                'type': 'THz',
                'position': (i * 200, 0),
                'range': 80,  
                'capacity': 5,
                'users': 0
            })

    def get_v2i_state(self, ad_action):
        """
        Get V2I communication state
        Returns state for DDQN agent to select base station
        """
        if not self.is_running or self.ego_id not in traci.vehicle.getIDList():
            return None
    
        ego_pos = traci.vehicle.getPosition(self.ego_id)
    
        # Count reachable base stations
        reachable_rf = 0
        reachable_thz = 0
        closest_bs = None
        min_distance = float('inf')
    
        for bs in self.base_stations:
            distance = np.sqrt((ego_pos[0] - bs['position'][0])**2 + 
                          (ego_pos[1] - bs['position'][1])**2)
        
            # Check if in range
            if distance < bs['range']:
                if bs['type'] == 'RF':
                    reachable_rf += 1
                else:
                    reachable_thz += 1
            
                # Track closest base station
                if distance < min_distance:
                    min_distance = distance
                    closest_bs = bs['id']
    
        # Detect handover
        handover_occurred = False
        if self.current_bs is not None and closest_bs != self.current_bs:
            handover_occurred = True
            self.handover_count += 1
    
        self.current_bs = closest_bs
    
        return {
            'reachable_rf': reachable_rf,
            'reachable_thz': reachable_thz,
            'ad_action': ad_action,
            'handover': handover_occurred,
            'current_bs': closest_bs
        }

    def get_handover_count(self):
        """Get total number of handovers"""
        return self.handover_count

    def reset_handovers(self):
        """Reset handover counter"""
        self.handover_count = 0
        self.current_bs = None

