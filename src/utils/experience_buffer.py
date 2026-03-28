from collections import deque
import numpy as np

class ExperienceBuffer:
    def __init__(self, max_size=1000):
        self.good_experiences = deque(maxlen=max_size)
        self.bad_experiences = deque(maxlen=max_size)
        
    def add_experience(self, state, action, reward, is_good):
        """Add experience to the appropriate buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward
        }
        
        if is_good:
            self.good_experiences.append(experience)
        else:
            self.bad_experiences.append(experience)
    
    def get_relevant_examples(self, current_state, k=3):
        """
        Retrieve the K most relevant past experiences
        using Euclidean distance as similarity measure
        """
        # Get relevant examples from both good and bad experiences
        good_relevant = self._get_k_nearest(current_state, self.good_experiences, k)
        bad_relevant = self._get_k_nearest(current_state, self.bad_experiences, k)
        
        return good_relevant, bad_relevant
    
    def _get_k_nearest(self, current_state, experience_pool, k):
        """Find K nearest experiences using Euclidean distance"""
        if not experience_pool:
            return []
            
        # Calculate distance for each experience
        distances = []
        for exp in experience_pool:
            dist = self._calculate_state_distance(current_state, exp['state'])
            distances.append((dist, exp))
        
        # Sort by distance and return top k
        distances.sort(key=lambda x: x[0])
        return [exp for _, exp in distances[:k]]
    
    def _calculate_state_distance(self, state1, state2):
        """Calculate Euclidean distance between two states"""
        # Distance based on ego vehicle state
        ego_dist = np.sqrt(
            (state1['ego']['speed'] - state2['ego']['speed'])**2 +
            (state1['ego']['lane'] - state2['ego']['lane'])**2
        )
        
        # This can be enhanced with surrounding vehicles
        return ego_dist
