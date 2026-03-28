import ollama
from src.utils.experience_buffer import ExperienceBuffer
from src.utils.prompts import create_ad_prompt

class LLMAgent:
    def __init__(self, model_name="llama3.1:8b"):
        """
        Initialize LLM-based Autonomous Driving Agent
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.experience_buffer = ExperienceBuffer(max_size=1000)
        
        # Define action mapping
        self.action_map = {
            'FASTER': 0,
            'SLOWER': 1,
            'LANE_LEFT': 2,
            'LANE_RIGHT': 3,
            'IDLE': 4
        }
        
        # Verify model availability
        try:
            available_models = ollama.list()
            # Handle different response formats
            if isinstance(available_models, dict) and 'models' in available_models:
                model_names = [m.get('name', m.get('model', '')) for m in available_models['models']]
            else:
                model_names = []
        
            if model_name in model_names:
                print(f"✓ LLM Agent initialized with {model_name}")
            else:
                print(f"⚠ Model {model_name} not verified in list, but will attempt to use it")
                if model_names:
                    print(f"Available models: {model_names}")
        except Exception as e:
            print(f"⚠ Warning: Could not verify Ollama models: {e}")
            print(f"Will proceed with {model_name} anyway")
    
        print(f"✓ LLM Agent ready with model: {model_name}")
    
    def select_action(self, state):
        """
        Use LLM to select the best driving action
        Args:
            state: Current environment state
        Returns:
            action: Integer (0-4) representing the action
        """
        # Get relevant past experiences
        good_examples, bad_examples = self.experience_buffer.get_relevant_examples(state, k=3)
        
        # Create prompt for LLM
        prompt = create_ad_prompt(state, good_examples, bad_examples)
        
        # Query LLM via Ollama
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,      # Control randomness
                    'num_predict': 10,       # Limit response length
                    'stop': ['\n', 'Action:', 'Available Actions:']  # Stop at action
                }
            )
            
            # Extract and parse the response
            response_text = response['response'].strip().upper()
            
            # Look for each action in the response
            for action_name, action_idx in self.action_map.items():
                if action_name in response_text:
                    return action_idx
                    
        except Exception as e:
            print(f"Error querying LLM: {e}")
        
        # Default to IDLE if parsing fails
        return 4
    
    def store_experience(self, state, action, reward, is_collision):
        """
        Store experience for future learning
        Args:
            state: Current state
            action: Action taken (0-4)
            reward: Reward received
            is_collision: Whether collision occurred
        """
        # Convert action index to name
        action_name = list(self.action_map.keys())[action]
        
        # Determine if experience is good
        # Good: High reward and no collision
        is_good = (reward > 0.5) and (not is_collision)
        
        # Store in experience buffer
        self.experience_buffer.add_experience(state, action_name, reward, is_good)
