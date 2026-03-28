import ollama

# Test basic inference with your local model
print("Testing Llama 3.1 8B via Ollama...")

response = ollama.generate(
    model='llama3.1:8b',  # Use your model name from 'ollama list'
    prompt='You are an autonomous driving assistant. What should a vehicle do if there\'s a car ahead moving slower?'
)

print("\nResponse:")
print(response['response'])
print("\n✓ Ollama test successful!")
