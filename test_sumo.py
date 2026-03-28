import os
import sys

# Set SUMO_HOME environment variable
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print("✓ SUMO_HOME found:", os.environ['SUMO_HOME'])
else:
    print("❌ SUMO_HOME not set!")
    print("Please set SUMO_HOME environment variable to your SUMO installation directory")
    sys.exit(1)

try:
    import traci
    print("✓ TraCI imported successfully!")
except ImportError:
    print("❌ Could not import traci. Please check SUMO installation.")
    sys.exit(1)

print("\n=== SUMO Setup Complete ===")
