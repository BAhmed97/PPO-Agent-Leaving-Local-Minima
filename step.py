from env import create_env  # Import the function to create the environment
import numpy as np

def main():
    # Initialize environment
    print("Initializing the environment...")
    env = create_env(policy_class=None)  # Use the policy class if needed
    print(f"Environment initialized: {env}")

    # Print environment details
    print("\n--- ENVIRONMENT DETAILS ---")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Environment Config: {env.config}")
    print(f"Policy Class: {getattr(env, 'policy_class', None)}")

    # Reset the environment and get initial observation
    obs, info = env.reset()
    print("\n--- INITIAL OBSERVATION AND INFO ---")
    print(f"Initial Observation: {obs}")
    print(f"Info: {info}")

    # Step through the environment to collect information
    print("\n--- ENVIRONMENT STEPPING ---")
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)

        # Log step details
        print(f"Step {steps + 1}")
        print(f"Action Taken: {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"Info: {info}")

        total_reward += reward
        done = terminated or truncated
        steps += 1

    print("\n--- FINAL DETAILS ---")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
