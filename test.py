import torch
from env import create_env
from policy import PPOPolicy

def test_trained_model(model_path):
    """
    Test a trained model in the custom MetaDrive environment and print detailed step information.
    """
    # Load the custom MetaDrive environment
    env = create_env()
    policy = PPOPolicy(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0])
    
    # Load trained model weights into the policy
    state_dict = torch.load(model_path)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Reset environment and policy
    obs = env.reset()
    done = False
    total_reward = 0
    total_steps = 0

    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, _, _ = policy.get_action(obs_tensor)

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)

        # Print step information
        print("\n=== Step Information ===")
        print(f"Step: {total_steps + 1}")
        print(f"Observation: {obs}")
        print(f"Action Taken: {action}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        
        total_reward += reward
        total_steps += 1

    print("\n=== Episode Summary ===")
    print(f"Total Reward: {total_reward}")
    print(f"Total Steps: {total_steps}")
    env.close()

if __name__ == "__main__":
    model_path = "ppo_metadrive_model.pth"  # Path to the trained model
    test_trained_model(model_path)
