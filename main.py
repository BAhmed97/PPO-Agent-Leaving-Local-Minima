import os
import torch
import matplotlib.pyplot as plt
from env import CustomMetaDriveEnv
from policy import PPOPolicy, PPOTrainer


def main():
    # Define observation and action dimensions
    obs_dim = 260  # Adjust based on observation shape
    act_dim = 2  # Assuming 2 actions: steering and throttle

    # Initialize PPO policy
    policy = PPOPolicy(obs_dim, act_dim)

    # Create custom environment
    env_config = {
        "use_render": False,
        "traffic_density": 0.3,
        "start_seed": 0,
        "num_agents": 1
    }
    env = CustomMetaDriveEnv(env_config, policy)

    # Initialize PPO trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2
    )

    # Train the policy
    rewards, losses = trainer.train(
        num_episodes=500,
        steps_per_update=5000,
        epochs=10
    )

    # Save the trained policy model
    model_path = "ppo_metadrive_model.pth"
    torch.save(policy.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Plot rewards over episodes
    plt.figure()
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.title("Rewards per Episode")
    plt.savefig("rewards_plot.png")
    plt.show()

    # Plot loss over training iterations
    plt.figure()
    plt.plot(losses, label="Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Update")
    plt.savefig("losses_plot.png")
    plt.show()

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
