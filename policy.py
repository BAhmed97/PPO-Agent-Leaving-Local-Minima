import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(PPOPolicy, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        mean_action = self.policy_net(obs)
        state_value = self.value_net(obs)
        return mean_action, state_value

    def get_action(self, obs):
        # Ensure obs is a PyTorch tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        mean_action, _ = self.forward(obs)
        std = self.log_std.exp()
        dist = Normal(mean_action, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -1.0, 1.0)  # Clip action values to [-1, 1]
        log_prob = dist.log_prob(action_clipped).sum()  # Log probability
        entropy = dist.entropy().sum()  # Entropy
        return action_clipped.detach().cpu().numpy(), log_prob, entropy


class PPOTrainer:
    def __init__(self, env, policy, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def train(self, num_episodes, steps_per_update, epochs):
        all_rewards = []
        all_losses = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            rewards = []
            log_probs = []
            values = []
            entropies = []
            actions = []
            done = False

            for _ in range(steps_per_update):
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, entropy = self.policy.get_action(obs_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)

                obs, reward, terminated, truncated, info = self.env.step(action)
                rewards.append(reward)

                if terminated or truncated:
                    break

            # Compute discounted rewards
            discounted_rewards = self._compute_discounted_rewards(rewards)

            # Train policy
            policy_loss, value_loss = self._update_policy(
                log_probs, entropies, discounted_rewards, actions
            )
            total_loss = policy_loss + value_loss

            all_rewards.append(sum(rewards))
            all_losses.append(total_loss.item())

            print(f"Episode {episode + 1}/{num_episodes}: Reward = {sum(rewards):.2f}, Loss = {total_loss:.4f}")

        return all_rewards, all_losses

    def _compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        return torch.tensor(discounted_rewards, dtype=torch.float32)

    def _update_policy(self, log_probs, entropies, discounted_rewards, actions):
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        policy_loss = []
        value_loss = []

        for log_prob, reward, entropy in zip(log_probs, discounted_rewards, entropies):
            advantage = reward - log_prob.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(advantage ** 2)

        policy_loss = torch.stack(policy_loss).sum()
        value_loss = torch.stack(value_loss).sum()

        self.optimizer.zero_grad()
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropies.sum()
        total_loss.backward()
        self.optimizer.step()

        return policy_loss, value_loss
