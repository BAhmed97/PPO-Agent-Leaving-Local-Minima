import gymnasium as gym
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv


class CustomMetaDriveEnv(MetaDriveEnv):
    def __init__(self, config=None, policy_class=None):
        config = config or {}
        config["use_render"] = False  # Disable rendering
        super().__init__(config)

        self.policy_class = policy_class  # Optional policy class
        self.agent_id = "default_agent"

        # Adjust observation space
        dummy_obs = self._get_dummy_augmented_obs()
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
        )

    def _get_dummy_augmented_obs(self):
        dummy_obs = super().reset()[0]  # Initial observation
        dummy_lidar_data = np.zeros(120)  # Placeholder for LiDAR data
        dummy_lane_deviation = [0.0]  # Placeholder for lane deviation
        augmented_obs = np.concatenate([dummy_obs, dummy_lidar_data, dummy_lane_deviation])
        assert len(augmented_obs) == 260, f"Augmented observation has incorrect length: {len(augmented_obs)}"
        return augmented_obs

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._augment_obs(obs)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)  # Ensure action is within valid range
        obs, reward, terminated, truncated, info = super().step(action)
        reward += self._compute_reward(info, action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        agent = self.agent_manager.active_agents[self.agent_id]
        lidar_data = getattr(agent.lidar, "raw_data", np.zeros(120))
        lane_deviation = self._compute_lane_deviation(agent)
        augmented_obs = np.concatenate([obs, lidar_data, [lane_deviation]])
        assert len(augmented_obs) == 260, f"Augmented observation has incorrect length: {len(augmented_obs)}"
        return augmented_obs

    def _compute_lane_deviation(self, agent):
        if hasattr(agent.navigation, "lane"):
            current_position = agent.position
            lane_position = agent.navigation.lane.position(current_position)
            return np.linalg.norm(np.array(current_position) - np.array(lane_position))
        return 0.0

    def _compute_reward(self, info, action):
        reward = 0.0
        if not info.get("out_of_road", False):
            reward += 1.0
        if abs(action[0]) > 0.8:
            reward -= 0.5
        reward -= info.get("step_energy", 0) * 0.1
        return reward


def create_env(policy_class=None):
    """
    Creates and returns an instance of CustomMetaDriveEnv with predefined configurations.
    """
    env_config = {
        "use_render": False,  # Disable rendering
        "traffic_density": 0.3,
        "random_traffic": True,
        "need_inverse_traffic": True,
        "map": "S",
        "manual_control": False,
        "decision_repeat": 5,
        "vehicle_config": {
            "lidar": {"num_lasers": 120, "distance": 50},
            "show_dest_mark": True,
            "show_navi_mark": True,
        },
        "show_interface": False,
        "top_down_camera_initial_x": 0,
        "top_down_camera_initial_y": 0,
        "top_down_camera_initial_z": 50,
        "crash_object_done": True,
        "crash_vehicle_done": True,
        "out_of_route_done": True,
    }
    return CustomMetaDriveEnv(env_config, policy_class=policy_class)
