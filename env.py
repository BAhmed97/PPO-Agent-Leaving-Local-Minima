import gym
from metadrive.envs.metadrive_env import MetaDriveEnv

class CustomMetaDriveGymWrapper(gym.Env):
    def __init__(self, config, policy_class):
        super(CustomMetaDriveGymWrapper, self).__init__()
        self.env = MetaDriveEnv(config)
        self.policy = policy_class
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Augment the observation if needed
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Modify the step logic if additional processing is needed
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

def create_env(policy_class=None):
    """
    Creates and returns an instance of CustomMetaDriveGymWrapper with predefined configurations.
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
    return CustomMetaDriveGymWrapper(config=env_config, policy_class=policy_class)
