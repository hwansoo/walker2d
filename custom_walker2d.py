import numpy as np
import gymnasium as gym
import os
import bisect

# The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
# | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
# | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
# | 0   | x-coordinate of the torso                          | -Inf | Inf | rootz                            | slide | position (m)             |
# | 1   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
# | 2   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
# | 3   | angle of the thigh joint                           | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
# | 4   | angle of the leg joint                             | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
# | 5   | angle of the foot joint                            | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
# | 6   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
# | 7   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
# | 8   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
# | 9   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
# | 10  | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
# | 11  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
# | 12  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
# | 13  | angular velocity of the leg hinge                  | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
# | 14  | angular velocity of the foot hinge                 | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
# | 15  | angular velocity of the thigh hinge                | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
# | 16  | angular velocity of the leg hinge                  | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
# | 17  | angular velocity of the foot hinge                 | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |

# The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
# | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
# |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
# | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
# | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
# | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
# | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
# | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
# | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, render_mode="human", bump_practice=False, bump_challenge=False):
        if bump_challenge:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0),
            )
        elif bump_practice:
            env = gym.make(
                "Walker2d-v5",
                xml_file=os.getcwd() + "/asset/custom_walker2d_bumps_practice.xml",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
                healthy_z_range=(0.5, 10.0),
            )
        else:
            env = gym.make(
                "Walker2d-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
                frame_skip=10,
            )

        super().__init__(env)

        if bump_challenge or bump_practice:
            # bump 지형 정보 추출
            geom_names = self.env.unwrapped.model.geom_names
            geom_pos = self.env.unwrapped.model.geom_pos
            geom_size = self.env.unwrapped.model.geom_size

            bump_infos = []

            for i, name in enumerate(geom_names):
                if "bump" in name:
                    x_center, _, _ = geom_pos[i]
                    half_length, _, height = geom_size[i]

                    bump_start_x = x_center - half_length
                    bump_end_x = x_center + half_length

                    bump_infos.append((bump_start_x, bump_end_x, height))

            # 왼쪽부터 bump 순서로 정렬 (겹치는 범프는 없다고 가정)
            bump_infos.sort(key=lambda x: x[0])

            self.bump_infos = bump_infos
        else:
            self.bump_infos = []

        self.bump_practice = bump_practice
        self.bump_challenge = bump_challenge

        ## change observation space according to the new observation
        obs, _ = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs),), dtype=np.float64
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        custom_obs = self.custom_observation(obs)
        return custom_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        custom_obs = self.custom_observation(obs)
        custom_reward = self.custom_reward(obs, reward)
        custom_terminated = self.custom_terminated(terminated, obs)
        custom_truncated = self.custom_truncated(truncated)
        return custom_obs, custom_reward, custom_terminated, custom_truncated, info

    def custom_terminated(self, terminated, obs):
        # TODO: Implement your own termination condition
        return terminated

    def custom_truncated(self, truncated):
        # TODO: Implement your own truncation condition
        return truncated

    def custom_observation(self, obs):
        # 기본 observation 뒤에 다음과 같은 observation을 추가함
        # 0-17: 기본 observation
        # 8: 바닥 높이 - 현재 torso의 x 좌표 위치의 바닥 높이 (bump 위에 있는 경우의 정보 제공)
        # 9-23: 범프 중심이 torso의 x 좌표 위치보다 오른쪽에 있는 다섯 개의 범프의 시작, 끝, 높이 정보

        base_obs = obs.copy()
        torso_x = obs[0]

        floor_height = 0

        # torso_x가 어떤 범프 위에 있는 경우 그 범프 높이를 floor_height로 설정
        lefts = [left for left, _, _ in self.bump_infos]
        idx = bisect.bisect_right(lefts, torso_x) - 1
        if idx >= 0:
            left, right, height = self.bump_infos[idx]
            if left <= torso_x < right:
                floor_height = height

        # 범프 중심이 torso_x보다 오른쪽에 있는 범프만 추출
        bumps_ahead = [x for x in self.bump_infos if (x[0] + x[1]) > 2 * torso_x]
        bumps_ahead = bumps_ahead[:5]  # 다섯 개만 선택

        while len(bumps_ahead) < 5:
            # 다섯 개가 안 되는 경우 (0, 0, 0)으로 패딩
            bumps_ahead.append((0.0, 0.0, 0.0))

        bump_features = np.array(bumps_ahead).flatten()

        custom_obs = np.concatenate([base_obs, np.array([floor_height]), bump_features])

        return custom_obs

    def custom_reward(self, obs, original_reward):
        x_pos = obs[0]
        x_vel = obs[9]
        z_pos = obs[1]
        torso_angle = obs[2]
        floor_height = obs[18]

        reward = 0.0

        # --- 기본 전진 보상 ---
        reward += x_vel * 1.5

        # --- 균형 보상 ---
        reward -= abs(torso_angle) * 0.2

        # --- 높이 안정성 보상 (전체 torso z 높이 기준)
        if 0.9 <= z_pos <= 2.0:
            reward += 0.5
        else:
            reward -= 0.5

        # target height reward
        target_clearance = 1.25  # base height over terrain
        target_z = floor_height + target_clearance
        height_error = z_pos - target_z

        # dense reward: 쫓아가도록 유도
        reward += np.exp(-3 * height_error**2) * 1.0  # peak=1, falls off quickly
        # 또는 reward -= abs(height_error) * 0.5  # L1 penalty 방식도 가능

        # --- bump 통과 보상 ---
        if hasattr(self, "prev_x"):
            for left, right, height in self.bump_infos:
                if self.prev_x < left <= x_pos:
                    reward += 3.0
        self.prev_x = x_pos

        # --- 기존 reward 일부 반영 ---
        reward += original_reward * 0.1

        return reward


## Test Rendering
if __name__ == "__main__":
    env = CustomEnvWrapper()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs = env.reset()
