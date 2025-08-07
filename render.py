import argparse
import cv2
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_walker2d import CustomEnvWrapper
import gymnasium as gym


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default=None, help="Path to the saved model (.zip)"
)
parser.add_argument(
    "--bump_practice", action="store_true", help="Enable bumping (practice mode)"
)
parser.add_argument(
    "--bump_challenge", action="store_true", help="Enable bumping (challenge mode)"
)
parser.add_argument(
    "--record", action="store_true", help="Enable recording with R key toggle"
)
args = parser.parse_args()

render_mode = "rgb_array" if args.record else "human"
base_env = CustomEnvWrapper(
    render_mode=render_mode,
    bump_practice=args.bump_practice,
    bump_challenge=args.bump_challenge,
)

# DummyVecEnv로 감싸고 normalization 정보 불러오기
vec_env = DummyVecEnv([lambda: base_env])

# 자동으로 VecNormalize 정보 로드
vecnormalize_path = (
    args.model.replace(".zip", "_vecnormalize.pkl") if args.model else None
)
if vecnormalize_path and os.path.exists(vecnormalize_path):
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(f"[Load] VecNormalize loaded from {vecnormalize_path}")
else:
    print(
        f"[Warning] VecNormalize file not found at {vecnormalize_path}. Using unnormalized env."
    )

model = PPO.load(args.model) if args.model is not None else None
obs = vec_env.reset()

video_writer = None
recording = False
frames = []

if args.record:
    print("Recording enabled. Press 'R' to start/stop recording, 'Q' to quit.")

while True:
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
    else:
        action = vec_env.action_space.sample()

    obs, reward, done, info = vec_env.step(action)

    if args.record:
        frame = base_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if recording:
            frames.append(frame_bgr)

        cv2.imshow("Walker2D", frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if not recording:
                print("Recording")
                recording = True
                frames = []
            else:
                print("Recording stopped & saving...")
                recording = False
                if frames:
                    height, width, _ = frames[0].shape
                    file_name = f"recorded_{os.path.splitext(os.path.basename(args.model))[0]}.mp4"
                    video_writer = cv2.VideoWriter(
                        file_name, cv2.VideoWriter_fourcc(*"mp4v"), 60, (width, height)
                    )
                    for f in frames:
                        video_writer.write(f)
                    video_writer.release()
                    print("recorded.mp4 saved successfully")

        if key == ord("q"):
            print("Exiting...")
            break

    if done[0]:
        obs = vec_env.reset()

vec_env.close()
