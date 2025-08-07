from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from custom_walker2d import CustomEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
import os
import argparse

# 학습 환경 개수
N_ENVS = 4


# 사용자 정의 체크포인트 콜백 (모델 + VecNormalize 동시 저장)
class NormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step = self.n_calls
            model_path = os.path.join(self.save_path, f"walker_model_{step}_steps.zip")
            vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")

            self.model.save(model_path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(vecnorm_path)

            if self.verbose > 0:
                print(f"[Checkpoint] Saved: {model_path}, {vecnorm_path}")
        return True


def make_env(bump_practice=False, bump_challenge=False):
    def _init():
        return CustomEnvWrapper(
            render_mode=None, bump_practice=bump_practice, bump_challenge=bump_challenge
        )

    return _init


# 네트워크 구조 정의
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])], log_std_init=-1.0
)

# 커맨드라인 인자
parser = argparse.ArgumentParser()
parser.add_argument(
    "--bump_practice", action="store_true", help="Enable bumping (practice mode)"
)
parser.add_argument(
    "--bump_challenge", action="store_true", help="Enable bumping (challenge mode)"
)
parser.add_argument(
    "--continue_from",
    type=str,
    default=None,
    help="Path to previous model checkpoint (.zip)",
)
parser.add_argument(
    "--resume_timesteps",
    type=int,
    default=10_000_000_000,
    help="Timesteps to train (total)",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=N_ENVS,
    help="Number of CPUs to use",
)
args = parser.parse_args()

if args.bump_practice:
    folder_name = "bump_practice"
elif args.bump_challenge:
    folder_name = "bump_challenge"
else:
    folder_name = "walker_model"

save_path = f"./checkpoints/{folder_name}/"

num_cpu = args.n_cpu
env = SubprocVecEnv(
    [make_env(args.bump_practice, args.bump_challenge) for _ in range(num_cpu)]
)
env = VecMonitor(env)

if args.continue_from:
    vecnorm_path = args.continue_from.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        print(f"[Load] Loaded VecNormalize from {vecnorm_path}")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        print(f"[Warning] VecNormalize not found at {vecnorm_path}, starting fresh.")
    model = PPO.load(args.continue_from, env)
    print(f"[Load] Loaded model from {args.continue_from}")
else:
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        policy_kwargs=policy_kwargs,
        device="cpu",
        learning_rate=0.0001,
    )
    print("[Init] New model initialized.")


# 커스텀 체크포인트 콜백 사용
checkpoint_callback = NormalizeCheckpointCallback(
    save_freq=100000, save_path=save_path, verbose=1
)

model.learn(total_timesteps=args.resume_timesteps, callback=checkpoint_callback)
