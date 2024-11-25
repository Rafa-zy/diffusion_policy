"""
Usage:
python sample_trajectory.py --data_path data/pusht/pusht_cchi_v7_replay.zarr -o data/sampled_pusht.zarr
"""
import click
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.option('-dp', '--data_path', required=True, type=str)
@click.option('-o', '--output', required=True, type=str)
@click.option('-sr', '--sample_rate', default=0.2, type=float)
@click.option('-sd', '--seed', default=42, type=int)
def main(data_path, output, sample_rate, seed):
    replay_buffer = ReplayBuffer.copy_from_path(data_path)
    new_replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    n_sampled_episodes = int(replay_buffer.n_episodes * sample_rate)

    np.random.seed(seed)
    sampled_indices = np.random.choice(replay_buffer.n_episodes, n_sampled_episodes, replace=False)
    for idx in sampled_indices:
        episode = replay_buffer.get_episode(idx)
        new_replay_buffer.add_episode(episode, compressors='disk')
    
    print(f"successfully sampled {n_sampled_episodes} episodes")
    print(f"saved to {output}")


if __name__ == "__main__":
    main()