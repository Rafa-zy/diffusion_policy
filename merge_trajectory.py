"""
Usage:
python merge_trajectory.py --data_patha data/pusht/pusht_cchi_v7_replay.zarr --data_pathb data/hgr_pusht_1000epoch_base_20.zarr -o data/hybrid_pusht_error_kp.zarr
"""
import click
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.option('-dpa', '--data_patha', required=True, type=str)
@click.option('-dpb', '--data_pathb', required=True, type=str)
@click.option('-o', '--output', required=True, type=str)
def main(data_patha, data_pathb, output):
    replay_buffer = ReplayBuffer.copy_from_path(data_patha)
    replay_buffer_b = ReplayBuffer.copy_from_path(data_pathb)
    new_replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    for idx in range(replay_buffer.n_episodes):
        episode = replay_buffer.get_episode(idx)
        new_replay_buffer.add_episode(episode, compressors='disk')
    
    for idx in range(replay_buffer_b.n_episodes):
        episode = replay_buffer_b.get_episode(idx)
        new_replay_buffer.add_episode(episode, compressors='disk')

    print(f"saved to {output}")


if __name__ == "__main__":
    main()