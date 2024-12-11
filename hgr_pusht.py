import numpy as np
import click
import hydra
import dill
import torch
import cv2
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply
import skimage.transform as st
from tqdm import tqdm

def images_to_video(image_list, output_path, fps=10):
    # Get the dimensions of the images
    height, width, layers = image_list[0].shape
    size = (width, height)
    
    # Initialize the video writer
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for img in image_list:
        # Convert the image from RGB to BGR format
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    
    out.release()

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-c', '--checkpoint', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-ntraj', '--num_trajs_per_shape', type=int, default=25)
@click.option('-rthres', '--reward_thres', type=float, default=0.9)
def main(output, checkpoint, device, num_trajs_per_shape, reward_thres):
    """
    Collect hindsight relabeled data for the Push-T task.
    
    Usage: python hgr_pusht.py -o data/hgr_pusht.zarr -c data/outputs/checkpoints/last.ckpt
    """
    
    # create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output, mode='a')

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy: BaseLowdimPolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # create PushT env with keypoints
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
    env = PushTKeypointsEnv(
        legacy=cfg.task.env_runner.legacy_test,
        keypoint_visible_rate=cfg.task.env_runner.keypoint_visible_rate,
        agent_keypoints=cfg.task.env_runner.agent_keypoints,
        **kp_kwargs
    )
    menv = MultiStepWrapper(
        PushTKeypointsEnv(
            legacy=cfg.task.env_runner.legacy_test,
            keypoint_visible_rate=cfg.task.env_runner.keypoint_visible_rate,
            agent_keypoints=cfg.task.env_runner.agent_keypoints,
            **kp_kwargs
        ),
        n_obs_steps=cfg.task.env_runner.n_obs_steps + cfg.task.env_runner.n_latency_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        max_episode_steps=cfg.task.env_runner.max_steps,
    )
    n_obs_steps = cfg.task.env_runner.n_obs_steps
    n_latency_steps = cfg.task.env_runner.n_latency_steps
    use_past_action = cfg.task.env_runner.past_action
    max_traj_per_shape = 100
    save_video = False
    
    # episode-level while loop
    for shape in ["tee", "vee", "al", "gamma"]:
        n_sampled_traj = 0
        for seed in tqdm(range(max_traj_per_shape), desc='Taking action...'):
            imgs_old = []
            episode = list()
            # set seed for env
            env.seed(seed)
            env.set_block_shape(shape)
            menv.seed(seed)
            menv.env.set_block_shape(shape)
            # reset env and get observations (including info and render for recording)
            obs = env.reset()
            info = env._get_info()
            img = env.render(mode='rgb_array')
            imgs_old.append(img)
            mobs = menv.reset() # dual obs from multistep environment as policy input
            mobs = np.expand_dims(mobs, axis=0)
            past_action = None
            policy.reset()
            done = False
            # step-level while loop
            while not done:
                Do = mobs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': mobs[...,:n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': mobs[...,:n_obs_steps,Do:] > 0.5
                }
                if use_past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(n_obs_steps-1):].astype(np.float32)
                
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                
                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][0,n_latency_steps:]

                for aid in range(action.shape[0]):
                    act = action[aid]
                    # state dim 2+3
                    state = np.concatenate([info['pos_agent'], info['block_pose']])
                    data = {
                        'state': state,
                        'action': act,
                    }
                    episode.append(data)
                    # step env and render
                    obs, reward, done, info = env.step(act)
                    img = env.render(mode='rgb_array')
                    imgs_old.append(img)
                
                # step multistep env
                mobs, _, mdone, _ = menv.step(action)
                mobs = np.expand_dims(mobs, axis=0)
                past_action = action
                if mdone:
                    break

            # save episode buffer to replay buffer (on disk)
            print("old reward", reward)
            # save old images as a video file
            if save_video:
                video_path = f"debug_old_{shape}_{reward:.4f}_{seed}.mp4"
                images_to_video(imgs_old, video_path)

            # disable the requirement of a middle valued initial reward
            if (reward <= reward_thres and reward > 0.2) or True:
                block_pose = info['block_pose']
                goal_pose = info['goal_pose']
                translation = goal_pose[:2] - block_pose[:2]
                angle = goal_pose[2] - block_pose[2]
                rotation_tf = st.AffineTransform(rotation=angle)
                valid = True

                # First transform the initial position of the block
                env.seed(seed)
                env.set_block_shape(shape)
                obs = env.reset()
                init_info = env._get_info()
                block_pose = init_info['block_pose']
                block_pose[:2] = block_pose[:2] + translation
                block_pose[:2] = block_pose[:2] - goal_pose[:2]
                block_pose[:2] = rotation_tf(block_pose[:2])[0] + goal_pose[:2]
                block_pose[2] = block_pose[2] + angle
                
                agent_pose = init_info['pos_agent']
                agent_pose = agent_pose + translation
                agent_pose = agent_pose - goal_pose[:2]
                agent_pose = rotation_tf(agent_pose)[0] + goal_pose[:2]
                if not env.action_space.contains(agent_pose):
                    print(f"Invalid agent pose: {agent_pose}")
                    valid = False

                # Then transform the position of every action
                if valid:
                    for x in episode:
                        old_action = x['action']
                        old_action = old_action + translation
                        old_action = old_action - goal_pose[:2]
                        new_action = rotation_tf(old_action)[0] + goal_pose[:2]
                        if not env.action_space.contains(new_action):
                            valid = False
                            print(f"Invalid action: {new_action}")
                            break
                        x['action'] = new_action
                
                if valid:
                    imgs_new = []
                    env._set_state(np.concatenate([agent_pose, block_pose]))
                    obs = env._get_obs()
                    info = env._get_info()
                    img = env.render(mode='rgb_array')
                    imgs_new.append(img)
                    done = False
                    relabeled_episode = list()
                    for x in episode:
                        act = x['action']
                        # state dim 2+3
                        state = np.concatenate([info['pos_agent'], info['block_pose']])
                        # discard unused information such as visibility mask and agent pos
                        # for compatibility
                        keypoint = obs.reshape(2,-1)[0].reshape(-1,2)[:9]
                        data = {
                            'img': img,
                            'state': np.float32(state),
                            'keypoint': np.float32(keypoint),
                            'action': np.float32(act),
                            'n_contacts': np.float32([info['n_contacts']])
                        }
                        relabeled_episode.append(data)
                        obs, reward, done, info = env.step(act)
                        img = env.render(mode='rgb_array')
                        imgs_new.append(img)
                        if done:
                            break
                    print("new reward", reward)
                    # save new images as a video file
                    if save_video:
                        video_path = f"debug_new_{shape}_{reward:.4f}_{seed}.mp4"
                        images_to_video(imgs_new, video_path)
                    if reward >= 0.95:
                        n_sampled_traj += 1
                        print(f"Sampled {n_sampled_traj} trajectories")
                        data_dict = dict()
                        for key in relabeled_episode[0].keys():
                            data_dict[key] = np.stack(
                                [x[key] for x in relabeled_episode])
                        replay_buffer.add_episode(data_dict, compressors='disk')
                        if n_sampled_traj >= num_trajs_per_shape:
                            break


if __name__ == "__main__":
    main()
