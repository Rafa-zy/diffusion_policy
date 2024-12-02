name=sampled_20pc
task_name=pushT
export CUDA_VISIBLE_DEVICES=1,2,3  
ray start --head --num-gpus=3
## Image training config
# Transformer
# python train.py --config-dir=config --config-name=image_pusht_diffusion_policy_tf.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python ray_train_multirun.py --config-dir=config --config-name=image_pusht_diffusion_policy_tf.yaml --seeds=42,43,44 --monitor_key=test/mean_score -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'
# CNN
# python train.py --config-dir=config --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

## LowDim training config
# Transformer
# python train.py --config-dir=config --config-name=lowdim_pusht_dp_tf_20pc training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# CNN
# python train.py --config-dir=config --config-name=lowdim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
