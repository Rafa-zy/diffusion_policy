name=sampled_20pc
task_name=pushT
## Image training config
# Transformer
# python train.py --config-dir=config --config-name=image_pusht_diffusion_policy_tf.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# CNN
# python train.py --config-dir=config --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

## LowDim training config
# Transformer
# python train.py --config-dir=config --config-name=lowdim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# python train.py --config-dir=config --config-name=lowdim_pusht_diffusion_policy_tf_big.yaml training.seed=42 training.device=cuda:2 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# python train.py --config-dir=config --config-name=lowdim_pusht_diffusion_policy_cnn_20pc.yaml training.seed=42 training.device=cuda:4 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=config --config-name=lowdim_pusht_dp_tf_20pc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


# CNN
# python train.py --config-dir=config --config-name=lowdim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
