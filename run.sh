name=debug
task_name=pushT
## Image training config
# Transformer
# python train.py --config-dir=. --config-name=config/image_pusht_diffusion_policy_tf.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# CNN
# python train.py --config-dir=. --config-name=config/image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

## LowDim training config
# Transformer
python train.py --config-dir=. --config-name=config/lowdim_pusht_diffusion_policy_tf.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# CNN
python train.py --config-dir=. --config-name=config/lowdim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
