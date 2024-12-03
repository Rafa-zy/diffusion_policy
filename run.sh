export CUDA_VISIBLE_DEVICES=2
name=diffusion_policy_tf_hgr_20pc   #sampled_20pc
task_name=pushT
## Image training config
# Transformer
# python train.py --config-dir=config --config-name=image_pusht_dp_tf_20pc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
python train.py --config-dir=config --config-name=image_pusht_dp_tf_hgr_20pc.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# CNN
# python train.py --config-dir=config --config-name=image_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

## LowDim training config
# Transformer
# python train.py --config-dir=config --config-name=lowdim_pusht_dp_tf_20pc training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# CNN
# python train.py --config-dir=config --config-name=lowdim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
