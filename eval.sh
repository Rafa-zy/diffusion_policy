# python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output --device cuda:0
# python eval.py --checkpoint /mnt/diffusion_policy/data/outputs/2024.11.25/00.43.25_train_diffusion_unet_hybrid_pusht_image/checkpoints/epoch=0100-test_mean_score=0.804.ckpt --output_dir data/pusht_eval_output --device cuda:0
# python eval.py --checkpoint /mnt/diffusion_policy/data/outputs/2024.11.25/02.30.25_train_diffusion_unet_hybrid_pusht_image/checkpoints/epoch=0350-test_mean_score=0.534.ckpt --output_dir data/pusht_eval_output --device cuda:0
#python eval.py --checkpoint /local2/xingcheng/diffusion_policy/data/outputs/2024.11.25/00.19.26_train_diffusion_transformer_lowdim_pusht_lowdim/checkpoints/epoch=0100-test_mean_score=0.237.ckpt --output_dir data/pusht_eval_output --device cuda:0
python eval.py --checkpoint /ssd/fan/diffusion_policy/data/outputs/2024.12.02/edm_lowdim_pusht_lowdim/checkpoints/epoch=2950-test_mean_score=0.756.ckpt --output_dir data/pusht_eval_output --device cuda:0