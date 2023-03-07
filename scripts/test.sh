python test.py datamodule=mp datamodule.data_type=image model.arch=resnet18 data_dir=/localdata/hel ckpt_path=logs/experiments/runs/CNN/image-translate_aug2/checkpoints/best.pth name=image-translate_auc2
python test.py datamodule=mp datamodule.data_type=image model.arch=resnet18 data_dir=/localdata/hel ckpt_path=logs/experiments/runs/CNN/image-translate_aug2/checkpoints/checkpoint.pth name=image-translate_auc2

python test.py datamodule=mp datamodule.data_type=render model.arch=resnet18 data_dir=/localdata/hel ckpt_path=logs/experiments/runs/CNN/render-translate_aug/checkpoints/checkpoint.pth name=image-translate_auc2

