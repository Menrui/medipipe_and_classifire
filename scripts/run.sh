python train.py datamodule=detector
CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=1

python train.py datamodule=hel datamodule.data_type=image model.arch=resnet18 model.pretrain=True num_epochs=30 scheduler.step_size=15 data_dir=/localdata/hel

# python train.py datamodule=hel datamodule.data_type=render model.arch=resnet18 model.pretrain=True
# python train.py datamodule=hel datamodule.data_type=render model.arch=resnet18 model.pretrain=True data_dir=/localdata/hel
python train.py datamodule=hel datamodule.data_type=render model.arch=resnet18 model.pretrain=True num_epochs=30 scheduler.step_size=15 data_dir=/localdata/hel

python train.py datamodule=hel datamodule.data_type=mesh model=mlp num_epochs=30 model.drop_rate=0.2,0.4,0.6 model.mlp_dim=256,512,1024 scheduler.step_size=15 data_dir=/localdata/hel -m
