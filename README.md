# Pytorch Classifire

## Get Started

### conda installation

#### for Unix-like platform

download installer and run the script.

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

For more information, please refer to [miniforge official](https://github.com/conda-forge/miniforge) or [miniconda official](https://docs.conda.io/en/latest/miniconda.html)

### create python environment

```bash
mamba env create -f=environment.yml
```

## Directory Tree

```binary
.
|-- README.md
|-- configs             # コンフィグのデフォルト値
|-- data                # 使用するデータ
|-- environment.yaml    # pythonのパッケージ一覧
|-- logs                # logフォルダ
|-- notebooks           # 一時的な作業用のjupyter notebook
|-- predict.py          # 推論のみを行う場合のスクリプト
|-- scripts             # shellスクリプト
|-- src                 # ソースコード本体
|-- test.py             # テストのみ実行する場合のスクリプト
`-- train.py            # 学習を実行するためのスクリプト
```

## Usage

hydraの基本的な使用方法は[公式ドキュメント](https://hydra.cc/docs/intro/)を参照．
[pytorch-hydra-template](https://github.com/ashleve/lightning-hydra-template)を参考に作成しているため，基本的な使い方は類似している．

### 学習

```bash
python train.py datamodule=mp datamodule.data_type=image model.arch=resnet18 model.pretrain=True num_epochs=30 scheduler.step_size=15
```

### Argument

| Argument | type | status | default | discription |
| --- | --- | --- | --- | --- |
| datamodule | string | Optional | mp | [configs/datamodule](configs/datamodule) |
| model | string | Optional | resnet | [configs/model](configs/model) |
| optimizer | string | Optional | adam | [configs/optimizer](configs/optimizer) |
| scheduler | string | Optional | steplr | [configs/scheduler](configs/scheduler) |
| log_dir | string | Optional | default |  [configs/log_dir](configs/log_dir) |
| data_dir | string | Optional | ./data | データセットが置いてあるディレクトリのパス（データセットの親ディレクトリ） |
| seed | int | Optional | 9999 | 乱数シード値 |
| name | string | Optional | "default" | 実験ログのディレクトリ名 |
| train | bool | Optional | True | trainを実行するかどうか |
| test | bool | Optional | True | testを実行するかどうか |
| num_epochs | int | Optional | 3 | 学習エポック数 |

コンフィグ内の各パラメータの説明は各yamlファイルに記載．
```datamodule=mp```を指定すると，[configs/datamodule/mp.yaml](configs/datamodule/mp.yaml)に記載されたパラメータが学習時にオーバーライドされる．
使用するスケジューラーや最適化関数を変更したい場合には，[configs/scheduler](configs/scheduler)や[configs/optimizer](configs/optimizer)以下に適切にコンフィグを追加して引数で指定する．

### データについて

データを取り扱うのは[src/datamodules/components/mp_dataset.py](src/datamodules/components/mp_dataset.py)のMPDatasetクラスとなっている．
データを追加する場合は[data/photographed_data](data/photographed_data)以下に人物フォルダを追加し，アノテーションデータを[data/photographed_data/processed_data_table.csv](data/photographed_data/processed_data_table.csv)に追加し，MPDatasetクラスのhuman_ids引数を適切に書き換える．
