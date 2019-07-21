# STA module for Action Recognition and Action Detection

## Introduction
This codes based on https://github.com/kenshohara/3D-ResNets-PyTorch. 
The corresponding paper is "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?, CVPR 2018".

## Preparation
Please ref the installation steps in https://github.com/kenshohara/3D-ResNets-PyTorch.

## Train
```
# CUDA_VISIBLE_DEVICES=0,1   python main.py --video_path $DATASET_PATH/data_30/hmdb51 --annotation_path $DATASET_PATH/data_30/hmdb51_1.json --result_path results_hmdb51-64f/train --dataset hmdb51 --model resnext_fa --model_depth 101 --n_classes 51 --n_finetune_classes 51 --batch_size 10 --n_threads 32 --checkpoint 1  --pretrain_path models/resnext-101-64f-kinetics-hmdb51_split1.pth --resnet_shortcut B --resnext_cardinality 32   --n_epochs 3000 --sample_duration 64 --learning_rate 0.0001 --n_val_samples 1 2>&1 |tee results_hmdb51-64f/train/hmdb51-64f_training.log
```
## Test
```
# results_hmdb51_path=results_hmdb51-64f;
# CUDA_VISIBLE_DEVICES=0 python  main.py --no_train --no_val --test --video_path $DATASET_PATH/data_30/hmdb51 --annotation_path $DATASET_PATH/data_30/hmdb51_1.json --result_path $results_hmdb51_path --test_result_path $results_hmdb51_path/test --dataset hmdb51 --model resnext --model_depth 101 --n_classes 51 --n_finetune_classes 51 --batch_size 80 --n_threads 4 --resume models/hmdb51-64f-trained-well-model.pth --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 > tmp.log
# cd utils/
# python eval.py
```
