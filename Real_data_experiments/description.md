### Summary
This folder contains running codes for experiments on camelyon17, cifar10, cinic10 (imagenet vs cifar10) and svhn vs mnist.

## Notes

- For using CINIC-10 dataset. We refer to https://github.com/BayesWatch/cinic-10 where the original datasets can be downloaded and cifar10, imagenet can be splited into training and testing datasets following instructions therein.
- For using Camelyon17 dataset. We refer to https://github.com/p-lambda/wilds where they provide necessary tools for downloading and using Camelyon17 dataset.
- Once the datasets are ready, for complex image classification problems. Please follow https://github.com/facebookresearch/moco/tree/KaimingHe-patch-1?tab=coc-ov-file and run `main_moco.py` to train self-supervised model for extracting meaningful representations of image data. Utilizing self-supervised models improves the performance of supervised training and the low dimension representations are better suited for subsampling algorithm such as SimSRT rather than the raw input of image data.
- `Diabetes_and_Vessel.R` contains code for reproducing experiments on Diabetes dataset and Vessel dataset. We use h2o package for modeling Diabetes data with GBM (Gradient Boosting Machine) and modeling Vessel data with XGBoost (Extreme Gradient Boosting Tree). 

## Steps

- After self-supervised learning stage, our proposed method, SimSRT works directly on the features extracted by self-supervised models.
- See `SimSRT.ipynb` for details of implementing SimSRT. The function `uniform_subsampling` within returns the indices of uniform subsample. Save these indices and pass them for later robust training using `main_cls_camelyon17.py` for Camelyon17 dataset.
- Run `main_cls_camelyon17.py` (or other .py files for supervised training on different datasets) and performance results (test accuracy) are automatically reported.

## Default running command

# Train ResNet-50 on Camelyon17 with pretrained weights and robust subsample of size 10000 (5000 for random selection and 5000 for uniform selection). Indices file saved by SimSRT.ipynb should be pass to --robust-indices so that it can integrate with randomly selected samples.
```
python main_cls_camelyon17.py -a resnet50 --lr 0.001 --schedule 5 9  --batch-size 32 --robust-size 32 --subsetsize 5000 --epochs 10 --pretrained checkpoint_best_camelyon17.pth.tar --dist-url 'tcp://localhost:10001' --print-freq 300 --world-size 1 --rank 0 --learning-mode subset --robust-indices camelyon17_5000_unique.csv   --weight-decay 0  --rho 1 --save_dir record/camelyon17_10000/
```
# Train ResNet-18 on rotated CIFAR-10
```
python main_cls_cifar10_rotate_robust.py -a resnet18 --lr 0.0005 --schedule 30 45 --batch-size 256 --print-freq 100 --subsetsize 10000 --epochs 50 --pretrained checkpoint_best_cifar10.pth.tar --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 --learning-mode subset --robust-indices cifar10_rotated_10000_unique.csv --degree 90 --rho 1 cifar/
```

# Train ResNet-18 on CINIC-10 (Imagenet vs CIFAR10)
```
python main_cls_cinic-10-imagenet.py -a resnet18 --lr 0.002 --schedule 15 30 --batch-size 256 --robust-size 16 --subsetsize 25000 --epochs 35 --pretrained checkpoint_best_cinic_imagenet.pth.tar --dist-url 'tcp://localhost:10001' --print-freq 600 --world-size 1 --rank 0 --learning-mode subset --cosine --robust-indices imagenet_25000_unique.csv --rho 1 cifar/
```

# Train ResNet-18 on SVHN and test it on MNIST
```
python main_cls_svhn_mnist.py -a resnet18 --lr 0.001 --schedule 10 20 --batch-size 512 --robust-size 256 --subsetsize 25000 --epochs 30 --pretrained checkpoint_best_svhn.pth.tar --dist-url 'tcp://localhost:10001' --print-freq 300 --world-size 1 --rank 0 --learning-mode full --robust-indices svhn_25000_unique.csv  --cosine  --rho 1 --save_dir record/svhn_50000.csv cifar/
```


