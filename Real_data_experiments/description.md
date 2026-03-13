### Summary
This folder contains running codes for experiments on camelyon17, cifar10, cinic10 (imagenet vs cifar10) and svhn vs mnist.

## Notes

- For using CINIC-10 dataset. We refer to https://github.com/BayesWatch/cinic-10 where the original datasets can be downloaded and cifar10, imagenet can be splited into training and testing datasets following instructions therein.
- For using Camelyon17 dataset. We refer to https://github.com/p-lambda/wilds where they provide necessary tools for downloading and using Camelyon17 dataset.
- Once the datasets are ready, for complex image classification problems. Please follow https://github.com/facebookresearch/moco/tree/KaimingHe-patch-1?tab=coc-ov-file and run `main_moco.py` to train self-supervised model for extracting meaningful representations of image data. Utilizing self-supervised models improves the performance of supervised training and the low dimension representations are better suited for subsampling algorithm such as SimSRT rather than the raw input of image data.

## Steps

- After self-supervised learning stage, our proposed method, SimSRT works directly on the features extracted by self-supervised models.
- See `SimSRT.ipynb` for details of implementing SimSRT. The function `uniform_subsampling` within returns the indices of uniform subsample. Save these indices and pass them for later robust training using `main_cls_camelyon17.py` for Camelyon17 dataset.
- Run `main_cls_camelyon17.py` (or other .py files for supervised training on different datasets) and performance results (test accuracy) are automatically reported.


