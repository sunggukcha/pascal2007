# Image classification script for Python/PyTorch

### Supported dataset
1. caltech101 for classification (http://www.vision.caltech.edu/Image_Datasets/Caltech101/)


### How to use

1. Install requirements
2. Edit configurations in (a) mypath.py, (b) go_train.sh.


### Additional Notes

1. Selective-layer-wise freezing in training available.
2. tucker decomposition for freezed conv layer in training time is available. (for tucker decomposition implementation, I referred https://github.com/jacobgil/pytorch-tensor-decompositions)

For more details about configurations, please refer arguments(args) in train.py.

###
