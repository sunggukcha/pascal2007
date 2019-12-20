dataset="caltech101"
arch="resnet50"
lr=0.01
epochs=50
bs=32
clear

python train.py --dataset $dataset --arch $arch --lr $lr --epochs $epochs --batch-size $bs
