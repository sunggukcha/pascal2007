arch="resnet50"
lr=0.01
epochs=10
bs=32
clear

python train.py --arch $arch --lr $lr --epochs $epochs --batch-size $bs
