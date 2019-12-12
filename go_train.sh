dataset="embedding"
arch="resnet18"
lr=0.01
epochs=50
bs=32
clear

python train.py --dataset "embedding" --arch $arch --lr $lr --epochs $epochs --batch-size $bs
