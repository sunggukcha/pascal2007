lr=0.01
epochs=100
bs=4
clear

python train.py --lr $lr --epochs $epochs --batch-size $bs
