lr=0.01
epochs=100
bs=128
resume="./run/caltech101/resnet50/experiment_0/checkpoint.pth.tar"
clear

python train.py --lr $lr --epochs $epochs --batch-size $bs --test --resume $resume
