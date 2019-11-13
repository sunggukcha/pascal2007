clear

logdir='./run/caltech101/resnet18/'

echo "tensorboard summary @$logdir"

tensorboard --logdir $logdir --port 5005
