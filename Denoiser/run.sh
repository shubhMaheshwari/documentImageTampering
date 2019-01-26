
source activate py36
module add cuda/8.0
module add cudnn/6-cuda-8.0
python3.6 -m visdom.server &
python3.6 train.py
