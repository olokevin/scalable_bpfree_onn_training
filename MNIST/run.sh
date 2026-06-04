### FC layers
CUDA_VISIBLE_DEVICES=0 python -u main.py -config configs/dnn.yml
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -config configs/dnn.yml >/dev/null 2>&1 &

### TensorTrian layers
CUDA_VISIBLE_DEVICES=0 python -u main.py -config configs/tt.yml
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -config configs/tt.yml >/dev/null 2>&1 &

### ONN
CUDA_VISIBLE_DEVICES=0 python -u main.py -config configs/onn.yml
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -config configs/onn.yml >/dev/null 2>&1 &

### TONN
CUDA_VISIBLE_DEVICES=0 python -u main.py -config configs/tonn.yml
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py -config configs/tonn.yml >/dev/null 2>&1 &