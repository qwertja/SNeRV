# SNeRV

This is the repository of [SNeRV: Spectra-preserving Neural Representation for Video]() - ECCV 2024


## Get started
* Prepare environment
```
pip install -r requirements.txt 
```

* Prepare dataset files at [data/](./data)

## Examples
### SNeRV
```
CUDA_VISIBLE_DEVICES=0 python train_snerv.py  --outf exp_1 --data_path  data/bunny --vid bunny \
    --model snerv \
    --enc_strds 5 4 2 2 2 --dec_strds 5 4 2 2 2 \
    --num_blocks 6 --fc_dim 111 \
    --crop_list 640_1280 --data_split 1_1_1 --eval_freq 50 -e 300 \
    --lr 0.001 --suffix SNeRV --emb_size 0
```

### Temporal extension of SNeRV
```
CUDA_VISIBLE_DEVICES=0 python train_snerv_t.py  --outf exp_1 --data_path  data/bunny --vid bunny \
    --model snerv_t \
    --enc_strds 5 4 2 2 2 --dec_strds 5 4 2 2 2 \
    --num_blocks 6 --fc_dim 103 \
    --crop_list 640_1280 --data_split 1_1_1 --eval_freq 50 -e 300 \
    --lr 0.001 --suffix SNeRV_T --emb_size 20
```
* We also provide SNeRV_T(2D) model as ```--model snerv_t_2d```

## Acknowledgement
The implementation is based on [HNeRV](https://github.com/haochen-rye/HNeRV).

## Citation
```
```
