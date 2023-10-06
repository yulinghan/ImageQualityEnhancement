python3 ./train/train_denoise.py --arch Uformer_B --batch_size 2 --gpu '0' \
    --train_ps 128 --train_dir datasheet/sidd/train --env _0706 \
    --val_dir datasheet/sidd/val --save_dir ./logs/ \
    --dataset sidd --warmup 
