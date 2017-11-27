#!/bin/bash
MODEL=ConvVAE
CPKT=logdir/train/VAE2018/model.ckpt-42586
OPDIR="./logdir/$MODEL"

source activate vcwgan

for src in VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4; do
    for trg in VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2; do
        python convert.py --src "$src" --trg "$trg" --model "$MODEL" --checkpoint "$CPKT" \
        --file_pattern "./dataset/vcc2018/bin/Training Set/{}/*.bin" --output_dir "$OPDIR"
    done
done

source deactivate
