#!/bin/bash

MODEL=VAWGAN
CPKT=logdir/train/VAWGAN_S2018/model.ckpt-196230
OPDIR="./logdir/$MODEL"
LONGOPTIONS=model:,checkpoint:,output_dir:

# -temporarily store output to be able to check for errors
# -e.g. use “--options” parameter by name to activate quoting/enhanced mode
# -pass arguments only via   -- "$@"   to separate them correctly
PARSED=$(getopt --options="" --longoptions="$LONGOPTIONS" --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    # e.g. $? == 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CPKT="$2"
            shift 2
            ;;
        --output_dir)
            OPDIR="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

echo "Model: $MODEL; Checkpoint: $CPKT; Output_dir: $OPDIR"

source activate vcwgan

for src in VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4; do
    for trg in VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2; do
        python convert.py --src "$src" --trg "$trg" --model "$MODEL" --checkpoint "$CPKT" \
        --file_pattern "./dataset/vcc2018/bin/Training Set/{}/[0-9]*.bin" --output_dir "$OPDIR"
    done
done

source deactivate
