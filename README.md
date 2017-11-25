# VC-with-GAN
CS 753 ASR project

# Usage
1. Run `bash download.sh` to prepare the VCC2016 dataset.  
2. Run `analyzer.py` to extract features and write features into binary files. (This takes a few minutes.)  
3. Run `build.py` to record some stats, such as spectral extrema and pitch.  
4. To train a VAWGAN, for example, run
```bash
python main.py \
--model VAWGAN \
--model_module model.vawgan \
--trainer VAWGANTrainer \
--architecture architecture-vawgan-vcc2016.json
```  
5. You can find your models in `./logdir/train/[timestamp]`  
6. To convert the voice, run
```bash
python convert.py \
--src VCC2SF1 \
--trg VCC2TM1 \
--model VAWGAN \
--model_module model.vawgan \
--checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
--file_pattern "./dataset/vcc2016/bin/Testing Set/{}/*.bin"
```  
*Please fill in `timestamp` and `model id`.  
7. You can find the converted wav files in `./logdir/output/[timestamp]`  

