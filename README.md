# VC-with-GAN
CS 753 ASR project


# Usage Steps:
1. Run `bash download.sh` to prepare the VCC2018 dataset.
2. Run `analyzer.py` to extract features and write features into binary files. (This takes a few minutes.)
3. Run `build.py` to record some stats, such as spectral extrema and pitch.
4. To train a VAWGAN, for example, run
```bash
python main.py \
--model VAWGAN \
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
--checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
--file_pattern "./dataset/vcc2018/bin/Training Set/{}/[0-9]*.bin"
```  
*Please fill in `timestamp` and `model id`.  
7. You can find the converted wav files in `./logdir/output/[timestamp]`  
8. If you want to convert all the voices, run
```bash
./convert_all.sh \
--model VAWGAN \
--checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
--output_dir [directory to store converted audio]
```  

# Usage for Sentence Embeddings:
1. Ensure you have `w_prob_dict.pkl` and `w_vec_dict.pkl` in `data` directory. 
    1. For `w_prob_dict.pkl` you have two options. Either use `get_word_prob_from_corpus` this demands a corpus as an input. We used WikiText. Or you can get a csv file with unigram probabilities (we mentioned the source in the report http://norvig.com/ngrams/) and use the function get_w_prob_from_csv.
    2. For `w_vec_dict.pkl initialize` a Sentence_Embedding object and then call the function `prune_word_vec`. This essentially keeps only those embeddings which are present in the transcriptions since it takes a lot more time (and ram) to get the parse the whole fasttext data.
    3. All pickle files are shared here https://drive.google.com/drive/folders/1FWGGEQ9wTUewBDFq5ssT4BP4cMyt8lh1


2. Download the dataset using `bash download.sh`
3. Run `python sentence_embedding.py`. This should create `sent_emb.pkl` inside `data` directory.
4. Run `analyzer.py` to extract features, store them along with sentence embeddings.
5. Run `build.py` to find statistics about features.
5. To train with sentence embedding, run
```bash
python main.py \
--model VAWGAN_S \
--trainer VAWGAN_S \
--architecture architecture-vawgan-sent.json
```
5. For conversion, run
```bash
python convert.py \
--src VCC2SF1 \
--trg VCC2TM1 \
--model VAWGAN_S \
--checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
--file_pattern "./dataset/vcc2018/bin/Training Set/{}/[0-9]*.bin"
```
or
```bash
./convert_all.sh \
--model VAWGAN_S \
--checkpoint logdir/train/[timestamp]/[model.ckpt-[id]] \
--output_dir [directory to store converted audio]
```
