# 2. SeqGAN using pokewiki data

## Running the tests
1. Create directory `embed` and Download the `ko.tsv` file from the link below to the directory.
    - [Pretrained korean word2vec](https://drive.google.com/open?id=0B0ZXk88koS2KbDhXdWg1Q2RydlU)
    
2. Create directory `data` and Move `pk_data_g*.csv` files into the directory.
    - .csv files are created `0_web_crawler`

3. Run the code in the following order:
    1. load_embed.py
    2. preprocess_util.py
    3. preprocess_data.py
    4. sequence_gan.py
    
## Reference
- Pretrained word2vec : https://github.com/Kyubyong/wordvectors