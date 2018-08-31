# 3. Conditional SeqGAN using pokewiki data

## Running the tests
  
1. Create directory `data` and Move files from `2_seqgan` directory to `data` directory.
    - Files are `pos2idx.pkl`, `idx2pos.pkl`, `pretrian_embedding_vec.pkl`.
    
2. Move `type_dict.pickle` file into the `data` directory.
    - `type_dict.pickle` is created `load_crawling_data.py` of `0_web_crawler`

3. Run the code in the following order:
    1. preprocess_util.py
    2. preprocess_data.py
    3. sequence_gan.py
    
## Reference
- Pretrained word2vec : https://github.com/Kyubyong/wordvectors