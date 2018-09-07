# 4. TFIDF Keyword Extraction & Sentence Creation by Type

## Running the tests
1. Move `pk_type_dic.pkl` and `pk_pos_dict.pkl` files into the directory `data`.
    - Above files are created from `preprocess_data.py` and `preprocess_util.py` of `2_seqgan`

2. Set the range of keyword in `tfidf_extract.py` of `4_tfidf`.
    - The range is the number of keywords extracted from TFIDF model.

3. Run the code `sequence_gan_load_test.py` of `4_tfidf`.