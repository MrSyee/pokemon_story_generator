import glob
import pandas as pd
import numpy as np
import re
from konlpy.tag import Kkma
from konlpy.utils import pprint
import pickle
import os
import csv

path = './data/'
filenames = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in filenames:
    df = pd.read_csv(file_, index_col=None)
    list_.append(df)
frame = pd.concat(list_)


def _save_pickle(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()

# exchange type1 and type2 and make new_type

# convert 비행 and 노말
def Type(type1, type2):
    if type1 == '노말' and type2 == type2:
        return type2
    elif type2 == '비행':
        return '비행'
    else:
        return type1

# convert jsut 비행
def Type2(type1, type2):
    if type2 == '비행':
        return '비행'
    else:
        return type1
    

frame['new_type'] = frame[['type1','type2']].apply(lambda x: Type(x['type1'], x['type2']), axis=1)
# frame.head(10)



poke_type_list = list(frame['new_type'].value_counts().index)

# 타입, 타입에 따른 전체 포켓몬, 사용되는 포켓몬, 문장수, 제외시킨 포켓몬 수
info_data = pd.DataFrame(columns=('type', 'poke_cnt', 'useful_poke', 'sentence_cnt', 'extract_poke'))

special_data = [] # extract poke_name, sentence # 제외시킨 데이터 저장
type_dict = dict()

for poke_type in poke_type_list:
    new_data = []
    poke_name = []
    etc_poke = []
    print("....devide by {} type poketmon ...".format(poke_type))
    print("poketmon counts....{}".format(len(frame[frame['new_type']==poke_type])))
    
    # get desc by type
    print(".... get desc by {} type....".format(poke_type))
    for desc in frame.loc[frame['new_type']== poke_type, ['name', 'desc']].values.tolist():
        if '(' in desc[1]:
            special_data += desc
            etc_poke.append(desc[0])
        else:
            poke_name.append(desc[0])
            new_data += re.sub(r"[^ㄱ-힣a-zA-Z0-9.]+", ' ', desc[1]).strip().split('.')
            
    # make uniq sentence
    print("....extract uniq sentence....")
    new_data = list(set(new_data))
    new_data.remove('')
    print("Number of sentence .... {}".format(len(new_data)))
    
    # to word to pos
    print("....convert to pos....")
    kkma = Kkma()
    pos_sentence = []
    for sentence in new_data:
        pos_sentence += [kkma.pos(sentence)]
    
    print("....add dict....")
    # add to dict
    type_dict[poke_type] = pos_sentence
    
    
    # make info data
    info_data = info_data.append([{'type': poke_type, 
                                   'poke_cnt': len(frame[frame['new_type']==poke_type]),
                                    'useful_poke': len(poke_name),
                                     'sentence_cnt': len(new_data),
                                     'extract_poke': len(etc_poke)}], ignore_index=True)
    
            
#     break


DATA_PATH = "./data/"

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)
    
_save_pickle(DATA_PATH + 'type_dict.pickle', type_dict)

info_data.to_csv(DATA_PATH + "pk_info_data.csv", index=False, quotechar='"', encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)


# a = open('type_dict.pickle', 'rb')
# type_dict_load = pickle.load(a)