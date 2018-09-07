# coding: utf-8

# <h2>Step for checking overfitting problem</h2>
# <h4>The output sentences from Sequential GAN are compared with the real value, the positive sentense.</h4>

# In[1]:


import numpy as np

f1 = open('./save/eval_file.txt', 'r')  # f1 < SeqGAN output
f2 = open('./data/3_pk_data_index.txt', 'r')  # f2 < real value

lines1 = f1.readlines()
lines2 = f2.readlines()


# In[2]:


def Jaccard_sim(lines1, lines2):

    UNK_num = '4979' # 6693(previous) 4979(current)

    # Jaccard similarity between two lines
    def Jaccard_sim_lines(line1, line2):
        # omitting \n at the end of the lines
        words1 = line1.split()[:-1]
        words2 = line2.split()[:-1]

        # omitting UNK
        def Temp_len(words):
            temp_len = len(words)
            for i in range(len(words1)):
                if words[i] == UNK_num:
                    temp_len -= 1
            return temp_len

        # Find the same word between two lines. if there is: 1 else: 0
        temp = np.zeros(Temp_len(words1))
        for num1 in range(Temp_len(words1)):
            for num2 in range(Temp_len(words2)):
                if words1[num1] == words2[num2] and words1[num1]!=UNK_num:
                    temp[num1] = 1
                    words2[num2] == -1  # the number counted will not be considered from now on
                    break
        return sum(temp)/Temp_len(words1)

    # calculrate the number of union bewtween two documents
    def Union_documnets(lines1, lines2):
        temp = len(lines1) + len(lines2)
        for num1 in range(len(lines1)):
            for num2 in range(len(lines2)):
                if lines1[num1] == lines2[num2]:
                    temp -= 1
        return temp

    #     <<        main      >>
    #     Jaccard Similary
    js = np.zeros(len(lines1), dtype=float)
    if len(lines1) == 0:
        print('The txt file is not exist')
    else:
        for num1 in range(len(lines1)):
            for num2 in range(len(lines2)):
                temp = 0
                temp = Jaccard_sim_lines(lines1[num1], lines2[num2])
                if temp > js[num1]:
                    js[num1] = temp
        return sum(js) / Union_documnets(lines1, lines2)


# In[3]:

print("similarity test")
print('두 문서는 %.2f' % (100 * Jaccard_sim(lines1, lines2)), '% 유사합니다.')
