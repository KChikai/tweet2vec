# coding: UTF-8
"""
エンコーディングしたツイートから，ランダムで取り出したのち類似度計算を行い
類似度の高い文を表示するスクリプト．
"""

import io
import cPickle as pkl
from scipy import spatial
import random
import numpy as np

####################
## for similarity ##
####################

with io.open('tweet_encoding/data.pkl', 'rb') as f:
    titles = pkl.load(f)
with io.open('tweet_encoding/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)
n = len(titles)


def most_similar(idx):

    sims = [(i, spatial.distance.cosine(embeddings[idx], embeddings[i])) for i in range(n) if i != idx]
    sorted_sims = sorted(sims, key=lambda sim: sim[1])
    print titles[idx]
    for sim in sorted_sims[:5]:
        print "%.3f" % (1 - sim[1]), titles[sim[0]]

if __name__ == '__main__':

    # 似たツイートを検索
    for i in range(1):
        most_similar(random.randint(0, n - 1))
        print ""

    # テキストの読み込み確認
    # Xt = []
    # with io.open("../data/encoder_example.txt", 'r', encoding='utf-8') as f:
    #     for line in f:
    #         print line
    #         Xc = line.rstrip('\n')
    #         print Xc
    #         Xt.append(Xc[:145])
    #         break
    #
    # for Xc in Xt:
    #     print Xc

    # テストデータの精度確認（テストデータとエンコーディングデータが等しい場合，precision@1の値は等しくなる）
    # all_num = 0
    # correct = 0
    # for index, line in enumerate(open("tweet_encoding/predicted_tags.txt", "r")):
    #     all_num += 1
    #     tags = line.split(' ')
    #     if index <= 199:
    #         if tags[0] == 'angry':
    #             correct += 1
    #     else:
    #         if tags[0] == 'happy':
    #             correct += 1
    # score = float(correct) / all_num
    # print score