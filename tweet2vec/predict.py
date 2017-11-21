# coding: UTF-8
"""
インタプリタ形式で入力文のタグを予測するスクリプト
"""

import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
import batch_char as batch
import lasagne

from t2v import tweet2vec, load_params
from settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES


#################
## for predict ##
#################

def invert(d):
    out = {}
    for k,v in d.iteritems():
        out[v] = k
    return out


def classify(tweet, t_mask, params, n_classes, n_chars):
    # tweet embedding
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    # Dense layer for classes
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'], nonlinearity=lasagne.nonlinearities.softmax)

    return lasagne.layers.get_output(l_dense), lasagne.layers.get_output(emb_layer)


def predict():

    model_path = 'model/tweet2vec/'

    # Model
    print("Loading model params...")
    params = load_params('%s/best_model.npz' % model_path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)
    inverse_labeldict = invert(labeldict)

    print("Building network...")
    # Tweet variables
    tweet = T.itensor3()
    t_mask = T.fmatrix()

    # network for prediction
    predictions, embeddings = classify(tweet, t_mask, params, n_classes, n_char)

    # Theano function
    print("Compiling theano functions...")
    predict = theano.function([tweet, t_mask], predictions)
    encode = theano.function([tweet, t_mask], embeddings)

    # Encoding
    print("You can input a sentence which you want to know about pos/neg...")
    while True:
        Xt = []
        print ">>",
        Xc = raw_input()
        if Xc == 'exit':
            print "See you again!"
            break

        Xt.append(Xc[:MAX_LENGTH])
        out_data = []
        out_pred = []
        out_emb = []
        numbatches = len(Xt)/N_BATCH + 1
        for i in range(numbatches):
            xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
            x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
            p = predict(x, x_m)
            e = encode(x, x_m)
            ranks = np.argsort(p)[:, ::-1]

            for idx, item in enumerate(xr):
                out_data.append(item)
                out_pred.append(' '.join([inverse_labeldict[r] if r in inverse_labeldict else 'UNK' for r in ranks[idx, :5]]))
                out_emb.append(e[idx, :])

        # show result
        for tag, item in zip(out_pred, Xt):
            print tag, item

if __name__ == '__main__':
    # 入力データのタグ予測
    predict()
