import json
import jieba
import numpy as np
import requests
from keras_preprocessing import sequence, text


def remove_stop_words(sentence, stopwordpath="./data/stopwordlist/baidu.txt"):
    stopwords = [line.strip() for line in open(stopwordpath, 'r', encoding='utf-8').readlines()]
    processed_docs_train = []
    # jieba.add_word("添加jieba不能分的词")
    for word_list in sentence:
        outstr = []
        word = jieba.cut(word_list.strip().rstrip(), cut_all=True)
        for i in word:
            if i not in stopwords:
                if i != '\t' and i != ' ':
                    outstr.append(i)
        processed_docs_train.append(" ".join(outstr))
    return processed_docs_train


def predict(string):
    string = [string]
    processed_docs_all_test = remove_stop_words(string)
    word_seq_all_test = tokenizer.texts_to_sequences(processed_docs_all_test)
    word_seq_all_test = sequence.pad_sequences(word_seq_all_test, maxlen=512)
    payload = {
        "instances": [{'input': word_seq_all_test.tolist()[0]}]
    }
    r = requests.post('http://192.168.66.59:8501/v1/models/illegal_classifier:predict', json=payload)
    output = json.loads(r.content.decode('utf-8'))["predictions"]
    text_class = int(np.argmax(output, axis=1))
    predict_score = max(output[0])
    class_dict = {0: "网络赌博", 1: "打击", 2: "网络黑产", 3: "网络诈骗", 4: "网络色情", 5: "网络传销", 6: "白名单"}
    print(class_dict[text_class])
    print(predict_score)
    return class_dict[text_class]


if __name__ == '__main__':
    with open("./data/tokenizer.json", "r") as f:
        tokenizer_json = f.read()
    tokenizer = text.tokenizer_from_json(tokenizer_json)
    predict("""
            需要预测的文本
            """)
