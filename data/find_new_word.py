import numpy as np
import pandas as pd
import re
from collections import ChainMap, Counter

stop_word = [line.strip() for line in open('./find_new_word/stopwords.txt', 'r', encoding="utf-8").readlines()]
common_word = [line.strip() for line in open('./find_new_word/common_words.txt', 'r', encoding="utf-8").readlines()]
garbage_words = [line.strip() for line in open('./find_new_word/garbage_words.txt', 'r', encoding="utf-8").readlines()]


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


# 导入文本数据，并且去除标点符号等停用词
def read_texts(texts):
    texts = "".join(texts)
    texts = strQ2B(texts)
    for i in stop_word:
        texts = texts.replace(i, "")
    texts = texts.replace('\n', '').replace('\u200b', '').replace('\xa0', '').replace('\u3000', '').replace(' ', '')
    return texts


# ngram函数，
def ngrams(texts, n):
    texts = texts.split(' ')
    output = {}
    for i in range(len(texts) - n + 1):
        g = ''.join(texts[i:i + n])
        output.setdefault(g, 0)
        output[g] += 1
    return output


# 一元、二元、三元、四元分词,设置词频阈值word_frequnency
def seg_texts(texts, word_frequnency):
    split_text = " ".join(texts)
    n_gram_words = []
    unigram = ngrams(split_text, 1)
    bigram = ngrams(split_text, 2)
    trigram = ngrams(split_text, 3)
    fougram = ngrams(split_text, 4)
    global words_dic, num_words
    words_dic = ChainMap(unigram, bigram, trigram, fougram)
    num_words = np.sum(list(words_dic.values()))
    for word in words_dic.keys():
        if words_dic[word] >= word_frequnency:
            n_gram_words.append(word)
    return n_gram_words


# 计算频数大于等于4的词的互信息
def get_MI_value(n_gram_words):
    MI_dic = {}
    for word in n_gram_words:
        if len(word) == 1:
            pass
        elif len(word) == 2:
            AB = words_dic[word] / num_words
            A = words_dic[word[0]] / num_words
            B = words_dic[word[1]] / num_words
            MI_value = np.log(AB / (A * B))
            MI_dic[word] = MI_value
        elif len(word) == 3:
            ABC = words_dic[word] / num_words
            AB = words_dic[word[:2]] / num_words
            C = words_dic[word[2]] / num_words
            A = words_dic[word[0]] / num_words
            BC = words_dic[word[1:]] / num_words
            MI_value = np.mean([np.log(ABC / (AB * C)), np.log(ABC / (A * BC))])
            MI_dic[word] = MI_value
        elif len(word) == 4:
            ABCD = words_dic[word] / num_words
            ABC = words_dic[word[:3]] / num_words
            D = words_dic[word[3]] / num_words
            AB = words_dic[word[:2]] / num_words
            CD = words_dic[word[2:]] / num_words
            A = words_dic[word[0]] / num_words
            BCD = words_dic[word[1:]] / num_words
            MI_value = np.mean([np.log(ABCD / (ABC * D)), np.log(ABCD / (AB * CD)), np.log(ABCD / (A * BCD))])
            MI_dic[word] = MI_value
        else:
            pass
    return MI_dic


# 设置互信息阈值MI_value
def get_MI_words(n_gram_words, MI_value):
    PMI_dic = get_MI_value(n_gram_words)
    MI_words = []
    for k, v in PMI_dic.items():
        if v >= MI_value:
            MI_words.append(k)
    return MI_words


# 计算左右邻接熵
def get_FBE_value(MI_words, texts):
    FR_BE_list = []
    for word in MI_words:
        left_reight_word = re.findall(r'(.{0,1})%s(.{0,1})' % word, texts)
        left_word = [x[0] for x in left_reight_word if x != '']
        left_word_dic = pd.value_counts(left_word).to_dict()
        left_word_num = np.sum(list(left_word_dic.values()))
        F_BE = -np.sum(
            [left_word_dic[i] / left_word_num * np.log(left_word_dic[i] / left_word_num) for i in left_word_dic.keys()])

        right_word = [x[1] for x in left_reight_word if x != '']
        right_word_dic = pd.value_counts(right_word).to_dict()
        right_word_num = np.sum(list(right_word_dic.values()))
        R_BE = -np.sum([right_word_dic[i] / right_word_num * np.log(right_word_dic[i] / right_word_num) for i in
                        right_word_dic.keys()])
        FR_BE_tup = (F_BE, word, R_BE)
        FR_BE_list.append(FR_BE_tup)
    return FR_BE_list


# 左邻接熵小于右邻接熵，向左扩展字，直到满足条件
def get_left_word(words, texts):
    new_left_words = []
    for num in range(1, 8):
        left_words = re.findall(r'(.{{0,{num}}})({word})'.format(word=words, num=num), texts)
        new_word = list(set([x[0] + x[1] for x in left_words]))
        new_word_list = get_FBE_value(new_word, texts)
        for new_word in new_word_list:
            new_left_BE = new_word[0]
            new_BE_word = new_word[1]
            new_right_BE = new_word[2]
            if np.round(new_left_BE / (new_left_BE + new_right_BE), decimals=1) == 0.5:
                new_left_words.append(new_BE_word)
                break
    return new_left_words


# 右邻接熵小于左邻接熵，向右扩展字，直到满足条件
def get_right_word(words, texts):
    new_right_words = []
    for num in range(1, 8):
        right_words = re.findall(r'({word})(.{{0,{num}}})'.format(word=words, num=num), texts)
        new_word = list(set([x[0] + x[1] for x in right_words]))
        new_word_list = get_FBE_value(new_word, texts)
        for new_word in new_word_list:
            new_left_BE = new_word[0]
            new_BE_word = new_word[1]
            new_right_BE = new_word[2]
            if np.round(new_left_BE / (new_left_BE + new_right_BE), decimals=1) == 0.5:
                new_right_words.append(new_BE_word)
                break
    return new_right_words


# 设定左右邻接熵条件
def get_final_words(MI_words, texts):
    final_words = []
    lR_BE_words = get_FBE_value(MI_words, texts)
    for word in lR_BE_words:
        left_BE = word[0]
        BE_word = word[1]
        right_BE = word[2]
        threshold_value = np.round(left_BE / (left_BE + right_BE), decimals=1)
        if threshold_value >= 0.4 and threshold_value <= 0.6:
            final_words.append(BE_word)
        elif len(BE_word) == 4 and threshold_value < 0.4:  # 左邻接熵小，左边界不全，向左扩展一个字
            final_left_words = get_left_word(BE_word, texts)
            final_words = final_words + final_left_words
        elif len(BE_word) == 4 and threshold_value > 0.6:
            final_right_words = get_right_word(BE_word, texts)
            final_words = final_words + final_right_words
        else:
            pass
    return final_words


# 对结果去除数字,筛除常用词，筛除垃圾词
def drop_dupilicates(final_words):
    new_words = list(set(final_words))
    new_words = filter(lambda x: x.isdigit() == False, new_words)
    new_words = filter(lambda x: x not in common_word, new_words)
    new_words = filter(lambda x: x not in garbage_words, new_words)
    new_words = list(new_words)
    return new_words


# 溯源新词的词频
def get_word_freq(new_words, texts):
    new_words_dict = {}
    for word in new_words:
        values = re.findall('%s+' % word, texts)
        freq = len(values)
        new_words_dict[word] = freq
    return new_words_dict


# 运行程序函数
def find_new_word(text, word_frequnency=4, MI_value=7):
    if len(text) <= 1000:
        word_frequnency = 2
        MI_value = 5.5
    elif len(text) > 1000 and len(text) <= 10000:
        word_frequnency = 4
        MI_value = 6
    else:
        word_frequnency = 4
        MI_value = 7
    texts = read_texts(text)
    n_gram_words = seg_texts(texts, word_frequnency)
    MI_words = get_MI_words(n_gram_words, MI_value)
    final_words = get_final_words(MI_words, texts)
    new_word = drop_dupilicates(final_words)
    new_words_dict = get_word_freq(new_word, text)
    return new_words_dict


# 运行程序
if __name__ == '__main__':
    texts = '''
你的微信朋友圈安全吗
你的微信朋友圈安全吗？洛阳市反虚假信息诈骗中心洛阳市反虚假信息诈骗中心WeChatIDlysfzzxIntro洛阳市反虚假信息诈骗中心，旨在实现对虚假信息诈骗犯罪“以快制快、以专制专”的打防态势，广泛提升群众防骗意识，普及报警防骗常识。敲黑板，划重点啦~小伙伴们，你们有没有注意过自己的朋友圈呢~~~~如今微信营销已经成为一种很流行的手段，尤其是很多淘宝卖家们开始借助微信来宣传自己的产品，在朋友圈里推销自己，但是这其中也不缺乏一些诈骗来骗取大家的血汗钱。有不少不法分子将带有病毒或插件的网址，生成二维码，再对外宣称为优惠券、软件或视频，诱导用户扫描，再采用强制下载、安装应用软件等方式以达到获取推广费用或恶意扣费的目的，手机里存储的通讯录、银行卡号等隐私信息更可能会泄露。代购以“低价代购”为诱饵，称可以提供打折代购，但当受害人付了代购款后，诈骗分子就会以“商品被海关扣下，要加缴关税”等理由要加付“关税”，等受害人钱付了，怎么也收不到货品。防骗提示：使用支付宝等安全付款方式。二维码以商品为诱饵，给你返利或者便宜，再发送商品二维码，实则是木马病毒。一旦安装，木马就会盗取你的应用账号、密码。防骗提示：不要随便扫描二维码。伪装身份诈骗者一般装成高富帅、白富美，搭讪后骗取你的信任，进而以借钱、商业金紧张、手术等为由骗取钱财。防骗提示：不要轻易添加陌生人。点赞一是说集满多少个赞就可以获得礼品或优惠，实际等你集满了足够的赞，去兑换礼品或是领取免费消费卡时，却发现拿到手的奖励“缩水”了。另一种诈骗是有的商家发布“点赞”信息时，就留了“后手”，并不透露商家具体位置，而是写着电话通知，要求参与者将自己的电话和姓名发到微信平台，一旦所征集的信息数量够了，这种“皮包”网站就会自动消失，其目的是套取个人信息。防骗提示：这种诱导分享的方式其实是违反微信公众号规定的，一旦发现要勇于举报，避免其他人上当受骗。盗号此种诈骗与盗用QQ号诈骗类似，诈骗者冒充你的家人跟您联系，并以各种理由向你要钱。防骗提示：用语音或者电话联系对方就能识破啦。公众账号在微信平台上使用类似于“交通违章查询”这样的公众账号名称，让你误以为这是官方的微信发布账号，然后再进行诈骗。防骗提示：看准V字认证，或者看看公众号的历史消息内容，就大概能分辨啦。以上六种方法就是今天和大家分享的如何防范微信朋友圈诈骗，大家在平时生活中一定要注意，千万不要相信天上会掉馅饼的事，就算有那也是陷阱。我们是洛阳市反虚假信息诈骗中心，请您和您身边的人关注我们，我们会及时发布预警提示，解密新型诈骗，宣传防骗技巧，接受咨询举报.微信号：lysfzzx长按左侧二维码关注
'''
    result = find_new_word(texts, word_frequnency=2, MI_value=5)
    print(result)
