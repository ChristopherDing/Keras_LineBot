from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from keras.preprocessing import sequence
from keras.models import load_model
import jieba
import numpy as np
import os

mycwd = os.getcwd()
os.chdir(mycwd)

app = Flask(__name__)

model = load_model('school.h5')

# Channel Access Token
line_bot_api = LineBotApi(
    'alzsTgpLNdbsPh2MJMqTubPkanoq5glqyah01Q3Ry7YXMI8tQz2V66sFmzNySmDripQcT+LhnD2pCCY9jC5+1ZH8BeYz/oQL1iZYtHCGT3sVMUyXhGoQhZ1Z8dRIt4NJqQeN0cU4HHzIpZ3nlmLmKQdB04t89/1O/w1cDnyilFU=')
# Channel Secret
handler = WebhookHandler('02800c7708ac1715560f38103153be47')


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


def stopwordslist():
    stopwords = [line.strip() for line in open('stopwords_Chinese.txt', 'r', encoding='utf-8').readlines()]
    return stopwords

def list2file(list, file):
    # 將list寫入文件
    fout = open(file, 'a', encoding='utf-8')
    for item in list:
        for i in item:
            fout.write(str(i) + ' ')
        fout.write('\n')
    fout.close()

def getdata():
    with open('word_to_int_tables.txt', 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f:
            if (cnt == 0):
                tmpline = line.replace("\n", "")
                tmpkey = tmpline.split(',')
            else:
                tmpline = line.replace("\n", "")
                tmpval = tmpline.split(',')
                tmpval = [int(x) for x in tmpval]
            cnt += 1
        word_index = dict(zip(tmpkey, tmpval))
    f.close()
    return word_index

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    print("message: " + event.message.text)
    INPUT_SENTENCES = [event.message.text]
    list2file([INPUT_SENTENCES], 'data.txt')
    XX = np.empty(len(INPUT_SENTENCES), dtype=list)
    i = 0
    chinese_punctuations = ['，', '。', '：', '；', '?', '？', '（', '）', '「', '」', '！', '“', '”', '\n', ' ']  # 中文標點去除
    stopwords = stopwordslist()
    word_index = getdata()
    for sentence in INPUT_SENTENCES:
        words = jieba.cut(sentence)
    seq = []
    for word in words:
        if ((word in word_index) and (word not in chinese_punctuations) and (word not in stopwords)):
            seq.append(word_index[word])
    XX[i] = seq
    i += 1

    XX = sequence.pad_sequences(XX, maxlen=21)
    for index, l in enumerate(XX):
        ls = set(l)
        ls.remove(0)
        if len(ls) > 0:
            labels = int(round(np.argmax(model.predict(np.array([l])))))
        else:
            labels = 16
    label2word = {0: '教務處以及教務處長室在行政樓一樓', 1: '該小姐/先生在行政一樓教務處', 2: '請前往註冊組，註冊組在行政一樓', 3: '該小姐/先生在行政一樓註冊組'
        , 4: '該小姐/先生在行政一樓課務組', 5: "請前往行政一樓課務組詢問相關規則，或前往課務組網頁"
        , 6: '該小姐/先生在行政一樓招生組', 7: '請前往行政一樓招生組詢問相關規則', 8: '生活輔導組在行政樓一樓', 9: '該小姐/先生在行政一樓生活輔導組', 10: '陳振遠校長在行政二樓校長室'
        , 11: '副校長室在行政二樓', 12: '影印部在宗教一樓，可以付費提供影印服務', 13: '衛生保健組在行政三樓，可以提供測升高體重，傷口應急處理的服務'
        , 14: '咨商辅导組在行政三樓，你可以找咨商辅导组的老师或志工解决心中烦恼', 15: '你好！我是義大校務達人，你可以問我學生在行政大樓的相關事務哦', 16: '對不起，我不懂你在說什麼'}
    # display
    content = '{}'.format(label2word[labels])
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=content))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)