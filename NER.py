import spacy
import pandas as pd
from spacy.util import minibatch
nlp = spacy.load('zh_core_web_sm')

#doc = nlp('西門子將努力參與中國的三峽工程建設。')
#for ent in doc.ents:
#    print(ent.text,ent.label_)
"""document = 
中國IT之家報導，據彭博社引述知情人士消息指出，三星電子已經完成為期兩個月摺疊螢幕手機GalaxyFold的重新設計，
以解決尷尬的螢幕故障問題，該故障迫使該機發售延遲。知情人士表示，三星目前正處於推出GalaxyFold商業版的最後階段，
但尚無法確定開賣的日期。據看過最新版GalaxyFold人士透露，目前螢幕保護裝置膜已拉伸，會包裹整個螢幕和外邊框，
因此不可能用手剝離；三星也重新設計了鉸鏈，與顯示器齊平，這樣在手機打開時會進一步拉伸保護膜，
這種張力會使得保護膜更硬，更像是裝置的自然部分，而不是可拆卸的配件；並且隨著時間的推移，
可能有助於減少螢幕中間出現摺痕的可能性。據其中一位知情人士表示，三星很快將開始將GalaxyFold的主要零件，
包括顯示器和電池運送到越南的工廠進行組裝，同時該公司正在討論開賣日期。但另一位知情人士表示，
三星不太可能在8月7日紐約舉行的「Unpack」活動中推出重新設計的GalaxyFold，其屆時將推出旗艦Note10手機。
"""
#document = nlp(document)
#print(dir(document))

#nlp = spacy.blank("zh")
data = pd.read_csv('tbrain_train_final_0610_0723.csv')
#print(spam.head(10))
data = data[['name','full_content']]
print(data.head())

##create text label
textcat = nlp.create_pipe(
              "textcat",
              config={
                "exclusive_classes": True,
                "architecture": "bow"})

nlp.add_pipe(textcat)
textcat.add_label("AML")
textcat.add_label("NOT AML")

lst = []
for i in range(data.shape[0]):
    if data.name[i] == '[]':
        lst.append("Not_AML")
    else:
        lst.append("AML")
data["label"] = lst
data = data[['full_content','label']]
data.columns = ['text','label']
print(data.head())

##train model
train_texts = data['text'].values
train_labels = [{'cats': {'AML': label == 'AML',
                          'Not_AML': label == 'Not_AML'}}
                for label in data['label']]
train_data = list(zip(train_texts, train_labels))[:70]
print(train_data)

spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

# Create the batch generator with batch size = 8
batches = minibatch(train_data, size=8)
# Iterate through minibatches
for batch in batches:
    # Each batch is a list of (text, label) but we need to
    # send separate lists for texts and labels to update().
    # This is a quick way to split a list of tuples into lists
    texts, labels = zip(*batch)
    nlp.update(texts, labels, sgd=optimizer)
import random

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()
## reduce losses
losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        # Each batch is a list of (text, label) but we need to
        # send separate lists for texts and labels to update().
        # This is a quick way to split a list of tuples into lists
        texts, labels = zip(*batch)
        nlp.update(texts, labels, sgd=optimizer, losses=losses)
    print(losses)
texts = ["一張信用卡，如何讓一個6,600人的銀行展開大轉型？今年春天，台北富邦銀行推出的信用卡J卡，上市後連續120天，每天都有超過2,000份申請，「這是台北富邦歷史上從沒有發生過的事情，」北富銀總經理程耀輝說。一位國銀信用" ]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)
print(scores)
