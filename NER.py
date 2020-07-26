import spacy
import pandas as pd
nlp = spacy.load('zh_core_web_sm')

# doc = nlp('今天天氣很好，我出門曬太陽')
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
# token.shape_, token.is_alpha, token.is_stop)

#nlp = spacy.blank("zh")
spam = pd.read_csv('tbrain_train_final_0610_0723.csv')
#print(spam.head(10))
data = spam[['content','full_content']]
textcat = nlp.create_pipe(
              "textcat",
              config={
                "exclusive_classes": True,
                "architecture": "bow"})

# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)
textcat.add_label("ham")
textcat.add_label("spam")
print(textcat)

train_texts = spam['content'].values
train_labels = [{'cats': {'洗錢': label == 'ham',
                          '法院': label == 'spam'}}
                for label in spam['full_content']]
train_data = list(zip(train_texts, train_labels))
print(train_data[:3])