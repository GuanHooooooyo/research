research
建立文本預測模型，(研究bertviz的自注意模式視覺化)，會每個文字轉化為tokens，這些tokens會與每個tokens
進行自注意分數，給予分數之後會進行句子比對，找出最符合的文本句子組合

使用bert.server.client 環境執行有異常，呼叫bert.client()時導致程式停滯無動作(長達約20分鐘)，本次利用lee meng所著Bert這篇文章的下游分類任務來做文本預測
給予兩個句子去訓練，透過各項tokens的比對找出分類的類別(unrelated, aggreed,disaggreed)

訓練樣本數： 2657
預測樣本數： 80126
unrelated    0.679338  (樣本的占比)
agreed       0.294317
disagreed    0.026346

[原始文本]
句子 1：苏有朋要结婚了，但网友觉得他还是和林心如比较合适
句子 2：好闺蜜结婚给不婚族的秦岚扔花球，倒霉的秦岚掉水里笑哭苏有朋！
分類  ：unrelated

name            module
----------------------
bert:embeddings
bert:encoder
bert:pooler
dropout         Dropout(p=0.1, inplace=False)
classifier      Linear(in_features=768, out_features=3, bias=True)
device: cpu 
classification acc: 0.030485509973654498   (分類準確率)



##問題紀錄
BUG: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa4 in position 0: invalid start byte
0xaa
