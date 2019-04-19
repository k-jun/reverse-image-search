# reverse-image-sarch

## **概要**
画像の類似検索をおこなったもの。1000枚の画像で簡単な画像検索を試せる。

## **使用技術**
VGG16, Annoy

## **手順**
ramen_images.zipを解凍後 ramen_images というファイルを作成しそこに格納。

その後以下を実行してAnnoyのmodelを作成し保存

```
python main.py
```

search.py で実際にAnnoyModelを利用して検索を行なっているので、検索したいファイルをpathで指定することで色々遊ぶことができる。

```
python search.py
```
