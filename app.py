import embedding as em
print("run app.py")

dataset = em.MakeDataset()
embed_dataset = dataset.make_embed_dataset()

embed = em.MakeEmbed()
embed.word2vec_init()
embed.word2vec_build_vocab(embed_dataset)

embed.word2vec_train(embed_dataset, 10)

print(embed.word2vec_most_similar("미세먼지"))

sentence = embed_dataset[0]
embed.query2idx(sentence)
w2i = embed.query2idx(sentence)

embed.load_word2vec()