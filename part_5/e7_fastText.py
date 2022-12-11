from gensim.models.fasttext import FastText

# download the bin + text model for your language of choice
MODEL_PATH = "../src/ukr_fastText/cc.uk.300.bin.gz"
ft_model = FastText.load_fasttext_format(model_file=MODEL_PATH)
print(ft_model.most_similar(positive=['сало'], topn=5))

