import spacy
nlp = spacy.load("en")
import seaborn as sns
import matplotlib.pyplot as plt

from data import t0, t1, t2, t3, t4, t5, t6
from functions import  tf_idf_scores

text_list = [t0, t1, t2, t3, t4, t5, t6]
doc_list = [nlp(i) for i in text_list]

dataframe = tf_idf_scores(doc_list)

sns.set()
df_norm = (dataframe - dataframe.values.min())/(dataframe.values.max() - dataframe.values.min())
plt.figure(figsize = (15,3))
sns.heatmap(df_norm)
plt.savefig("Heatmap")
