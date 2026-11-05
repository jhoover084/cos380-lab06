import random
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF

random_state = 2
np.random.seed(random_state)
random.seed(random_state)

PATH = "all_book_titles.txt"

# ---- load + clean (same as lab06) ----
with open(PATH, encoding="utf-8") as f:
    titles = [t.strip() for t in f.read().splitlines() if t.strip()]

df = pd.DataFrame({"title": titles}).drop_duplicates(subset="title").reset_index(drop=True)

noise = ["edition", "th", "s", "access", "introduction", "approach",
         "5th", "volume", "series", "book", "brief", "essentials", "card", "real"]
stop_list = list(ENGLISH_STOP_WORDS) + noise

# ---- bigram vectorizers ----
# ngram_range=(2, 2) -> ONLY two-word phrases ("organic chemistry")
# ngram_range=(1, 2) -> single words AND two-word phrases
# Stop words are removed BEFORE bigrams are built, so "History of Art" becomes
# the bigram "history art". Bigrams are rarer than single words, so min_df is
# lowered to 3 (min_df=5 would throw away most of them).
NGRAM = (2, 2)   # change to (1, 2) to mix unigrams and bigrams
MIN_DF = 3

c_v = CountVectorizer(stop_words=stop_list, ngram_range=NGRAM, min_df=MIN_DF)
X_counts = c_v.fit_transform(df["title"])

tfidf_v = TfidfVectorizer(stop_words=stop_list, ngram_range=NGRAM, min_df=MIN_DF)
X_tfidf = tfidf_v.fit_transform(df["title"])

print("ngram_range:", NGRAM)
print("vocab size:", len(c_v.get_feature_names_out()))
# Most common bigrams overall
totals = np.asarray(X_counts.sum(axis=0)).ravel()
vocab = c_v.get_feature_names_out()
print("\nTop 20 bigrams:")
for j in totals.argsort()[-20:][::-1]:
    print(f"  {vocab[j]}: {int(totals[j])}")


# ---- topic models ----
def show_topics(model, vectorizer, n=10):
    words = vectorizer.get_feature_names_out()
    for i, comp in enumerate(model.components_):
        top = [words[j] for j in comp.argsort()[-n:][::-1]]
        print(f"Topic {i}: {', '.join(top)}")


n_topics = 10

lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
lda.fit(X_counts)

nmf = NMF(n_components=n_topics, random_state=random_state)
nmf.fit(X_tfidf)

print("\nLDA (bigrams)")
show_topics(lda, c_v)
print("\nNMF (bigrams)")
show_topics(nmf, tfidf_v)
print("\nNMF error:", nmf.reconstruction_err_)

# ---- top titles per NMF topic ----
nmf_scores = nmf.transform(X_tfidf)
for topic in range(n_topics):
    print(f"\nNMF topic {topic}")
    top_5 = nmf_scores[:, topic].argsort()[-5:][::-1]
    for pos in top_5:
        print("   ", df["title"].iloc[pos])
