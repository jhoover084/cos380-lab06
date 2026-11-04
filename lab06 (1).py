import random
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

random_state = 2
#need both bc ones for py gen and other controls np
np.random.seed(random_state)
random.seed(random_state)

path = "all_book_titles.txt"

#got code from other labs
with open(path, encoding="utf-8") as f:
    raw_lines = f.read().splitlines()

titles = [t.strip() for t in raw_lines if t.strip()]

print("raw lines:", len(raw_lines))
print("non-empty titles:", len(titles))
#should make one column table
df = pd.DataFrame({"title": titles})
df.head(10)

uni = df["title"].nunique()
print("NEED TO BE REMOVED:", len(df) - uni)


#shows the stats of each title length
df["n_words"] = df["title"].str.split().str.len()
print(df["n_words"].describe())
#looked this up, splits titles into words, puts every word on its own row, counts them, output a table of 40 most common words
print(df["title"].str.lower().str.split().explode().value_counts().head(40))

#Found 142 duplicates want to remove them in the cleaning section. Titles have a median of 6 words (min 1, max 36). Noise words I plan to remove are the stop words, along with
#publishing words (edition, th, access), the "s" leftover from possesives, and other vague words (introduction, approach).


#cleaning
#drops dups
df = df.drop_duplicates(subset="title").reset_index(drop=True)
print("Should be 0:", len(df) - uni)

noise = ["edition", "th", "s", "access", "introduction", "approach",
         "5th", "volume", "series", "book", "brief", "essentials", "card", "real"]
stop_list = list(ENGLISH_STOP_WORDS) + noise

#skips stop words and just keeps words that are in at least 5 titles -- maybe switch later i just think rarer words will show no pattern whatsoever
c_v = CountVectorizer(stop_words=stop_list, min_df=5)

#makes the vocab from c_v and then builds table with on row per title and on column per word, each row tells us how many times that word appears
X_counts = c_v.fit_transform(df["title"])

#basically same thing except each cell is weighted. lower score = appear more / higher score = rarer words
tfidf_v = TfidfVectorizer(stop_words=stop_list, min_df=5)
X_tfidf = tfidf_v.fit_transform(df["title"])

print(len(c_v.get_feature_names_out()))

#LDA is given counts because it is a probability model. It treeats each title as if its words were drawn form topics, so it is built on how many times a word shows up.
#Raw counts fit perfectly with this model. While TF-IDF scores are scaleed dec that don't track word count. NMF only needs non-neg numbers, so we can use TF_IDF, which
#lowers the weight to words that appear more vis versa.

#Choose to remove dup words bc it would make words look more comman than they actually are inflating them
#using stop words bc they have nothing to do with topics
# added  "edition, th, access" to stop words bc it sjust publishing noise
#choose the min doc freq as 5 bc fewer is too rare to show any pattern at all


from sklearn.decomposition import LatentDirichletAllocation, NMF

n_topics = 3

#just creates the model
lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
#uses the x_counts talbe made above one row per title, one column per word, each cells tells how many time it appears in that title. Looks at which words co-occur across all the titles and figures
lda.fit(X_counts)
#basically same thing but nowa bigger number means that word matters more to a particular topic.
nmf = NMF(n_components=n_topics, random_state=random_state)
nmf.fit(X_tfidf)


#next need to get the top ten words for every topic in the model. 

def show_topics(model, vectorizer, n=10):
    words = vectorizer.get_feature_names_out()
    for i, comp in enumerate(model.components_):
        top = [words[j] for j in comp.argsort()[-n:][::-1]]
        print(f"Topic {i}: {', '.join(top)}")

print("LDA"); show_topics(lda, c_v)
print("\nNMF"); show_topics(nmf, tfidf_v)


#labels for part 4

nmf_labels = {
    0: "History, art and culture",              #history, art, american, film, modern, western
    1: "Computing, stats and applied science",  #computer, statistics, probability, analysis, design. business is mixed in so it's broad
    2: "Life sciences",                         #biology, physiology, molecular, anatomy, chemistry. the cleanest topic
}

lda_labels = {
    0: "Math, engineering and data",            #4 of 5 top titles are math/technical, medical dictionary is an outlier
    1: "Finance (plus 2 NLP conference titles)", #3 finance books, 2 NLP proceedings mixed in. repeated publisher text in long titles
    2: "Unclear",                               #unclear bc top 5 titles are all diff subjects (anatomy, art history, anthropology, geography, biology), only thing they share is the "Plus ... Access Card Package" publisher text
}


#step 5: find the top 5 titles for each topic

#get each title's score for every topic
nmf_scores = nmf.transform(X_tfidf)
lda_scores = lda.transform(X_counts)

#NMF: go through topics 0, 1, 2
for topic in range(3):
    print("NMF topic", topic, "-", nmf_labels[topic])

    #get the scores for this one topic
    scores = nmf_scores[:, topic]

    #sort title positions from lowest to highest score, keep the last 5, then flip so highest is first
    sorted_positions = scores.argsort()
    top_5_positions = sorted_positions[-5:][::-1]

    #print each of those titles
    for position in top_5_positions:
        print("   ", df["title"].iloc[position])
    print()

#LDA: same thing
for topic in range(3):
    print("LDA topic", topic, "-", lda_labels[topic])

    scores = lda_scores[:, topic]

    sorted_positions = scores.argsort()
    top_5_positions = sorted_positions[-5:][::-1]

    for position in top_5_positions:
        print("   ", df["title"].iloc[position])
    print()
    
    #For steph 5, I chcke the five highest-scoring titles for three topics for each model. The NMF titles matched all three of my labels, th ehistory topic return art history books,
    #the computing topic returned computer science books, and the life sciences topic returned biology books. LDA didn't do aswell. Its math topic was motly differeintla equation, actuarial,
    #and network security books, with one medical dictionary that didn't fit. LDA works from raw word counts, so long titles that repeat the same words, like the repeated publisher 
    #series names, pull it toward them, while NMF’s TF-IDF weighting gave cleaner topics overall.


# **Step 5 (top titles).** The NMF titles matched all three labels (art history books, computer science books, biology books). LDA matched less. Its math topic was mostly differential equations, actuarial and network security books with one medical dictionary that doesn't fit, and its finance topic had 2 NLP proceedings in it. LDA works from raw counts, so long titles that repeat the same words (like series names) pull it around, while NMF's TF-IDF weighting gave cleaner topics. Note that many titles tie on score (all the Art History editions score the same), so which five show up can change a little between runs on different computers.


#step 6: try 5, 10, and 15 topics
for k in [5, 10, 15]:
    print("k =", k)

    lda_k = LatentDirichletAllocation(n_components=k, random_state=random_state)
    lda_k.fit(X_counts)
    print("LDA")
    show_topics(lda_k, c_v)

    nmf_k = NMF(n_components=k, random_state=random_state)
    nmf_k.fit(X_tfidf)
    print("NMF")
    show_topics(nmf_k, tfidf_v)
    print("NMF error:", nmf_k.reconstruction_err_)
    print()
    
#For step 6, I tried 5, 10, and 15 topics. At 5, the topics were still mixed, and one LDA topic was mostly publisher filler like “plus” and “package”.
#At 10, the topics were clearer, with separate finance, religion, computing, and biology topics. At 15, some topics started to overlap, like the systems and management ones.
#NMF error only dropped from 45.53 to 43.27 and keeps falling as k grows, so I picked k = 10 based on how clear the topics were.


# **Step 6 (number of topics).** We tried 5, 10 and 15 topics. At 5 the topics were still mixed. At 10 they were clearer, with separate finance, religion, computing and biology topics. At 15 some topics started to overlap, like the systems and management ones. The NMF reconstruction error went from 45.53 (5 topics) to 44.35 (10) to 43.27 (15), but it keeps going down as the number of topics grows, so it can't pick the number for us. We didn't use LDA perplexity. We picked 10 because the topics were the clearest to read.


#step 7: topic mixtures for five titles (k = 10)
nmf10 = NMF(n_components=10, random_state=random_state)
nmf10.fit(X_tfidf)

#each title's score for each of the 10 topics
mix = nmf10.transform(X_tfidf)

#make each row add up to 1 so it reads like percentages
mix = mix / (mix.sum(axis=1, keepdims=True) + 0.000001)

#five titles to look at
for position in [0, 100, 500, 1000, 2000]:
    print(df["title"].iloc[position])
    print(mix[position].round(2))
    print()


#For step 7, I looked at the topic mixtures for five titles. “Essentials of Human Anatomy & Physiology” was about one thing, with 94% on one topic.
#The other four were mixed. “The Theater Experience” split across three topics (36%, 26%, 22%), and “Essentials of Pharmacology” was 49% on one topic and 22% on another.
#Titles with specific technical words land on one topic, while broader titles spread out because their words fit several topics.
