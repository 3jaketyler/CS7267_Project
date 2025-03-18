import concurrent.futures

###############################################################################
#                              File loading                                   #
###############################################################################

import os

def load_files(path):
    documents = []
    file_names = []

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
                file_names.append(file)
    
    return documents, file_names

train_path = "./texts/training"
test_path = "./texts/testing"

train_docs, train_names = load_files(train_path)
test_docs, test_names = load_files(test_path)


###############################################################################
#                                 KeyBert                                     #
###############################################################################

from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keybert(text):
    return kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)


###############################################################################
#                                   Yake                                      #
###############################################################################

from yake import KeywordExtractor

def extract_yake(text):
    yake_extractor = KeywordExtractor(n=2, top=5, stopwords=None)
    return [kw[0] for kw in yake_extractor.extract_keywords(text)]


###############################################################################
#                                 tfidf                                       #
###############################################################################

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf(docs, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).tolist()[0]

    ranked_keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [word for word, _ in ranked_keywords[:top_n]]


###############################################################################
#                                 testing                                     #
###############################################################################

train_keybert = [extract_keybert(doc) for doc in train_docs]
train_yake = [extract_yake(doc) for doc in train_docs]
train_tfidf = extract_tfidf(train_docs)

print("Sample Extracted Keywords:")
print(f"KeyBERT: {train_keybert[0]}")
print(f"YAKE!: {train_yake[0]}")
print(f"TF-IDF: {train_tfidf[:5]}")


'''
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        results = list(executor.map(extract_keywords, documents))

    for file_name, keywords in zip(file_names, results):
        print(f"\nKeywords for {file_name}: {keywords}")'
        '''