from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class ParagraphRanker:
    """
    Splits doccuments into paragraphs and picks most relevant N paragraphs
    """
    def __init__(self, es_results):
        self.es_results = es_results

    def rank_paragaraphs(self, question, n_paragraphs=3):
        # split documents into paragraphs
        paragraphs_f = []

        for document in self.es_results['hits']['hits']:
            paragraphs = document['_source']['document_text'].splitlines(True)
            for i, paragraph in enumerate(paragraphs):
                paragraphs[i] = paragraph.rstrip()
            paragraphs = [i for i in paragraphs if i]
            paragraphs_f.append(paragraphs)

        # flatten the array
        paragraphs_f = [item for sublist in paragraphs_f for item in sublist]

        # covnert things to a tuple for tf-idf
        documents = (
            question,
        )

        for item in paragraphs_f:
            documents = (*documents, item)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        cos_rez = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

        indices = (-cos_rez[0]).argsort()[:n_paragraphs+1] # we have to do +1 here because it returns indice of the question itself and we don't want to count that None

        contexts = []

        for index in indices:
            if index != 0:
                contexts.append(documents[index])

        return contexts
