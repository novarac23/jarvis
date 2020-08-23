from elasticsearch import Elasticsearch

class DocumentRetriever:
    """
    Class that connects to elastic search instance and retrieves relevant docs
    """
    def __init__(self, config={'host':'localhost', 'port':9200}):
        self.es = Elasticsearch([config])

    def retrieve_docs(self, question, index, size=3, query=None):
        if not query:
            query = {
                'query': {
                    'match': {
                        'document_title': question
                    }
                }
            }

        return self.es.search(index=index, body=query, size=size)
