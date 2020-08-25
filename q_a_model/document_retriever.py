from elasticsearch import Elasticsearch

class DocumentRetriever:
    """
    Class that connects to elastic search instance and retrieves relevant docs
    """
    def __init__(self, config={'host':'localhost', 'port':9200}):
        self.es = Elasticsearch([config])

    def retrieve_docs(self, question, index, size=3, query=None, title=True):
        if title:
            search_subject = 'document_title'
        else:
            search_subject = 'document_text'
        
        
        if not query:
            query = {
                'query': {
                    'match': {
                        search_subject: question
                    }
                }
            }

        return self.es.search(index=index, body=query, size=size)
