from vertexai.language_models import TextEmbeddingModel

from Utils.gcloud_utils import init_vertexai


class Embedder(object):
    def __init__(self, model_name="text-multilingual-embedding-002"):
        """
        Initializes a Embedder object.
        :param model_name: Embedding model name from Vertex AI
        """
        init_vertexai()
        # embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(model_name)

    def get_text_embeddings(self, text, only_embeddings=True):
        """
        Returns embeddings for a given text (single string).
        :param text: Text string to get embeddings for.
        :param only_embeddings: Whether to return only the embeddings or the full response.
        :return: Embeddings for the given text.
        """
        embeddings = self.embedding_model.get_embeddings([text])[0]
        if only_embeddings:
            embeddings = embeddings.values
        return embeddings

    def get_text_embeddings_batch(self, text_list, only_embeddings=True):
        """
        Returns embeddings for a given text (list of strings).
        :param text_list: List of text strings to get embeddings for.
        :param only_embeddings: Whether to return only the embeddings or the full response.
        :return: Embeddings for the given text.
        """
        embeddings = self.embedding_model.get_embeddings(text_list)
        if only_embeddings:
            embeddings = [embedding.values for embedding in embeddings]
        return embeddings

#
# embedder = Embedder()
# # e = embedder.get_text_embeddings("hello")
# e = embedder.get_text_embeddings_batch(["hello", "world"])
# print(e)
