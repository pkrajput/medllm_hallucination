import pinecone
import yaml
import openai


class PineconeIndex:
    def __init__(self):
        with open("./vector_db/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        pinecone_api_key = config["pinecone"]["api_key"]
        environment = config["pinecone"]["environment"]
        self.index_name = "vector-database"
        pinecone.init(api_key=pinecone_api_key, environment=environment)

        self.index = pinecone.Index(self.index_name)

    def get_index(self):
        return self.index


if __name__ == "__main__":
    pinecone = PineconeIndex()
