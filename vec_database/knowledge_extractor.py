import yaml
import openai
from index_creator import PineconeIndex


class KnowledgeExtractor:
    def __init__(self):
        with open("./vector_db/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.index = PineconeIndex().get_index()
            self.model_name = config["openai"]["embedding"]["model_name"]
            openai.api_key = config["openai"]["embedding"]["api_key"]

    def get_related_knowledge(self, query, top_k=3, passback_gpt = False):

        if len(query) == 0:
            return [""]

        embeddings = openai.Embedding.create(model=self.model_name, input=[query])
        res = self.index.query(
            embeddings["data"][0]["embedding"], top_k=top_k, include_metadata=True
        )

        contexts = [x["metadata"]["text"] for x in res["matches"]]
        
        if passback_gpt == False:
        
            return contexts
        
        elif passback_gpt == True:
            
    
            joined_context = "".join(contexts)

            chat_text = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate information given context",
                },
                {"role": "user", "content": query},
                {"role": "assistant", "content": joined_context},
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=chat_text
            )

            return response.choices[0].message.content
