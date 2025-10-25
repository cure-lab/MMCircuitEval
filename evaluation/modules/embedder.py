# a demo text embedder
# default: OpenAI GPT
# for customization, the model should have:
# - a `__call__` method:
#     - input: a list of texts
#     - output: a list of pytorch tensors as embeddings
from openai import OpenAI


class Embedder:
    def __init__(self, api_key, base_url, model_id):
        self.agent = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def __call__(self, texts):
        raw_out = self.agent.embeddings.create(
            input=[texts], model=self.model_id).data
        return [o.embedding for o in raw_out]
