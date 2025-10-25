# the data module for question augmentation and information enhancement
# default: OpenAI GPT
# for customization, the module should have:
# - a `query` method:
#     - input: a string as prompt
#     - output: a string as result
from openai import OpenAI


class DataModule:
    def __init__(self, api_key, base_url, model_id):
        self.agent = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id

    def query(self, prompt):
        chat_completion = self.agent.chat.completions.create(
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            model=self.model_id
        )
        try:
            return chat_completion.choices[0].message.content
        except (TypeError, KeyError, ValueError) as e:
            print(chat_completion)
            print(e)
            return ''
