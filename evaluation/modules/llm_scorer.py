# a demo LLM answer scorer
# default: OpenAI GPT
# for customization, the model should have:
# - a `__call__` method:
#     - input: a string as the student answer
#              a string as the true answer
#     - output: a float number between 0 and 1 as score
from openai import OpenAI


class LLMScorer:
    def __init__(self, api_key, base_url, model_id):
        self.agent = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = '''
        You are a professional expert in electronic design automation (EDA) that needs to evaluate whether the answer to a question given by a student is correct and consistent with the true answer. You need to evaluate the similarity between the two answer texts.
        Please read the following answer texts and provide a similarity score between 0 and 1 based on how similar the two texts are in terms of the given answer and the corresponding reasoning process, where 0 means the student is totally incorrect and 1 means the student answer is totally correct. Ratings like 0.5 mean that the answer and the reasoning process are partially correct.
        Note that you should strictly only output the score without any additional information.
        **True answer:**
        {gt}
        **Student answer:**
        {pred}
        '''
        self.model_id = model_id

    def __call__(self, pred, gt):
        prompt = self.prompt.format(gt=gt, pred=pred)
        chat_completion = self.agent.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model_id)
        try:
            return float(chat_completion.choices[0].message.content)
        except (TypeError, KeyError, ValueError):
            return 0.5
