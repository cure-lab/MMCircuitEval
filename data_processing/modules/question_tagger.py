import json

from .base import DataModule


class QuestionTagger(DataModule):
    def __init__(self, api_key, base_url, gpt_id='gpt-4-turbo'):
        super().__init__(api_key, base_url, gpt_id)
        self.prompt = '''
        You are a professional expert in electronic design automation (EDA) that needs to specify the difficulty level, the tested ability, and the circuit type of a given EDA question.

        You will be given:
        - The question text along with its answer and explanation

        You need to ensure that:
        - You need to choose `difficulty` (the question's difficulty level) between:
            - `easy`, indicating that the question is relatively easy to solve
            - `medium`, indicating that the question is of medium difficulty
            - `hard`, indicating that the question is relatively hard to solve
        - You need to choose `ability` (the question's tested ability) between:
            - `knowledge`, indicating that the question tests the student's knowledge base
            - `comprehension`, indicating that the question tests the student's ability to understand the question text and the given figures
            - `inference`, indicating that the question tests the student's ability to conduct logic reasoning
            - `computation`, indicating that the question tests the student's ability to use the correct formula and perform mathematical calculations
        - You need to choose `ic_type` (the question's circuit type) between:
            - `digital`, indicating that the question is related to digital circuits
            - `analog`, indicating that the question is related to analog circuits
            - `none`, indicating that the question is related to neither digital nor analog circuits
        - You need to specify questions' difficulty levels objectively

        Note that you should strictly ONLY output the specifications in JSON format, WITHOUT any additional information.

        Here is an example:
        **Inputs:**
        **Question:**
        -6dB is equivalent to ____ power gain. (A) 0.5 (B) 0.25 (C) 0.75 (D) 0.8
        **Answer:**
        B
        **Explanation:**
        Power gain in dB can be calculated using formula:
        gain(dB)=10lg(Pout/Pin)
        Substituting -6dB into the formula, we can get:
        Pout/Pin = 10^(-6/10) = 0.251
        **Output:**
        {
            "difficulty": "medium",
            "ability": "computation",
            "ic_type": "analog"
        }

        Here is the question you need to rewrite:
        **Inputs:**
        **Question:**
        {question}
        **Answer:**
        {answer}
        **Explanation:**
        {explanation}
        Please specify the difficulty level, the tested ability, and the circuit type of the given question.
        '''

    def __call__(self, question, answer, explanation):
        prompt = self.prompt.format(
            question=question, answer=answer, explanation=explanation)
        try:
            return json.loads(self.query(prompt))
        except json.decoder.JSONDecodeError:
            return {'difficulty': None, 'ability': None, 'ic_type': None}
