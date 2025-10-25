from .base import DataModule


class QuestionAugmenter(DataModule):
    def __init__(self, api_key, base_url, gpt_id='gpt-4-turbo'):
        super().__init__(api_key, base_url, gpt_id)
        self.prompt = '''
        You are a professional expert in electronic design automation (EDA) that needs to rewrite a given EDA question.

        You will be given:
        - The question text along with its answer and explanation
        - The type of the question (e.g., single-answer choice, multiple-answer choice, fill-in-the-blank, and open-ended)

        You need to ensure that:
        - Your output question is different from the input question, but still maintains the same meaning
        - Your output question is of the same type as the input question
        - The answers to your output question and the input question are the same

        Note that you should strictly ONLY output the rewritten question in STRING format, WITHOUT any additional information.

        Here is an example:
        **Inputs:**
        **Question:**
        Is there any electrical difference between even spacing and paired power and ground mesh methods?
        **Answer:**
        Yes, there are differences. In the even spacing method, the distance between power and ground is maximized, increasing internal resistance. In the paired method, internal resistance is minimized, improving power efficiency.
        **Explanation:**
        None.
        **Question type:**
        open-ended
        **Output:**
        Analyze if there is any electrical difference between even spacing and paired power and ground mesh methods. If so, explain the difference.

        Here is the question you need to rewrite:
        **Inputs:**
        **Question:**
        {question}
        **Answer:**
        {answer}
        **Explanation:**
        {explanation}
        **Question type:**
        {question_type}
        Please rewrite the given question.
        '''

    def __call__(self, question, answer, explanation, question_type):
        if question_type == 'single':
            question_type = 'single-answer choice'
        elif question_type == 'multi':
            question_type = 'multiple-answer choice'
        elif question_type == 'blank':
            question_type = 'fill-in-the-blank'
        elif question_type == 'open':
            question_type = 'open-ended'
        else:
            print(f'Invalid question type: {question_type}')
            raise NotImplementedError
        prompt = self.prompt.format(
            question=question, answer=answer,
            explanation=explanation, question_type=question_type)
        return self.query(prompt)
