import os
import json
from base64 import b64encode

from .base import DataModule


class QuestionProposer(DataModule):
    def __init__(self, api_key, base_url, gpt_id='gpt-4-turbo'):
        super().__init__(api_key, base_url, gpt_id)
        img_path = f'{os.path.dirname(
            os.path.abspath(__file__))}/2N2222_page2.png'
        with open(img_path, 'rb') as f:
            img = b64encode(f.read()).decode('utf-8')
        self.prompt = '''
        You are a professional expert in electronic design automation (EDA) that needs to propose an EDA-related question along with its answer and explanation according to several given figures.

        You will be given:
        - A list of EDA-related figures
        - The type of the question (e.g., single-answer choice, multiple-answer choice, fill-in-the-blank, and open-ended)

        You need to ensure that:
        - Your output question is closely-related to the given figures
        - Your output question follows the given question type
        - Your output question can be answered in pure text
        - Your output question is not extremely easy or difficult
        - Your output question is not extremely open-ended
        - Your output answer is precisely correct
        - Your output explanation is precisely correct, and matches your output question and answer

        Note that you should strictly ONLY output the proposed question, answer, and explanation in JSON format, WITHOUT any additional information.

        Here is an example:
        **Inputs:**
        **Figures:**
        Figure 1:\n
        ''' + img + '\n' + \
        '''
        **Question type:**
        single-answer choice
        **Output:**
        {
            "question": "What is the minimum transition frequency of the transistor 2N2222 in MHz? (A) 200 (B) 250 (C) 300 (D) 350",
            "answer": "B",
            "explanation": "According to the table `QUICK REFERENCE DATA`, the minimum transition frequency of 2N2222 is 250MHz."
        }

        Here is the question you need to rewrite:
        **Inputs:**
        **Figures:**
        {figures}
        **Question type:**
        {question_type}
        Please propose a question according to the given figures and the question type.
        '''

    def __call__(self, figure_paths, question_type):
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
        figures = ''
        for i, figure_path in enumerate(figure_paths):
            figures += f'Figure {i + 1}:\n{self.encodeImg(figure_path)}\n'
        prompt = self.prompt.format(
            figures=figures, question_type=question_type)
        try:
            return json.loads(self.query(prompt))
        except json.decoder.JSONDecodeError:
            return {'question': None, 'answer': None, 'explanation': None}

    def encodeImg(self, img_path):
        with open(img_path, 'rb') as f:
            img = b64encode(f.read()).decode('utf-8')
        return img
