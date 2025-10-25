import sys

sys.path.append('.')
from modules.question_proposer import QuestionProposer

api_key = ''
base_url = ''
proposer_id = 'gpt-4o'

# load an image from the demo corpus
figure_paths = ['./assets/2N2222_page1.png',
                './assets/2N2222_page2.png',
                './assets/2N2222_page3.png']
question_type = 'single'

# load question proposer
question_proposer = QuestionProposer(api_key, base_url, proposer_id)
# propose a question
question = question_proposer(figure_paths, question_type)
print('Question:')
print(f'Question: {question["question"]}')
print(f'Answer: {question["answer"]}')
print(f'Explanation: {question["explanation"]}')
