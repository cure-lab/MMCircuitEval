import sys
import json

sys.path.append('.')
from modules.question_augmenter import QuestionAugmenter

api_key = ''
base_url = ''
augmenter_id = 'gpt-4o'

# load a question from the demo corpus
with open('./assets/demo_corpus.json', 'r') as f:
    demo_corpus = json.load(f)
question_idx = list(demo_corpus.keys())[0]
question = demo_corpus[question_idx]['statement'] + '\n' + \
    demo_corpus[question_idx]['questions'][0]
question_type = demo_corpus[question_idx]['question_types'][0]
answer = demo_corpus[question_idx]['answers'][0]
explanation = demo_corpus[question_idx]['explanations'][0]
print('Original question:')
print(f'Question: {question}')
print(f'Question type: {question_type}')
print(f'Answer: {answer}')
print(f'Explanation: {explanation}')

# load question augmenter
question_augmenter = QuestionAugmenter(api_key, base_url, augmenter_id)
# augment the question
new_question = question_augmenter(question, answer, explanation, question_type)
print(f'New question: {new_question}')