import sys
import json

sys.path.append('.')
from modules.question_tagger import QuestionTagger

api_key = ''
base_url = ''
tagger_id = 'gpt-4o'

# load a question from the demo corpus
with open('./assets/demo_corpus.json', 'r') as f:
    demo_corpus = json.load(f)
question_idx = list(demo_corpus.keys())[0]
question = demo_corpus[question_idx]['statement'] + '\n' + \
    demo_corpus[question_idx]['questions'][0]
answer = demo_corpus[question_idx]['answers'][0]
explanation = demo_corpus[question_idx]['explanations'][0]
print('Original question:')
print(f'Question: {question}')
print(f'Answer: {answer}')
print(f'Explanation: {explanation}')

# load question tagger
question_tagger = QuestionTagger(api_key, base_url, tagger_id)
# tag the question
tags = question_tagger(question, answer, explanation)
print('Question tags:')
print(f'Difficulty: {tags["difficulty"]}')
print(f'Ability: {tags["ability"]}')
print(f'IC type: {tags["ic_type"]}')
