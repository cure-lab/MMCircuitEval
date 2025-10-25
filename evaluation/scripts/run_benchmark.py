import os
import sys

sys.path.append('.')
from modules.runner import Runner
from models.demo_llm import Model
from modules.embedder import Embedder
from modules.evaluator import Evaluator
from modules.llm_scorer import LLMScorer

field = sys.argv[1]

use_cot = False
evaluator_version = 'v1'
bleu_weight = 1
rouge_weight = 1
emb_weight = 1
llm_weight = 2
out_dir = f'out/{field}'
device = 'cuda:0'
llm_scorer_key = ''
llm_scorer_base_url = ''
llm_scorer_id = 'gpt-3.5-turbo'
embedder_key = ''
embedder_base_url = ''
embedder_id = 'text-embedding-3-large'

os.makedirs(out_dir, exist_ok=True)

runner = Runner(version=evaluator_version, bleu_weight=bleu_weight,
                rouge_weight=rouge_weight, emb_weight=emb_weight,
                llm_weight=llm_weight)
model = Model(device)
llm_scorer = LLMScorer(llm_scorer_key, llm_scorer_base_url, llm_scorer_id)
embedder = Embedder(embedder_key, embedder_base_url, embedder_id)
evaluator = Evaluator(llm_scorer, embedder)

predictions = runner.runInference(
    model, field, f'{out_dir}/preds.json', cot=use_cot)
evaluation_results = runner.runEvaluation(
    predictions, field, evaluator, f'{out_dir}/results.json')
runner.showResults(evaluation_results, field)
