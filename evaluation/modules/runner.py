import os
import json
from tqdm import tqdm
from datasets import load_dataset

from .captioner import Captioner
from ..utils.formatter import formatAnswer, formatModelOutput, formatScore
from ..utils.prompts import getQuestionPrompt, image_prompt, caption_prompt


class Runner:
    def __init__(self, version='v1', bleu_weight=1,
                 rouge_weight=1, emb_weight=1, llm_weight=2):
        self.version = version
        self.bleu_weight = bleu_weight
        self.rouge_weight = rouge_weight
        self.emb_weight = emb_weight
        self.llm_weight = llm_weight
        self.captioner = None

    def runInference(self, model, field, out_path, cot=False):
        assert field in ['general', 'spec', 'frontend', 'backend'], \
            f'Unsupported field: {field}'
        assert model.modality in ['text', 'multimodal'], \
            f'Unsupported model modality: {model.modality}'
        if model.modality == 'text' and self.captioner is None:
            self.captioner = Captioner(model.device)
        # load data
        data = load_dataset('charlie314159/MMCircuitEval', split=field)
        preds = {}
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                preds = json.load(f)
        # run inference
        for question_idx, Q in tqdm(enumerate(data)):
            if str(question_idx) in preds.keys() and \
                None not in preds[str(question_idx)]['answers'] and \
                None not in preds[str(question_idx)]['explanations'] and \
                None not in preds[str(question_idx)]['raw_preds']:
                continue
            statement = Q['statement']
            questions = Q['questions']
            question_types = Q['question_types']
            images = Q['images']
            answers = []
            explanations = []
            raw_preds = []
            for i, question in enumerate(questions):
                prompt = getQuestionPrompt(statement, question,
                                           question_types[i], self.version, cot)
                if len(images) > 0:
                    if model.modality == 'multimodal':
                        prompt += (' ' + image_prompt)
                    else:
                        captions = '. '.join(self.captioner(images))
                        prompt += (' ' + caption_prompt + ' ' + captions)
                try:
                    out = model(prompt, images)
                except (TypeError, IndexError, ValueError):
                    out = None
                answer, explanation = formatModelOutput(out, self.version)
                answers.append(answer)
                explanations.append(explanation)
                raw_preds.append(out)
            preds[str(question_idx)] = {
                'answers': answers,
                'explanations': explanations,
                'raw_preds': raw_preds,
            }
            with open(out_path, 'w') as f:
                json.dump(preds, f, indent=4)
        return preds

    def runEvaluation(self, preds, field, evaluator, out_path):
        data = load_dataset('charlie314159/MMCircuitEval', split=field)
        # run evaluation
        results = {}
        for question_idx, pred in tqdm(preds.items()):
            gt = data[int(question_idx)]
            assert 'answers' in pred and 'answers' in gt, \
                f'Missing answers for question {question_idx}'
            assert 'explanations' in pred and 'explanations' in gt, \
                f'Missing explanations for question {question_idx}'
            assert 'raw_preds' in pred, \
                f'Missing raw predictions for question {question_idx}'
            assert len(pred['answers']) == len(gt['answers']) == \
                len(pred['explanations']) == len(gt['explanations']) == \
                len(pred['raw_preds']), \
                f'Answer length mismatch for question {question_idx}'
            scores = []
            for i in range(len(pred['answers'])):
                answer_pred = formatAnswer(
                    pred['answers'][i], pred['explanations'][i],
                    pred['raw_preds'][i])
                answer_gt = formatAnswer(
                    gt['answers'][i], gt['explanations'][i])
                scores.append(evaluator(answer_pred, answer_gt))
            results[question_idx] = scores
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)
        return results

    def showResults(self, results, field):
        data = load_dataset('charlie314159/MMCircuitEval', split=field)
        scores = []
        knowledge_scores = []
        comprehension_scores = []
        reasoning_scores = []
        computation_scores = []
        text_scores = []
        multimodal_scores = []
        for question_idx, scores in tqdm(results.items()):
            scores = [formatScore(s) for s in scores]
            scores += scores
            if data[int(question_idx)]['images'] != []:
                multimodal_scores += scores
            else:
                text_scores += scores
            for i, score in enumerate(scores):
                ability = data[int(question_idx)]['abilities'][i]
                if ability == 'knowledge':
                    knowledge_scores.append(score)
                elif ability == 'comprehension':
                    comprehension_scores.append(score)
                elif ability == 'reasoning':
                    reasoning_scores.append(score)
                elif ability == 'computation':
                    computation_scores.append(score)
        avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
        knowledge_score = sum(knowledge_scores) / len(
            knowledge_scores) if len(knowledge_scores) > 0 else 0
        comprehension_score = sum(comprehension_scores) / len(
            comprehension_scores) if len(comprehension_scores) > 0 else 0
        reasoning_score = sum(reasoning_scores) / len(
            reasoning_scores) if len(reasoning_scores) > 0 else 0
        computation_score = sum(computation_scores) / len(
            computation_scores) if len(computation_scores) > 0 else 0
        text_score = sum(text_scores) / len(
            text_scores) if len(text_scores) > 0 else 0
        multimodal_score = sum(multimodal_scores) / len(
            multimodal_scores) if len(multimodal_scores) > 0 else 0
        print(f'Number of problems: {len(results)}')
        print(f'Number of sub-questions: {len(scores)}')
        print(f'Overall Score: {avg_score * 100}')
        print(f'Knowledge Score: {knowledge_score * 100}')
        print(f'Comprehension Score: {comprehension_score * 100}')
        print(f'Reasoning Score: {reasoning_score * 100}')
        print(f'Computation Score: {computation_score * 100}')
        print(f'Text-only Score: {text_score * 100}')
        print(f'Multimodal Score: {multimodal_score * 100}')
