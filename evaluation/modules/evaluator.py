import torch
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class Evaluator:
    def __init__(self, llm_scorer, embedder):
        self.llm_scorer = llm_scorer
        self.embedder = embedder
        self.bleu_smooth = SmoothingFunction().method1
        self.rouge = Rouge()

    def __call__(self, pred, gt):
        bleu_score = self.BLEUScore(pred, gt)
        rouge_score = self.RougeScore(pred, gt)
        emb_score = self.embScore(pred, gt)
        llm_score = self.llmScore(pred, gt)
        return {'bleu': bleu_score, 'rouge': rouge_score,
                'emb': emb_score, 'llm': llm_score}

    def BLEUScore(self, pred, gt):
        bleu_score = sentence_bleu([gt.split()], pred.split(),
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=self.bleu_smooth)
        return bleu_score

    def RougeScore(self, pred, gt):
        try:
            rouge_score = self.rouge.get_scores(hyps=[pred], refs=[gt])
            rouge1_score = rouge_score[0]['rouge-1']['f']
            rouge2_score = rouge_score[0]['rouge-2']['f']
            rougel_score = rouge_score[0]['rouge-l']['f']
            return (rouge1_score + rouge2_score + rougel_score) / 3
        except RecursionError:
            return 0

    def embScore(self, pred, gt):
        pred_emb = self.embedder(pred)
        gt_emb = self.embedder(gt)
        return self.cosSim(pred_emb, gt_emb)

    def llmScore(self, pred, gt):
        return self.llm_scorer(pred, gt)

    def cosSim(self, vector1, vector2):
        if isinstance(vector1, torch.Tensor):
            vector1 = vector1.cpu().tolist()
        if isinstance(vector2, torch.Tensor):
            vector2 = vector2.cpu().tolist()
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        if magnitude1 * magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
