import json


def formatAnswer(answer, explanation, raw_pred=None):
    answer = '' if answer is None else answer
    explanation = '' if explanation is None else explanation
    raw_pred = '' if raw_pred is None else raw_pred
    if raw_pred != '':
        answer = raw_pred if answer == '' else answer
    formatted_answer = f'Answer: {answer}.'
    if explanation != '':
        formatted_answer += f' Explanation: {explanation}.'
    return formatted_answer


def formatModelOutput(out, version='v1'):
    if out is None:
        return None, None
    if version == 'v1':
        out = out.split('```json')[-1].split('```')[0].strip()
        try:
            pred = json.loads(out)
            return pred.get('answer', None), pred.get('explanation', None)
        except json.JSONDecodeError:
            return None, None
    elif version == 'v2':
        answer = out.split('### Answer ###')[-1].split(
            '### Explanation ###')[0].strip()
        explanation = out.split('### Explanation ###')[-1].strip()
        if answer == '':
            answer = None
        if explanation == '':
            explanation = None
        return answer, explanation
    else:
        raise NotImplementedError

def formatScore(score, bleu_weight, rouge_weight, emb_weight, llm_weight):
    total_score = bleu_weight * score['bleu'] + \
        rouge_weight * score['rouge'] + \
        emb_weight * score['embedding'] + \
        llm_weight * score['llm']
    return total_score / (bleu_weight + rouge_weight + emb_weight + llm_weight)
