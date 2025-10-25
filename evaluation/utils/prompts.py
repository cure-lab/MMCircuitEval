instruction_prompts = {
    'v1': {
        'single': '{statement}\n{question}\nAnswer this single-answer choice question in JSON format and return the JSON object only. The response format should be: {{"answer": `single choice among "A", "B", "C", and "D"`, "explanation": `no more than 3 sentences to explain your thoughts to give the answer`}}.',
        'multi': '{statement}\n{question}\nAnswer this multi-answer choice question in JSON format and return the JSON object only. The response format should be: {{"answer": `multiple choices among "A", "B", "C", and "D"`, "explanation": `no more than 3 sentences to explain your thoughts to give the answer`}}.',
        'blank': '{statement}\n{question}\nAnswer this fill-in-the-blank question in JSON format and return the JSON object only. The response format should be: {{"answer": `a word, phrase, or sentence`, "explanation": `no more than 3 sentences to explain your thoughts to give the answer`}}.',
        'open': '{statement}\n{question}\nAnswer this question in JSON format and return the JSON object only. The response format should be: {{"answer": `no more than 3 sentences to answer this question`, "explanation": `no more than 3 sentences to explain your thoughts to give the answer`}}.',
    },
    'v2': {
        'single': '{statement}\n{question}\nAnswer this single-answer choice question. The response should end with two paragraphs: one paragraph titled "### Answer ###" with a single choice among "A", "B", "C", and "D", and another paragraph titled "### Explanation ###" with no more than 3 sentences to explain your thoughts to give the answer.',
        'multi': '{statement}\n{question}\nAnswer this multi-answer choice question. The response should end with two paragraphs: one paragraph titled "### Answer ###" with multiple choices among "A", "B", "C", and "D", and another paragraph titled "### Explanation ###" with no more than 3 sentences to explain your thoughts to give the answer.',
        'blank': '{statement}\n{question}\nAnswer this fill-in-the-blank question. The response should end with two paragraphs: one paragraph titled "### Answer ###" with a word, phrase, or sentence to answer this question, and another paragraph titled "### Explanation ###" with no more than 3 sentences to explain your thoughts to give the answer.',
        'open': '{statement}\n{question}\nAnswer this question. The response should end with two paragraphs: one paragraph titled "### Answer ###" with no more than 3 sentences to answer this question, and another paragraph titled "### Explanation ###" with no more than 3 sentences to explain your thoughts to give the answer.',
    },
}
image_prompt = 'You may refer to the provided images.'
caption_prompt = 'You may refer to the cues extracted from the provided images:'
cot_prompt = '''
Please use the Chain of Thought (CoT) technique to reason through your answer step-by-step:
1. Understand the Question: Read the question carefully. What is being asked? Is it asking for a specific concept, calculation, or parameter identification? Clarify the core of the question.
2. Locate Relevant Information: Look through the document to find the section where the relevant information is described. This may include module functions, electrical characteristics, circuit configurations, or specific calculations.
3. Extract Key Data: Identify the specific details mentioned in the question. This could include module descriptions, or functional descriptions of components.
4. Apply Relevant Knowledge: Based on the extracted data, apply relevant electrical engineering concepts or formulas to interpret the information. If a calculation is required (e.g., capacitance, power, or voltage), recall and use the appropriate formulas.
5. Check Consistency and Reasonableness: After applying the knowledge, check if the results make sense in the context of the device. Are there any logical inconsistencies? Does the result align with the expected range or design requirements? If not, please go to step 2 and relocate more relevant information.
6. Conclusion: After completing the reasoning steps, summarize your findings and provide the answers and explanations in the last two paragraphs.
'''


def getQuestionPrompt(statement, question, question_type,
                      version='v1', cot=False):
    prompt = instruction_prompts[version][question_type].format(
        statement=statement, question=question)
    if cot:
        prompt += cot_prompt
    return prompt
