# a demo text-only LLM to be tested
# default: QWen3-4B
# for customization, the model should have:
# - a `modality` attribute: 'text'
# - a `__call__` method:
#     - input: a string as question
#     - output: a string as answer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, device):
        self.modality = 'text'
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
        self.model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen3-4B', torch_dtype=torch.float16)
        self.model.to(device)
        self.device = device

    # images are captioned and appended to the question text
    def __call__(self, question, **kwargs):
        messages = [{'role': 'user', 'content': question}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        model_inputs = self.tokenizer(
            [text], return_tensors='pt').to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        try:
            index = len(output_ids) - output_ids[: : -1].index(151668)
        except ValueError:
            index = 0
        content = self.tokenizer.decode(
            output_ids[index :], skip_special_tokens=True).strip('\n')
        return content
