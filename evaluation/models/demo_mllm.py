# a demo multi-modal LLM to be tested
# default: QWen2-VL-2B-Instruct
# for customization, the model should have:
# - a `modality` attribute: 'multimodal'
# - a `__call__` method:
#     - input: a string as question
#              a list of PIL Images
#     - output: a string as answer
import torch
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Model:
    def __init__(self, device):
        self.modality = 'multimodal'
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            'Qwen/Qwen2-VL-2B-Instruct', torch_dtype=torch.float16,
            attn_implementation='flash_attention_2')
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen2-VL-2B-Instruct')
        self.device = device

    def __call__(self, question, imgs=[]):
        content = []
        for img in imgs:
            buffered = BytesIO()
            img.save(buffered, format='PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            content.append({'type': 'image', 'image': img_base64})
        content.append({'type': 'text', 'text': question})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors='pt')
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in
                                 zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0]
