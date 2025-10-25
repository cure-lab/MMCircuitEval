# a demo image captioner
# default: blip-image-captioning-large
# for customization, the model should have:
# - a `__call__` method:
#     - input: a list of images
#     - output: a list of strings as captions
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


class Captioner:
    def __init__(self, device):
        self.processor = BlipProcessor.from_pretrained(
            'Salesforce/blip-image-captioning-large')
        self.model = BlipForConditionalGeneration.from_pretrained(
            'Salesforce/blip-image-captioning-large',
            torch_dtype=torch.float16).to(device)
        self.caption_prefix = 'a document of'
        self.device = device

    def __call__(self, imgs):
        return [self.__generate__(img) for img in imgs]

    def __generate__(self, img):
        inputs = self.processor(
            img, self.caption_prefix, return_tensors='pt').to(
            self.device, torch.float16)
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
