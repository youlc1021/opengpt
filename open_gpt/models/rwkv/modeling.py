
from open_gpt.models.modeling import BaseModel
class RWKVModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, prompt: str,
                 max_length: int,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 repetition_penalty: float,
                 do_sample:bool,
                 num_return_sequences:int,
                 **kwargs):
        if self._model_name_or_path.startswith('sgugger'):
            prompt = f"\n{prompt}"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.model.generate(inputs["input_ids"], max_new_tokens=max_length, top_p=top_p, do_sample=do_sample)
            output = self.tokenizer.decode(output[0].tolist())
        else:
            prompt = f"### Instruction: {prompt}\n### Response:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            output = self.model.generate(inputs["input_ids"], max_new_tokens=max_length)
            output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        return output