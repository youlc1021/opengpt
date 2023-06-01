
from open_gpt.models.modeling import BaseModel
import torch
import numpy as np
import torch.nn.functional as F
from .rwkvmodel import RWKV
class PIPELINE_ARGS():
    def __init__(self, temperature=1.0, top_p=0.85, top_k=0, alpha_frequency=0.2, alpha_presence=0.2, token_ban=[],
                 token_stop=[], chunk_len=256):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alpha_frequency = alpha_frequency  # Frequency Penalty (as in GPT-3)
        self.alpha_presence = alpha_presence  # Presence Penalty (as in GPT-3)
        self.token_ban = token_ban  # ban the generation of some tokens
        self.token_stop = token_stop  # stop generation whenever you see any token here
        self.chunk_len = chunk_len  # split input into chunks to save VRAM (shorter -> slower)

class PIPELINE():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        return self.tokenizer(x)['input_ids']

    def decode(self, x):
        return self.tokenizer.decode(x)

    def sample_logits(self, logits, temperature=1.0, top_p=0.85, top_k=0):
        probs = F.softmax(logits.float(), dim=-1)
        top_k = int(top_k)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if top_k < len(probs) and top_k > 0:
                probs[sorted_ids[:-top_k]] = 0
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    def generate(self, ctx, token_count=100, args=PIPELINE_ARGS(), callback=None, state=None):
        all_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}

        for i in range(token_count):

            # forward & adjust prob.
            tokens = self.encode(ctx) if i == 0 else [token]
            tokens = torch.tensor(tokens)
            while len(tokens) > 0:
                input = tokens[:args.chunk_len]
                out, state = self.model.forward(input, state)
                tokens = tokens[args.chunk_len:]

            for n in args.token_ban:
                out[n] = -float('inf')
            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

            # sampler
            token = self.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break
            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            # output
            tmp = self.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp:  # is valid utf-8 string?
                if callback:
                    callback(tmp)
                out_str += tmp
                out_last = i + 1
        return out_str

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
        # if self._model_name_or_path.startswith('RWKV/rwkv-raven'):
        #     prompt = f"### Instruction: {prompt}\n### Response:"
        # else:
        #     prompt = "\n" + prompt
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # output = self.model.generate(inputs["input_ids"], max_new_tokens=max_length)
        # output = self.tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        model = RWKV(model=self.model, strategy='cpu fp32')
        pipeline = PIPELINE(model, self.tokenizer)
        args = PIPELINE_ARGS(temperature=temperature, top_p=top_p, top_k=top_k,  # top_k = 0 then ignore
                             alpha_frequency=0.25,
                             alpha_presence=repetition_penalty,
                             token_ban=[0],  # ban the generation of some tokens
                             token_stop=[],  # stop generation whenever you see any token here
                             chunk_len=256)  # split input into chunks to save VRAM (shorter -> slower)
        output = pipeline.generate(prompt, token_count=max_length, args=args)
        return output
