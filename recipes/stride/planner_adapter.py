"""A shared-engine planner view with its own explicit sampling identity."""
import time
from flg.backends import Response


class PlannerView:
    def __init__(self, engine, top_p, top_k):
        self.engine = engine
        self.identity = dict(engine.identity, top_p=top_p, top_k=top_k)

    def generate(self, requests):
        from vllm import SamplingParams
        if not requests:
            return []
        assert all(request.stage == 'planner' for request in requests)
        engine = self.engine
        prompts = [engine.tokenizer.apply_chat_template(request.messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for request in requests]
        for prompt in prompts:
            if len(engine.tokenizer.encode(prompt)) + engine.max_tokens > engine.max_model_len:
                raise ValueError('Planner prompt exceeds model context; no implicit truncation')
        sampling = [SamplingParams(temperature=engine.temperature, top_p=self.identity['top_p'],
                    top_k=self.identity['top_k'], presence_penalty=0.0, max_tokens=engine.max_tokens,
                    seed=request.seed) for request in requests]
        start = time.monotonic()
        outputs = engine.model.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.monotonic() - start
        return [Response(output.outputs[0].text, len(output.prompt_token_ids), len(output.outputs[0].token_ids),
                         elapsed / len(outputs), output.outputs[0].finish_reason or 'unknown',
                         self.identity['model']) for output in outputs]
