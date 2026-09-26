import torch
from ..interventions import capture_hooks, patch_hooks, channels


class Qwen:
    def __init__(self, config):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen runner requires a CUDA allocation; use fixture smoke on CPU")
        if not config.get('revision') or config['revision'] == 'main':
            raise ValueError("Set revision to an immutable Hugging Face commit hash")
        dtype_name = config.get('dtype', 'float16')
        if dtype_name not in ('float16', 'bfloat16', 'float32'):
            raise ValueError('dtype must be float16, bfloat16 or float32')
        dtype = getattr(torch, dtype_name)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is not supported by this GPU')
        self.config = config
        torch.manual_seed(config['seed'])
        self.processor = AutoProcessor.from_pretrained(config['model_id'], revision=config['revision'],
                                                      use_fast=config.get('use_fast', True))
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype) if config['load_in_4bit'] else None
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(config['model_id'], revision=config['revision'], torch_dtype=dtype, device_map='auto', quantization_config=quant, attn_implementation='eager').eval()
        for path in ('model.language_model.layers', 'model.layers', 'language_model.model.layers'):
            obj = self.model
            try:
                for part in path.split('.'):
                    obj = getattr(obj, part)
                if len(obj) and hasattr(obj[0], 'self_attn'):
                    self.layers, self.layer_path = obj, path
                    break
            except (AttributeError, TypeError):
                continue
        else:
            raise RuntimeError("Unrecognized decoder layout")
        self.text_config = getattr(self.model.config, 'text_config', self.model.config)

    def module(self, layer, kind):
        if layer < 0 or layer >= len(self.layers) or kind not in ('q', 'k', 'v', 'o'):
            raise ValueError("Invalid decoder projection")
        return getattr(self.layers[layer].self_attn, kind+'_proj')

    def channels(self, layer, kind):
        module = self.module(layer, kind)
        width = module.in_features if kind == 'o' else module.out_features
        n = self.text_config.num_attention_heads if kind in ('q', 'o') else self.text_config.num_key_value_heads
        return channels(width, n, self.config.get('query_heads' if kind in ('q', 'o') else 'kv_heads'))

    def inputs(self, image, prompt):
        message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=[image], return_tensors='pt', min_pixels=self.config['min_pixels'], max_pixels=self.config['max_pixels']).to(self.model.device)

    def roi(self, inp, context, index):
        # Qwen2-VL image processor resizes the entire image; no crop/padding.
        ids = inp['input_ids'][0]
        visual = (ids == self.model.config.image_token_id).nonzero(as_tuple=True)[0].tolist()
        grid = inp['image_grid_thw']
        if len(grid) != 1:
            raise ValueError("Only one still image supported")
        t, h, w = map(int, grid[0].tolist())
        merge = int(self.processor.image_processor.merge_size)
        if t != 1 or h % merge or w % merge or not visual or len(visual) != (h//merge)*(w//merge) or visual != list(range(visual[0], visual[-1]+1)):
            raise ValueError("Unsupported visual-token layout")
        gh, gw = h//merge, w//merge
        x0, y0, x1, y1 = context['objects'][index]['bbox']
        sx, sy = gw/context['width'], gh/context['height']
        positions = [visual[r*gw+c] for r in range(gh) for c in range(gw) if c < x1*sx and c+1 > x0*sx and r < y1*sy and r+1 > y0*sy]
        if not positions:
            raise ValueError("Empty ROI")
        return positions, (gh, gw)

    @torch.inference_mode()
    def capture(self, inp, requests):
        values = {}
        with capture_hooks(self.module, requests, values):
            self.model(**inp, use_cache=False)
        return values

    @torch.inference_mode()
    def answer(self, inp, specs=()):
        length = inp['input_ids'].shape[1]
        check = self.config.get('check_finite_scores', False)
        with patch_hooks(self.module, specs, length):
            out = self.model.generate(**inp, max_new_tokens=self.config['max_new_tokens'], do_sample=False, use_cache=True,
                                      return_dict_in_generate=check, output_scores=check)
        if check:
            for scores in out.scores:
                # -inf may represent intentionally suppressed tokens; NaN/+inf
                # or an entirely unusable vocabulary is a numerical failure.
                if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(dim=-1).all():
                    raise RuntimeError('Non-finite generation scores; stop before interpreting intervention results')
            out = out.sequences
        return self.processor.tokenizer.decode(out[0, length:], skip_special_tokens=True).strip()

    def manifest(self):
        return {'model_class': type(self.model).__name__, 'model_config': self.model.config.to_dict(), 'processor': self.processor.image_processor.to_dict(), 'decoder_path': self.layer_path, 'quantized_4bit': bool(getattr(self.model, 'is_loaded_in_4bit', False)), 'hook_site': 'q/k/v projection output before reshape and RoPE; o projection input', 'cache': True, 'patch_scope': 'prefill_only', 'projection_shapes': {f'{l}:{k}': list(self.module(l, k).weight.shape) for l in self.config['layers'] for k in ('q', 'v')}}
