"""Oscar-only inference preflight; does not modify benchmark results or captures.

Modes: original = FP4/FP16 + fast processor; bf16-fast = unquantized BF16
+ fast processor; bf16-slow = unquantized BF16 + slow processor.
Original vs bf16-fast changes both quantization and compute precision; it does
not isolate their individual effects. bf16-fast vs bf16-slow isolates processor.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from mc_binding.io import atomic_json, environment, file_hash


def logit_summary(logits, tokenizer):
    values = logits.detach().float()
    finite = torch.isfinite(values)
    result = {'shape': list(values.shape), 'all_finite': bool(finite.all()),
              'nan_count': int(torch.isnan(values).sum()),
              'inf_count': int(torch.isinf(values).sum())}
    if result['all_finite']:
        scores, ids = values.topk(5)
        result['top5'] = [{'id': int(i), 'text': tokenizer.decode([int(i)]),
                           'logit': float(s)} for i, s in zip(ids, scores)]
    return result


@torch.inference_mode()
def probe(model, processor, prompt, image, config):
    content = ([{'type': 'image'}] if image is not None else []) + [{'type': 'text', 'text': prompt}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                         tokenize=False, add_generation_prompt=True)
    kwargs = {'text': [text], 'return_tensors': 'pt'}
    if image is not None:
        kwargs.update(images=[image], min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
    inp = processor(**kwargs).to(model.device)
    ids = inp['input_ids'][0]
    record = {'prompt': prompt, 'chat_template_text': text,
              'input_shapes': {k: list(v.shape) for k, v in inp.items()},
              'input_tail_ids': ids[-24:].tolist(),
              'input_tail_decoded': processor.tokenizer.decode(ids[-24:]),
              'image_token_count': int((ids == model.config.image_token_id).sum())}
    if image is not None:
        pixels = inp['pixel_values'].float()
        record['pixels'] = {'all_finite': bool(torch.isfinite(pixels).all()),
                            'min': float(pixels.min()), 'max': float(pixels.max()),
                            'grid_thw': inp['image_grid_thw'].tolist()}
    forward = model(**inp, use_cache=False)
    record['prefill_logits'] = logit_summary(forward.logits[0, -1], processor.tokenizer)
    del forward
    generated = model.generate(**inp, max_new_tokens=16, do_sample=False, use_cache=True,
                               return_dict_in_generate=True, output_scores=True)
    continuation = generated.sequences[0, ids.numel():].tolist()
    record.update(token_ids=continuation,
                  raw_with_special_tokens=processor.tokenizer.decode(continuation, skip_special_tokens=False),
                  raw=processor.tokenizer.decode(continuation, skip_special_tokens=True),
                  generation_scores=[logit_summary(s[0], processor.tokenizer) for s in generated.scores])
    print(prompt, '=>', repr(record['raw']), 'prefill finite:', record['prefill_logits']['all_finite'], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', action='append', choices=['original', 'bf16-fast', 'bf16-slow'], required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not config.get('revision') or config['revision'] == 'main':
        raise ValueError('Use the pinned recognition config')
    if not torch.cuda.is_available():
        raise RuntimeError('Run in a CUDA compute allocation')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    for mode in dict.fromkeys(args.mode):
        report = {'mode': mode, 'config': config, 'environment': environment(),
                  'gpu': torch.cuda.get_device_name(0), 'image_sha256': file_hash(args.image),
                  'state': 'running', 'probes': []}
        model = processor = None
        path = root/f'{mode}.json'
        try:
            dtype = torch.float16 if mode == 'original' else torch.bfloat16
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError('BF16 unsupported on this GPU')
            fast = mode != 'bf16-slow'
            torch.manual_seed(config['seed'])
            processor = AutoProcessor.from_pretrained(config['model_id'], revision=config['revision'], use_fast=fast)
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_quant_type='fp4', bnb_4bit_use_double_quant=False) if mode == 'original' else None
            model, loading = Qwen2VLForConditionalGeneration.from_pretrained(
                config['model_id'], revision=config['revision'], torch_dtype=dtype,
                device_map={'': 0}, attn_implementation='eager', quantization_config=quant,
                output_loading_info=True)
            model.eval()
            report.update(loading_info=loading, dtype=str(dtype),
                          quantized_4bit=bool(getattr(model, 'is_loaded_in_4bit', False)),
                          processor_class=type(processor.image_processor).__name__,
                          processor=processor.image_processor.to_dict())
            atomic_json(path, report)
            for prompt, rgb in [('What is 2 + 2? Answer with one digit.', None),
                                ('Describe the image in one short sentence.', image),
                                ('What color is the leftmost structure? Answer with color only.', image)]:
                report['probes'].append(probe(model, processor, prompt, rgb, config))
                atomic_json(path, report)
            report['state'] = 'complete'
        except BaseException as exc:
            report.update(state='error', error=str(exc))
            raise
        finally:
            atomic_json(path, report)
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
