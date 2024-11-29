import torch
torch.cuda.empty_cache()
import modeling_minicpm
from transformers import AutoConfig, AutoTokenizer, LlamaModel

from custom import *

import time
import pandas as pd

import argparse


path = '/data/models/MiniCPM3-4B'

def attn():
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.hidden_size = 256
    config.max_position_embeddings = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.q_lora_rank = 32
    config.kv_lora_rank = 32
    config.torch_dtype = torch.float32

    inputs = torch.randn(1, 64, 256).cuda()
    attn_mask = torch.ones(64,64).tril().cuda()
    attn_mask = torch.where(~attn_mask.bool(), torch.finfo(torch.float32).min, 0)
    attn_mask = attn_mask[None, None, :, :]

    attn = modeling_minicpm.MiniCPMAttention(config).cuda()

    custom_attn = TritonMLAMiniCPMAttention(config).cuda()
    # custom_attn = CustomMiniCPMAttention(config).cuda()
    custom_attn.load_state_dict(attn.state_dict())
    custom_attn.train2test()

    with torch.inference_mode():
        out1 = attn(inputs, attn_mask)[0]
        out2 = custom_attn(inputs, attn_mask)[0]
    print('='*50 + ' 原始模块输出 ' +'='*50)
    print(out1)
    print('='*50 + ' 改进模块输出 ' +'='*50)
    print(out2)

def speed():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=2, choices=[0,1,2,3,4], 
                        help='''0: torch_naive mha
                                1: torch_naive mla
                                2: triton mla
                                3: flash-attn-2 mha
                                4: sdpa mha
                                ''')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp32:
        dtype = torch.float32
    else:
        dtype = torch.float16
    device = 'cpu' if args.use_cpu else 'cuda'

    map = {0:['torch_naive mha', 0, 'eager'],
           1:['torch_naive mla', 1, 'eager'],
           2:['triton mla', 2, 'eager'],
           3:['flash_attn_2 mha', -1, 'flash_attention_2'],
           4:['sdpa mha', -1, 'sdpa']}
    name, idx, impl = map[args.idx]

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = impl
    config.eager_idx = idx
    config.torch_dtype = dtype

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = 0

    custom_model = modeling_minicpm.MiniCPM3ForCausalLM.from_pretrained(path, device_map=device, torch_dtype=dtype, config=config)
    for m in custom_model.modules():
        if isinstance(m, MiniCPMTrain2Test):
            m.train2test()
    print(custom_model)
    p2 = custom_model.num_parameters()
    
    settings = []
    times = []
    bs = 8
    for length in [64, 256, 512, 1024, 2048]:
        if length > 1000:
            bs = 4
        output_tokens = min(length, 512)
        input_ids = torch.randint(10000, 20000, (bs, length)).to(device)
        mask = torch.ones_like(input_ids)
        key = f'bs={bs} in={length} out={output_tokens}'
        with torch.inference_mode():
            start = time.time()
            model_outputs = custom_model.generate(input_ids, attention_mask=mask, do_sample=True, 
                                                max_new_tokens=output_tokens, temperature=1.3, eos_token_id=99999)
            t = time.time() - start
        print(key + f' 用时: {t:.2f}')
        settings.append(key)
        times.append(round(t, 3))
        torch.cuda.empty_cache()
        time.sleep(3)
    df = pd.DataFrame({'setting': settings, 'spend_time': times})
    print('='*10 + name + '='*10)
    print(df)
    
def generate():
    dtype = torch.float16
    device = 'cuda'
    max_new_tokens = 100

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = 'eager'
    config.eager_idx = 1
    config.torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = 0
    prompts1 = [{'role': 'user', 'content': '介绍一下故宫，不少于100字'}]
    prompts2 = [{'role': 'user', 'content': '你是谁'}]
    text = [tokenizer.apply_chat_template(i, add_generation_prompt=True, tokenize=False) for i in [prompts1, prompts2]]

    custom_model = modeling_minicpm.MiniCPM3ForCausalLM.from_pretrained(path, device_map=device, torch_dtype=dtype, config=config)
    
    for m in custom_model.modules():
        if isinstance(m, MiniCPMTrain2Test):
            m.train2test()
    print(custom_model)
    p2 = custom_model.num_parameters()

    inputs = tokenizer(text, return_tensors='pt', padding='longest')
    inputs = {k:v.cuda() for k,v in inputs.items()}
    print(inputs)
    with torch.inference_mode():
        torch.manual_seed(42)
        start = time.time()
        model_outputs = custom_model.generate(**inputs, do_sample=True, 
                                            max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.95)
        time2 = time.time() - start
    ans = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    print(ans)

if __name__ == '__main__':
    speed()