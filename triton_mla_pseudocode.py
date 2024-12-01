import torch

def triton_mla_pseudocode(qk_merge, compress_kv, v_weight, q_rope, k_rope, scale,v_head_dim, padding=None):
    ROPE_HEAD_DIM = q_rope.shape[-1]
    NOPE_HEAD_DIM = V_HEAD_DIM = v_head_dim
    Z,H,M,KV_LORA_RANK = qk_merge.shape
    N = k_rope.shape[-2]

    if padding is None:
        padding = torch.zeros(Z, device=qk_merge.device, dtype=torch.int32)

    BLOCK_M = 64 if M > 16 else 16
    BLOCK_N = 64 if (M > 16 or N <= 64) else 128

    o = torch.empty(Z,H,M,V_HEAD_DIM, device=qk_merge.device, dtype=qk_merge.dtype)

    # 最外两层循环式并行循环
    for off_hz in range(Z*H): # parallel for
        off_z = off_hz // H # 定位样本idx
        off_h = off_hz % H # 定位头idx
        for off_m in range(0, M, BLOCK_M): # parallel for
            qk_merge_tmp = qk_merge[off_z, off_h, off_m:off_m+BLOCK_M, :] # load
            q_rope_tmp = q_rope[off_z, off_h, off_m:off_m+BLOCK_M, :] # load
            num_pad = padding[off_z] # load
            device = qk_merge_tmp.device
            # 初始化行最大值m_i，行累加值l_i，循环累加变量acc
            # min(M-off_m, BLOCK_M)操作是为了防止溢出，比如M不是BLOCK_M的整数倍
            l_i = torch.zeros(min(M-off_m, BLOCK_M), device=device, dtype=torch.float32) 
            m_i = torch.full_like(l_i, -30000.)
            acc = torch.zeros(min(M-off_m, BLOCK_M), KV_LORA_RANK, device=device, dtype=torch.float32)

            end = off_m + min(M-off_m, BLOCK_M) if M!=1 else N
            # 这层循环是在一个chip里正常循环
            for off_n in range(0, end, BLOCK_N):
                compress_kv_tmp = compress_kv[off_z, off_n:off_n+BLOCK_N, :] # load
                k_rope_tmp = k_rope[off_z, 0, off_n:off_n+BLOCK_N, :] # load
                score_nope = qk_merge_tmp @ compress_kv_tmp.transpose(-1,-2)
                score_rope = q_rope_tmp @ k_rope_tmp.transpose(-1, -2)
                score = (score_rope + score_nope) * scale
                # mask设置-30000.就行了，精度没有损失。两个-30000.加起来-60000. < -65504.(float16最小值)，数值不会溢出。
                score += torch.where(off_n+torch.arange(k_rope_tmp.shape[-2], device=device)[None, :]>=num_pad, 0, -30000.)
                if M != 1:
                    score += torch.where((off_m+torch.arange(qk_merge_tmp.shape[-2], device=device))[:, None] >=
                                         (off_n+torch.arange(k_rope_tmp.shape[-2], device=device))[None, :], 0, -30000.)
                m_ij = score.max(-1).values
                m_ij = torch.where(m_ij > m_i, m_ij, m_i)
                score -= m_ij[:, None]
                p = score.exp()
                l_ij = p.sum(-1)
                alpha = (m_i - m_ij).exp()
                l_i = l_i * alpha + l_ij
                acc *= alpha[:, None]
                acc += p.to(q_rope_tmp.dtype) @ compress_kv_tmp
                m_i = m_ij

            v_weight_tmp = v_weight[off_h*V_HEAD_DIM: (off_h+1)*V_HEAD_DIM, :] # load
            acc = acc / l_i[:, None]
            o_tmp = o[off_z, off_h, off_m:off_m+BLOCK_M, :] # load
            o_tmp[:] = acc.to(v_weight_tmp.dtype) @ v_weight_tmp.transpose(-1, -2) # store
    return o

def torch_mla(qk_merge, compress_kv, v_weight, q_rope, k_rope, scale, v_head_dim, attention_mask=None):
    device = k_rope.device
    dtype = k_rope.dtype
    q_len = qk_merge.shape[-2]
    kv_len = k_rope.shape[-2]
    min_value = torch.finfo(dtype).min
    score1 = qk_merge @ compress_kv.unsqueeze(1).transpose(-1,-2)
    score2 = q_rope @ k_rope.transpose(-1,-2)
    if q_len > 1: # prefill
        causal_mask:torch.tensor = torch.full((q_len, kv_len), min_value, dtype=dtype, device=device).triu(diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(q_rope.shape[0], -1, -1, -1)
        if attention_mask is not None:
            causal_mask:torch.tensor = causal_mask.masked_fill((1-attention_mask[:, None, None, :]).bool(), min_value)
    else: # decode
        if attention_mask is None:
            causal_mask = 0
        else:
            causal_mask = torch.zeros_like(attention_mask).masked_fill((1-attention_mask).bool(), min_value)
            causal_mask = causal_mask[:, None, None, :]
    score = (score1 + score2) * scale + causal_mask
    attn_weight = torch.nn.functional.softmax(score, dim=-1, dtype=torch.float32).to(dtype)
    o = attn_weight @ compress_kv.unsqueeze(1)
    v_weight = v_weight.view(-1, v_head_dim, qk_merge.shape[-1]).unsqueeze(0) 
    o = o @ v_weight.transpose(-1, -2)
    return o

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.float16
    z,h,m,n = 8,8,256,256
    v_head_dim,rope_head_dim,nope_head_dim,kv_lora_rank=64,32,64,256
    scale = (rope_head_dim + nope_head_dim) ** (-0.5)
    qk_merge = torch.randn(z,h,m,kv_lora_rank).to(dtype).to(device)
    compress_kv = torch.randn(z,n,kv_lora_rank).to(dtype).to(device)
    v = torch.nn.Linear(kv_lora_rank, h*v_head_dim).to(qk_merge.device).to(qk_merge.dtype)
    v_weight = v.weight
    q_rope = torch.randn(z,h,m,rope_head_dim).to(dtype).to(device)
    k_rope = torch.randn(z,1,n,rope_head_dim).to(dtype).to(device)
    attention_mask = torch.ones(z, n).to(dtype).to(device)
    # 设置padding
    # attention_mask[:, :3] = 0
    padding = attention_mask.shape[-1] - attention_mask.sum(-1)
    a = triton_mla_pseudocode(qk_merge, compress_kv, v_weight, q_rope, k_rope, scale, v_head_dim)
    b = torch_mla(qk_merge, compress_kv, v_weight, q_rope, k_rope, scale, v_head_dim)
    # 如果有padding，不要比较padding的部分
    print(torch.allclose(a,b, 0.00001, 0.00001))
    print(torch.allclose(a,b, 0.0001, 0.0001))
    print(torch.allclose(a,b, 0.001, 0.001))
    print(torch.allclose(a,b, 0.005, 0.005))