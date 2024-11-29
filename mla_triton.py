import torch

import triton
import triton.language as tl


# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [16,32]\
#     for BN in [16, 32]\
#     for s in [1,2]\
#     for w in [1,2]\
# ]


# def keep(conf):
#     BLOCK_M = conf.kwargs["BLOCK_M"]
#     BLOCK_N = conf.kwargs["BLOCK_N"]
#     # if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
#     #     return False
#     return True

# @triton.autotune(list(filter(keep, configs)), key=["KV_LORA_RANK"])
@triton.jit
def _mla(QK_MERGE, COMPRESS_KV, V_WEIGHT, Q_ROPE, K_ROPE, scale, Out, PADDING,#
              stride_qk_merge_z, stride_qk_merge_h, stride_qk_merge_m, stride_qk_merge_k,  #
              stride_c_kv_z, stride_c_kv_n, stride_c_kv_k,  #
              stride_q_rope_z, stride_q_rope_h, stride_q_rope_m, stride_q_rope_k,
              stride_k_rope_z, stride_k_rope_h, stride_k_rope_n, stride_k_rope_k,
              stride_oz, stride_oh, stride_om, stride_ok,  #
              Z, H, M, N, 
              KV_LORA_RANK: tl.constexpr, V_HEAD_DIM: tl.constexpr,  #
              ROPE_HEAD_DIM: tl.constexpr,  #
              NOPE_HEAD_DIM: tl.constexpr, 
              BLOCK_M: tl.constexpr, #
              BLOCK_N: tl.constexpr,
              ):
    start_m = tl.program_id(0)
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H 

    qk_merge_offset = off_z.to(tl.int64) * stride_qk_merge_z + off_h.to(tl.int64) * stride_qk_merge_h
    c_kv_offset = off_z.to(tl.int64) * stride_c_kv_z
    q_rope_offset = off_z.to(tl.int64)  * stride_q_rope_z + off_h.to(tl.int64) * stride_q_rope_h
    k_rope_offset = off_z.to(tl.int64) * stride_k_rope_z
    o_offset = off_z.to(tl.int64)  * stride_oz + off_h.to(tl.int64)  * stride_oh

    num_pad = tl.load(PADDING+off_z)
    
    QK_MERGE_block_ptr = tl.make_block_ptr(
        base=QK_MERGE + qk_merge_offset,
        shape=(M, KV_LORA_RANK),
        strides=(stride_qk_merge_m, stride_qk_merge_k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, KV_LORA_RANK),
        order=(1, 0),
    )
    V_WEIGHT_block_ptr = tl.make_block_ptr(
        base=V_WEIGHT,
        shape=(KV_LORA_RANK, V_HEAD_DIM*H),
        strides=(1, KV_LORA_RANK),
        offsets=(0, off_h*V_HEAD_DIM),
        block_shape=(KV_LORA_RANK, V_HEAD_DIM),
        order=(0, 1),
    )
    COMPRESS_KV_block_ptr = tl.make_block_ptr(
        base=COMPRESS_KV + c_kv_offset,
        shape=(KV_LORA_RANK, N),
        strides=(stride_c_kv_k, stride_c_kv_n),
        offsets=(0, 0),
        block_shape=(KV_LORA_RANK, BLOCK_N),
        order=(0, 1),
    )
    Q_ROPE_block_ptr = tl.make_block_ptr(
        base=Q_ROPE + q_rope_offset,
        shape=(M, ROPE_HEAD_DIM),
        strides=(stride_q_rope_m, stride_q_rope_k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, ROPE_HEAD_DIM),
        order=(1, 0),
    )
    K_ROPE_block_ptr = tl.make_block_ptr(
        base=K_ROPE + k_rope_offset,
        shape=(ROPE_HEAD_DIM, N),
        strides=(stride_k_rope_k, stride_k_rope_n),
        offsets=(0, 0),
        block_shape=(ROPE_HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(M, V_HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, V_HEAD_DIM),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    qk_merge = tl.load(QK_MERGE_block_ptr, boundary_check=(0,), padding_option='zero')
    q_rope = tl.load(Q_ROPE_block_ptr, boundary_check=(0,), padding_option='zero')
    v_weight = tl.load(V_WEIGHT_block_ptr)


    # loop over k, v and update accumulator
    end = (start_m+1) * BLOCK_M if M != 1 else N
    for start_n in range(0, end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        compress_kv = tl.load(COMPRESS_KV_block_ptr, boundary_check=(1,), padding_option='zero')
        k_rope = tl.load(K_ROPE_block_ptr, boundary_check=(1,), padding_option='zero')
        qk_nope = tl.dot(qk_merge, compress_kv) * scale
        qk_rope = tl.dot(q_rope, k_rope) * scale
        qk = (qk_nope + qk_rope)
        qk += tl.where((start_n + off_n[None, :])>=num_pad, 0, -30000.) 
        if M != 1:
            qk += tl.where(off_m[:, None] >= (start_n + off_n[None, :]), 0, -30000.)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None] 
        p = tl.math.exp(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp(m_i - m_ij)
        # beta = tl.exp(m_ij - m_i_new)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        # v = tl.dot(tl.trans(compress_kv, 1, 0), v_weight)
        p = p.to(q_rope.dtype)
        acc += tl.dot(p, tl.trans(compress_kv, 1, 0))
        # update m_i and l_i
        m_i = m_ij
        K_ROPE_block_ptr = tl.advance(K_ROPE_block_ptr, (0, BLOCK_N))
        COMPRESS_KV_block_ptr = tl.advance(COMPRESS_KV_block_ptr, (0, BLOCK_N))
    acc = acc / l_i[:, None]
    acc = tl.dot(acc, v_weight.to(acc.dtype))
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))

class MLAOPS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qk_merge, compress_kv, v_weight, q_rope, k_rope, scale, v_head_dim, padding=None):
        '''Multi-Latent-Attention implement by triton
        Only implement the forward pass for inference, the backward pass doesn't implement 
        If you want to improve the performance better, you can set the BLOCK_M and BLOCK_N according to your gpu

        Default set nope_head_dim = v_head_dim

        Arguments:
            qk_merge: (bs, num_head, q_len, kv_lora_rank)
            compress_kv: (bs, kv_len, kv_lora_rank)
            v_weight: (num_head * v_head_dim, kv_lora_rank)
            q_rope: (bs, num_head, q_len, rope_head_dim)
            k_rope: (bs, 1, kv_len, rope_head_dim)
            scale: float. The scaling of QK^T before applying softmax.
            v_head_dim: int
            padding: torch.tensor, the num paddings in each sample
                For example:
                    input: [[1,2,3,4,5,6],
                            [pad,pad,pad,1,2,3]]
                    padding: [0, 3]
                you can compute this by "attention_mask.shape[-1] - attention_mask.sum(-1)"

        Return:
            out: (bs, num_head, q_len, v_head_dim)
        '''

        ROPE_HEAD_DIM = q_rope.shape[-1]
        NOPE_HEAD_DIM = V_HEAD_DIM = v_head_dim
        Z,H,M,KV_LORA_RANK = qk_merge.shape
        N = k_rope.shape[-2]

        o = torch.empty(Z,H,M,V_HEAD_DIM, device=qk_merge.device, dtype=qk_merge.dtype)
        if padding is None:
            padding = torch.zeros(N, device=qk_merge.device, dtype=torch.int32)

        BLOCK_M = 64 if M > 16 else 16
        BLOCK_N = 64 if (M > 16 or N <= 64) else 128
        grid = lambda args: (triton.cdiv(M, args["BLOCK_M"]), Z*H, )
        _mla[grid](
            qk_merge, compress_kv, v_weight, q_rope, k_rope, scale, o, padding, #
            qk_merge.stride(0), qk_merge.stride(1), qk_merge.stride(2), qk_merge.stride(3),  #
            compress_kv.stride(0), compress_kv.stride(1), compress_kv.stride(2),  #
            q_rope.stride(0), q_rope.stride(1), q_rope.stride(2), q_rope.stride(3),  #
            k_rope.stride(0), k_rope.stride(1), k_rope.stride(2), k_rope.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            Z, H, M, N, KV_LORA_RANK, V_HEAD_DIM, ROPE_HEAD_DIM, NOPE_HEAD_DIM, #
            BLOCK_M, BLOCK_N,
            num_stages=1, num_warps=8
            )
        return o
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        return None,None,None,None,None,None,None
    
triton_mla = MLAOPS.apply

def torch_mla(qk_merge, compress_kv, v, q_rope, k_rope, scale, v_head_dim, attention_mask=None):
    device = v.device
    dtype = v.dtype
    q_len = qk_merge.shape[-2]
    kv_len = k_rope.shape[-2]
    min_value = torch.finfo(dtype).min
    v = torch.nn.functional.linear(compress_kv, v)
    v = v.view(qk_merge.shape[0], kv_len, -1, v_head_dim).transpose(1,2)
    score1 = qk_merge @ compress_kv.unsqueeze(1).transpose(-1,-2)
    score2 = q_rope @ k_rope.transpose(-1,-2)
    if q_len > 1:
        causal_mask:torch.tensor = torch.full((q_len, kv_len), min_value, dtype=dtype, device=device).triu(diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(v.shape[0], -1, -1, -1)
        if attention_mask is not None:
            causal_mask:torch.tensor = causal_mask.masked_fill((1-attention_mask[:, None, None, :]).bool(), min_value)
    else:
        if attention_mask is None:
            causal_mask = 0
        else:
            causal_mask = torch.zeros_like(attention_mask).masked_fill((1-attention_mask).bool(), min_value)
            causal_mask = causal_mask[:, None, None, :]
    score = (score1 + score2) * scale + causal_mask
    o = torch.nn.functional.softmax(score, dim=-1, dtype=torch.float32).to(v.dtype) @ v
    return o

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.float16
    bs,num_head,q_len,kv_len, rope_head_dim,nope_head_dim=8,8,256,256, 32, 64
    kv_lora_rank = 256
    scale = (rope_head_dim + nope_head_dim) ** (-0.5)
    qk_merge = torch.randn(bs,num_head,q_len,kv_lora_rank).to(dtype).to(device)
    compress_kv = torch.randn(bs,kv_len,kv_lora_rank).to(dtype).to(device)
    v = torch.nn.Linear(kv_lora_rank, num_head*nope_head_dim).to(qk_merge.device).to(qk_merge.dtype)
    q_rope = torch.randn(bs,num_head,q_len,rope_head_dim).to(dtype).to(device)
    k_rope = torch.randn(bs,1,kv_len,rope_head_dim).to(dtype).to(device)
    mask = torch.ones(bs, kv_len, device=device, dtype=dtype)
    # mask[:, :] = 0
    padding = mask.shape[-1] - mask.sum(-1)
    a = triton_mla(qk_merge, compress_kv, v.weight, q_rope, k_rope, scale, nope_head_dim, padding)
    b = torch_mla(qk_merge, compress_kv, v.weight, q_rope, k_rope, scale, nope_head_dim, mask)
    print(torch.allclose(a, b, 0.001, 0.001))
    print(torch.allclose(a, b, 0.005, 0.005))