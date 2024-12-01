from typing import Any, Dict, Tuple
from torch import Tensor
from modeling_minicpm import *
# from mla_triton import mla_ops
from mla_triton import triton_mla


class MiniCPMTrain2Test(nn.Module):
    def __init__(self):
        super().__init__()
    
    def train2test(self):
        # 不可逆的操作,因为qk_merge是两个矩阵的乘法的结果，无法从结果倒推回两个矩阵。
        # 或者不删除原始的 self.q_b_proj， self.kv_b_proj，但是会增大显存开销

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        q_b_weight = self.q_b_proj.weight.data.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, self.q_lora_rank)
        q_b_nope_weight, q_b_rope_weight = q_b_weight.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=1)
        q_b_rope_weight = q_b_rope_weight.reshape(self.num_heads * self.qk_rope_head_dim, self.q_lora_rank)
        
        kv_b_weight = self.kv_b_proj.weight.data.reshape(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        k_b_nope_weight, v_b_weight = kv_b_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        v_b_weight = v_b_weight.reshape(self.num_heads * self.v_head_dim, self.kv_lora_rank)

        qk_megre_nope_weight = torch.einsum('hdq,hdk->hkq', q_b_nope_weight, k_b_nope_weight).reshape(self.num_heads * self.kv_lora_rank, self.q_lora_rank)

        self.qk_merge_nope = nn.Linear(self.q_lora_rank, self.num_heads * self.kv_lora_rank, bias=False).to(device).to(dtype)
        self.qk_merge_nope.weight.data.copy_(qk_megre_nope_weight)

        self.q_b_rope = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False).to(device).to(dtype)
        self.q_b_rope.weight.data.copy_(q_b_rope_weight)

        self.v_b = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False).to(device).to(dtype)
        self.v_b.weight.data.copy_(v_b_weight)

        del self.q_b_proj
        del self.kv_b_proj
        print(f'layer {self.layer_idx}: 替换成功, train -> test')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class TorchMLAMiniCPMAttention(MiniCPMAttention, MiniCPMTrain2Test):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        q_a_norm = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_rope = self.q_b_rope(q_a_norm).view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1,2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, key_rope_states = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        key_rope_states.unsqueeze_(1)

        kv_seq_len = compressed_kv.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(compressed_kv, seq_len=kv_seq_len)
        q_rope, key_rope_states = apply_rotary_pos_emb(q_rope, key_rope_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            compressed_kv, key_rope_states = past_key_value.update(
                compressed_kv, key_rope_states, self.layer_idx, cache_kwargs
            )

        nope_attn_score = self.qk_merge_nope(q_a_norm).view(bsz, q_len, self.num_heads, self.kv_lora_rank).transpose(1,2)
        nope_attn_score = torch.matmul(nope_attn_score, compressed_kv.unsqueeze(1).transpose(-1,-2))

        rope_attn_score = torch.matmul(q_rope, key_rope_states.transpose(-1,-2))

        attn_weights = (nope_attn_score + rope_attn_score) * self.softmax_scale

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_rope.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, compressed_kv.unsqueeze(1))
        v_weight = self.v_b.weight.view(-1, self.v_head_dim, self.kv_lora_rank).unsqueeze(0)
        attn_output = torch.matmul(attn_output, v_weight.transpose(-1, -2))

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class TritonMLAMiniCPMAttention(MiniCPMAttention, MiniCPMTrain2Test):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        q_a_norm = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_rope = self.q_b_rope(q_a_norm).view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1,2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, key_rope_states = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        key_rope_states.unsqueeze_(1)

        kv_seq_len = compressed_kv.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(compressed_kv, seq_len=kv_seq_len)
        q_rope, key_rope_states = apply_rotary_pos_emb(q_rope, key_rope_states, cos, sin, position_ids)
        # key_rope_states = key_rope_states.expand(-1, self.num_heads, -1, -1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            compressed_kv, key_rope_states = past_key_value.update(
                compressed_kv, key_rope_states, self.layer_idx, cache_kwargs
            )

        # value_states = self.v_b(compressed_kv).view(bsz, kv_seq_len, self.num_heads, self.v_head_dim).transpose(1,2)

        qk_merge = self.qk_merge_nope(q_a_norm).view(bsz, q_len, self.num_heads, self.kv_lora_rank).transpose(1,2)
        # key_rope_states = key_rope_states.expand(-1, self.num_heads, -1, -1)
        attn_output = triton_mla(qk_merge, compressed_kv,  
                              self.v_b.weight.data, q_rope, key_rope_states, self.softmax_scale, self.v_head_dim, attention_mask)
        # print(attn_output.shape)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    
