import torch
from transformers import LlamaConfig, LlamaForCausalLM

from deltaomni.continuous_kv import ContinuousKVRunner


def _model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    ).eval()


def test_incremental_kv_logits_match_one_causal_concatenated_forward() -> None:
    model = _model()
    runner = ContinuousKVRunner(model, position_axes=1)
    ids = torch.tensor([[1, 5, 7, 3, 9, 2]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
        first, state = runner.append(state=None, input_ids=ids[:, :2])
        second, state = runner.append(state=state, input_ids=ids[:, 2:5])
        third, state = runner.append(state=state, input_ids=ids[:, 5:])

    actual = torch.cat((first, second, third), dim=1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert state.tokens == ids.shape[1]
    assert state.attention_mask.shape == ids.shape


def test_prior_caption_tokens_remain_in_state_for_the_next_commit() -> None:
    model = _model()
    runner = ContinuousKVRunner(model, position_axes=1)
    prefix = torch.tensor([[1, 4, 8]])
    first_caption = torch.tensor([[6, 6, 2]])
    changed_caption = torch.tensor([[7, 7, 2]])
    next_update = torch.tensor([[10, 11]])

    _, state = runner.append(state=None, input_ids=prefix)
    _, state = runner.append(state=state, input_ids=first_caption)
    logits, state = runner.append(state=state, input_ids=next_update)

    _, changed_state = runner.append(state=None, input_ids=prefix)
    _, changed_state = runner.append(state=changed_state, input_ids=changed_caption)
    changed_logits, changed_state = runner.append(state=changed_state, input_ids=next_update)

    assert state.tokens == prefix.shape[1] + first_caption.shape[1] + next_update.shape[1]
    assert changed_state.tokens == state.tokens
    assert not torch.allclose(logits[:, -1], changed_logits[:, -1])


def test_generated_end_token_is_appended_before_the_next_chunk() -> None:
    model = _model()
    runner = ContinuousKVRunner(model, position_axes=1)
    _, state = runner.append(state=None, input_ids=torch.tensor([[1, 3]]))
    forced = torch.zeros(1, 1, 32)
    forced[:, :, 2] = 1

    generated, next_state = runner.greedy_append(
        forced,
        state,
        end_token_id=2,
        max_new_tokens=4,
    )

    assert generated.tolist() == [[2]]
    assert next_state.tokens == state.tokens + 1
