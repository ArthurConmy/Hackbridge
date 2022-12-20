import torch as t
import torch.nn as nn

from utils import allclose, allclose_atol, report, allclose_scalar_atol


def _copy(mine, theirs):
    mine.detach().copy_(theirs)


def _copy_weight_bias(mine, theirs, transpose=False):
    _copy(mine.weight, theirs.weight.T if transpose else theirs.weight)
    if getattr(mine, "bias", None) is not None:
        _copy(mine.bias, theirs.bias)


def _copy_self_attn_weights(yours, ref):
    _copy_weight_bias(yours.project_query, ref.project_query)
    _copy_weight_bias(yours.project_key, ref.project_key)
    _copy_weight_bias(yours.project_value, ref.project_value)
    _copy_weight_bias(yours.project_output, ref.project_output)


@report
def test_attention_pattern_pre_softmax(BertSelfAttention, batch_size=2, seq_len=5, hidden_size=6, num_heads=2):
    """The attention pattern should exactly match the reference solution."""
    import w2d1_solution

    config = w2d1_solution.BertConfig(hidden_size=hidden_size, num_heads=num_heads, head_size=3)
    ref = w2d1_solution.BertSelfAttention(config)
    yours = BertSelfAttention(config)
    yours.load_state_dict(ref.state_dict())

    x = t.randn(batch_size, seq_len, hidden_size)
    expected = ref.attention_pattern_pre_softmax(x)
    actual = yours.attention_pattern_pre_softmax(x)
    allclose(actual, expected)


@report
def test_attention(BertSelfAttention, batch_size=2, seq_len=5, hidden_size=6, num_heads=2):
    """The attention layer's output should exactly match the reference solution.

    Dropout is not tested!
    """
    import w2d1_solution

    config = w2d1_solution.BertConfig(hidden_size=hidden_size, num_heads=num_heads, head_size=3, dropout=0)
    ref = w2d1_solution.BertSelfAttention(config)
    yours = BertSelfAttention(config)
    yours.load_state_dict(ref.state_dict())

    x = t.randn(batch_size, seq_len, hidden_size)
    expected = ref(x)
    actual = yours(x)
    allclose(actual, expected)


def _copy_bert_mlp_weights(yours, ref):
    _copy_weight_bias(yours.first_linear, ref.first_linear)
    _copy_weight_bias(yours.second_linear, ref.second_linear)
    _copy_weight_bias(yours.layer_norm, ref.layer_norm)


@report
def test_bert_mlp_zero_dropout(BertMLP, batch_size=2, seq_len=5, hidden_size=6):
    """The MLP's output should exactly match the reference solution."""
    import w2d1_solution

    config = w2d1_solution.BertConfig(
        hidden_size=hidden_size,
        intermediate_size=3 * hidden_size,
        dropout=0,
    )
    ref = w2d1_solution.BertMLP(config)
    yours = BertMLP(config)
    yours.load_state_dict(ref.state_dict())

    ref.eval()
    yours.eval()
    x = t.randn(batch_size, seq_len, hidden_size)
    expected = ref(x)
    actual = yours(x)
    allclose(actual, expected)


@report
def test_bert_mlp_one_dropout(BertMLP, batch_size=2, seq_len=5, hidden_size=6):
    """With a dropout of 1.0, the output should just be layer_norm(x)."""
    import w2d1_solution

    config = w2d1_solution.BertConfig(
        hidden_size=hidden_size,
        intermediate_size=3 * hidden_size,
        dropout=1.0,
    )
    yours = BertMLP(config)
    yours.train()
    x = t.randn(batch_size, seq_len, hidden_size)
    expected = yours.layer_norm(x)  # TBD: expected is just zeros yeah?
    actual = yours(x)
    allclose(actual, expected)


@report
def test_layernorm_mean_1d(LayerNorm):
    """If an integer is passed, this means normalize over the last dimension which should have that size."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10)
    out = ln1(x)
    max_mean = out.mean(-1).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"


@report
def test_layernorm_mean_2d(LayerNorm):
    """If normalized_shape is 2D, should normalize over both the last two dimensions."""
    x = t.randn(20, 10)
    ln1 = LayerNorm((20, 10))
    out = ln1(x)
    max_mean = out.mean((-1, -2)).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"


@report
def test_layernorm_std(LayerNorm):
    """If epsilon is small enough and no elementwise_affine, the output variance should be very close to 1."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10, eps=1e-11, elementwise_affine=False)
    out = ln1(x)
    var_diff = (1 - out.var(-1, unbiased=False)).abs().max().item()
    assert var_diff < 1e-6, f"Var should be about 1, off by {var_diff}"


@report
def test_layernorm_exact(LayerNorm):
    """Your LayerNorm's output should match PyTorch for equal epsilon, up to floating point rounding error.

    This test uses float64 and the result should be extremely tight.
    """
    x = t.randn(2, 3, 4, 5, dtype=t.float64)
    # Use large epsilon to make sure it fails if they forget it
    ln1 = LayerNorm((5,), dtype=t.float64, eps=1e-2)
    ln2 = t.nn.LayerNorm((5,), dtype=t.float64, eps=1e-2)  # type: ignore
    actual = ln1(x)
    expected = ln2(x)
    allclose(actual, expected)


@report
def test_layernorm_backward(LayerNorm):
    """The backwards pass should also match PyTorch exactly."""
    x = t.randn(10, 3)
    x2 = x.clone()
    x.requires_grad_(True)
    x2.requires_grad_(True)

    # Without parameters, should be deterministic
    ref = nn.LayerNorm(3, elementwise_affine=False)
    ref.requires_grad_(True)
    ref(x).sum().backward()

    ln = LayerNorm(3, elementwise_affine=False)
    ln.requires_grad_(True)
    ln(x2).sum().backward()
    # Use atol since grad entries are supposed to be zero here
    assert isinstance(x.grad, t.Tensor)
    assert isinstance(x2.grad, t.Tensor)
    allclose_atol(x.grad, x2.grad, atol=1e-5)


def _copy_bert_block_weights(theirs, reference):
    _copy_self_attn_weights(theirs.attention.self_attn, reference.attention.self_attn)
    _copy_weight_bias(theirs.attention.layer_norm, reference.attention.layer_norm)
    _copy_bert_mlp_weights(theirs.mlp, reference.mlp)


@report
def test_bert_attention_dropout(BertAttention):
    """With a dropout of 1.0, BertAttention should reduce to layer_norm(x)."""
    import w2d1_solution

    config = w2d1_solution.BertConfig(dropout=1.0)
    yours = BertAttention(config)
    x = t.rand((2, 3, 768))
    yours.train()
    actual = yours(x)
    expected = yours.layer_norm(x)
    allclose(actual, expected)


@report
def test_bert_block(BertBlock):
    """Your BertBlock should exactly match the reference solution in eval mode.

    Dropout is not tested.
    """
    import w2d1_solution

    config = w2d1_solution.BertConfig()
    ref = w2d1_solution.BertBlock(config)
    yours = BertBlock(config)
    yours.load_state_dict(ref.state_dict())

    x = t.rand((2, 3, 768))
    ref.eval()
    yours.eval()
    allclose(yours(x), ref(x))


@report
def test_embedding(Embedding):
    """Indexing into the embedding should fetch the corresponding rows of the embedding."""
    emb = Embedding(6, 100)
    out = emb(t.tensor([1, 3, 5], dtype=t.int64))
    allclose(out[0], emb.weight[1])
    allclose(out[1], emb.weight[3])
    allclose(out[2], emb.weight[5])


@report
def test_embedding_std(Embedding):
    """The standard deviation should be roughly 0.02."""
    t.manual_seed(5)
    emb = Embedding(6, 100)
    allclose_scalar_atol(emb.weight.std().item(), 0.02, atol=0.001)


@report
def test_bert(your_module):
    """Your full Bert should exactly match the reference solution in eval mode.

    Dropout is not tested.
    """
    import w2d1_solution

    config = w2d1_solution.BertConfig()
    reference = w2d1_solution.BertLanguageModel(config)
    reference.eval()
    theirs = your_module(config)
    theirs.eval()
    theirs.load_state_dict(reference.state_dict())
    input_ids = t.tensor([[101, 1309, 6100, 1660, 1128, 1146, 102]], dtype=t.int64)
    allclose(theirs(input_ids=input_ids), reference(input_ids=input_ids))


@report
def test_bert_prediction(predict, model, tokenizer):
    """Your Bert should know some names of American presidents."""
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]