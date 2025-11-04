# Prefix-Tuning Parameter Count

## Number of Tuned Parameters

For prefix-tuning on BoolQ (binary classification) with prefix length 128:

- **Prefix embeddings**: `prefix_length × hidden_size = 128 × d`
- **Classifier**: `nn.Linear(hidden_size, num_labels)` where `num_labels = 2`
  - Weight matrix: `hidden_size × num_labels = d × 2 = 2d`
  - Bias: `num_labels = 2`
- **Total tuned parameters**: `prefix_length × d + 2d + 2 = 128d + 2d + 2 = 130d + 2`

Where `d` is the hidden size of the model.

### Example: RoBERTa-base
- Hidden size `d = 768`
- Prefix length: `128`
- Prefix parameters: `128 × 768 = 98,304`
- Classifier parameters: `2 × 768 + 2 = 1,538`
- **Total tuned parameters**: `98,304 + 1,538 = 99,842` parameters

Compared to:
- Full fine-tuning: ~125M parameters
- Head-tuning: 1,538 parameters
- Prefix-tuning: 99,842 parameters (more than head-tuning but much less than full fine-tuning)

