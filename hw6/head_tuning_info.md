# Head-Tuning Parameter Count

## Number of Tuned Parameters

For head-tuning on BoolQ (binary classification), only the classifier head is trained:

- **Classifier**: `nn.Linear(hidden_size, num_labels)` where `num_labels = 2`
- **Weight matrix**: `hidden_size × num_labels = d × 2 = 2d`
- **Bias**: `num_labels = 2`
- **Total tuned parameters**: `2d + 2`

Where `d` is the hidden size of the model.

### Example: RoBERTa-base
- Hidden size `d = 768`
- Number of tuned parameters: `2 × 768 + 2 = 1538` parameters

Compared to full fine-tuning which would train all ~125M parameters of RoBERTa-base.

