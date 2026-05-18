# Attention-from-scratch

This project is a small from-scratch walkthrough of self-attention in Python.

What is included:

- `src/attention.py` builds a tiny attention example with NumPy.
- `src/visualize.py` plots the resulting attention weights as a heatmap with Matplotlib.
- The token list and attention matrix are now kept consistent so the visualization runs correctly.

What the code does:

1. `src/attention.py` starts with a list of tokens and creates random embeddings for them.
2. It creates Query, Key, and Value weight matrices.
3. It multiplies embeddings by those matrices to get Q, K, and V.
4. It computes attention scores with `Q @ K.T`.
5. It scales the scores and applies softmax to get attention weights.
6. It uses the attention weights to build the final output representation.
7. `src/visualize.py` imports the attention weights and tokens and shows them as a heatmap.

Why this matters:

- It shows the basic idea behind transformer self-attention.
- Each token can look at the other tokens and decide which ones matter most.
- The heatmap makes those relationships easy to see.

How to run it:

1. Activate the virtual environment.
2. Run `python src/attention.py` to print the intermediate attention values.
3. Run `python src/visualize.py` to display the attention heatmap.

The visualizer imports the attention output directly, so there is no need to manually define `attention_weights` or `tokens` in the plot script.