# 🧱 Step-by-Step: Build a Small Language Model With the GPT-OSS Blueprint

A hands-on series that builds a small language model **from scratch**, following the architecture of OpenAI's open-weight [GPT-OSS](https://github.com/openai/gpt-oss) models, and trains it on a laptop. Every notebook explains each idea in plain language first, then in code.

## Notebooks

| # | Notebook | What you build | Runtime on an Apple-silicon Mac |
|---|---|---|---|
| 01 | [01_mini_gpt_oss_from_scratch.ipynb](01_mini_gpt_oss_from_scratch.ipynb) | Tokens → embeddings → RMSNorm → RoPE → grouped-query attention with sinks and sliding windows → Mixture of Experts with SwiGLU → full model → training → text generation | ~5 minutes |
| 02 | [02_visualizing_the_mini_gpt_oss.ipynb](02_visualizing_the_mini_gpt_oss.ipynb) | An X-ray of the trained model: where the parameters live, the path a sentence takes, input vs output character cards, RoPE clock hands, every attention head, sink usage, what the model stares at before guessing, router choices and expert specialisation, the SwiGLU gate, the logit lens (guess sharpening floor by floor), temperature, a step-by-step generation trace, and per-character surprise | seconds (needs the checkpoint from 01, or trains one in ~4 minutes) |

[mini_gpt_oss.py](mini_gpt_oss.py) holds the model code from notebook 01 as an importable module (plus a few `record` switches used by notebook 02).

The mini model has about 4 million parameters (gpt-oss-20b has 21 billion) and uses the same names and structure as the reference implementation in `gpt_oss/torch/model.py`, so every class maps directly onto the real thing.

## Running locally

You need Python 3.10+ with PyTorch and matplotlib. On a Mac the code uses the GPU through Metal (MPS) automatically; it also runs on CUDA or CPU.

```bash
# Option A: a fresh environment
conda create -n gptoss-mini python=3.11 -y
conda activate gptoss-mini
pip install torch matplotlib jupyter ipykernel
python -m ipykernel install --user --name gptoss-mini --display-name "Python (gptoss-mini)"

# Option B: an existing environment that already has torch + matplotlib
pip install ipykernel
python -m ipykernel install --user --name <env-name> --display-name "Python (<env-name>)"
```

Then open the notebook in VS Code or JupyterLab and pick that kernel. The first run downloads Tiny Shakespeare (1 MB) into `step-by-step/data/`, which is git-ignored along with the saved checkpoint.

Optional: `pip install tiktoken` to see how GPT-OSS's real `o200k_harmony` tokenizer splits text.

## Coming next

- A KV cache so generation does not re-read the whole prompt for every token
- A sub-word tokenizer instead of characters
- The Harmony chat format used by GPT-OSS
