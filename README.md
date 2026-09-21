# IRIS-LLM
IRIS LLM Architecture (PThNEA) description:

Introduction:

The current growth of LLM proficiency is about 3.5x every year while, on the other hand, the growth of hardware that husbands the training of LLMs only grows about 1.6x. It has been really hard to train LLMs because of the massive compute costs and the limitations of hardware, resulting in AI engineers inventing more efficient and accurate architectures. OpenAi alone spends 9 billion dollars in AI development and training every year and still hasn't been able to recover the 4 billion dollar burn. In this work, I, Hardik Sharma Phuyal, a 14-year-old researcher from Nepal, investigate this problem and propose a LLM architecture "Proficient Transformer Neural Efficient Architecture” or (PthNEA) aimed at reducing training compute costs while maintaining or improving performance. My goal is to make advanced AI models more accessible and affordable, enabling innovation and creativity for people around the world regardless of their computational resources.

Model Description:
My LLM architecture PThNEA has two different LLMs which have been trained solely on a GTX 1650 4GB Vram 16 GB DDR4 Ram Intel core i5 10th gen laptop. The LLMs are IRIS001, a 50 million parameter model and IRIS002, a 150 million parameter model. The model consists of embedding dimensions of 768, transformer layers of 12 and an attention head size of 12.

What I tried to improve:

SwiGLU Activation function:
My LLM needed accuracy as well as speed because even if I did manage to make a super efficient language model, if it couldn't respond to a simple form of “Hello” text it wouldn’t really be an LLM. Thus, from my research I found that the best Activation function that would help my model learn complex weights was SwiGLU because of its superior performance and enhanced training stability.

Positional Embedding to Rotary Positional Embeddings (RoPE):
During my research I found positional embeddings also require much compute in order to be calculated and used in a LLM model, thus to make my model super efficient I used RoPE  because it can encode relative position information naturally within the existing attention mechanism, without adding any significant computational overhead.

Use of FP16 for training and FP32 for loss calculation and updating weights:
While starting this project I only had a GTX 1650 in my 5 year old laptop which is a gpu dating many generations back. I knew that one of if not the most important hardware requirement for training LLMs is the GPU Vram and in my case I only had a 4 gb Vram card so being memory efficient was key. That is why I had to use mixed precision FP16 for my model to be able to train on my hardware. I used FP32 or loss calculation and updating weights to still not compromise the accuracy of the LLM.
Use of RMSnorm:
I used a simpler normalization layer, the RMS norm to reduce calculation compute because RMS norm simplifies the layer normalization by removing the mean and variance calculations. It is computationally efficient and also scalable for modern transformer architectures. It is also blazing fast as it speeds the training and gives similar or even better performances to the normal layer normalization.
	
Custom Efficient Self-Attention:
My attention mechanism is designed to be as lightweight and efficient as possible while still keeping the full power of modern transformer attention. I use Grouped-Query Attention (GQA) so that only a small number of key/value heads are computed, and the rest of the query heads reuse them. This drastically cuts computation and memory overhead because generating K and V is the most expensive part of attention. I also apply RoPE using precomputed cosine and sine values, which means positional encoding is basically free no repetitive recomputation every forward pass. The attention calculation itself is kept simple and clean using batched operations, and everything stays in a contiguous format so PyTorch doesn’t waste time rearranging tensors. By avoiding unnecessary concatenations, transposes, or oversized projections, the whole mechanism ends up being way faster and more efficient while still behaving like full multi-head attention.

Install
bash
# CUDA 11.8 build works on Turing (SM75) and is lighter than the cu12x wheels
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy tiktoken
pip install bitsandbytes   # optional, for --opt-8bit
1. Data
bash
# your own text
python prepare_data.py --input data/corpus.txt --out-dir data --val-frac 0.001

# or a streamed HF dataset (good starter corpus for this size of model)
python prepare_data.py --hf-dataset roneneldan/TinyStories --text-key text --out-dir data

# no internet? byte-level tokenizer, vocab 257, zero downloads
python prepare_data.py --input data/corpus.txt --out-dir data --tokenizer byte

Output is train.bin / val.bin: flat uint16 token streams that training memory-maps, so the corpus never has to fit in your 16 GB of RAM.

2. Train
bash
# IRIS002 (~142M) on 4 GB
python train.py --model iris002 --data-dir data \
  --seq-len 512 --micro-bs 2 --grad-accum 16 \
  --grad-checkpoint --max-steps 40000 --lr 3e-4

# IRIS001 (~50M), roomier and about 2.5x faster
python train.py --model iris001 --data-dir data \
  --seq-len 512 --micro-bs 6 --grad-accum 6 --max-steps 40000 --lr 6e-4

Resume anytime: --resume checkpoints/iris002/last.pt. Ctrl-C saves first.

Fitting in 4 GB

Rough budget for IRIS002 at seq 512, micro-batch 2:

item	memory
fp32 master weights	0.57 GB
fp16 autocast copies	0.28 GB
gradients (fp32)	0.57 GB
AdamW state (m, v)	1.14 GB (0.57 GB with --opt-8bit)
activations, checkpointed	~0.4 GB

If you OOM, apply in this order: --grad-checkpoint, then --micro-bs 1 --grad-accum 32, then --seq-len 256, then --opt-8bit, then --model iris001. Effective batch is micro_bs × grad_accum × seq_len; keep it near 16k–32k tokens per step and only the resident batch shrinks.

Notes for this specific card: TU117 has no tensor cores, so fp16 buys you memory rather than raw matmul throughput — but memory is the binding constraint here, so it is still the right call. bf16 does not exist on Turing; the script detects this and falls back. --compile is usually a net loss on this GPU.

3. Generate
bash
python inference.py --ckpt checkpoints/iris002/best.pt --prompt "Once upon a time"
python inference.py --ckpt checkpoints/iris002/best.pt --chat
python inference.py --ckpt checkpoints/iris002/best.pt --prompt "Hello" --bench

Controls: --temperature (0 = greedy), --top-k, --top-p, --repetition-penalty, --num-samples, --seed, --no-stream. Use the same --tokenizer you used in prepare_data.py.

Architecture as implemented
component	choice	why
Norm	RMSNorm, reduction in fp32	one reduction instead of two; fp32 keeps fp16 training stable
Position	RoPE, precomputed cos/sin	relative position inside attention, no extra parameters, no recompute
Attention	GQA (12 Q heads / 4 KV heads) + SDPA	3x smaller K/V projections and a 3x smaller KV cache
FFN	SwiGLU	better loss per parameter than GELU MLP
Head	tied to embedding	saves 38 M parameters at this vocab
Precision	fp16 forward/backward, fp32 loss + weights + step	the mixed-precision scheme from the design doc






