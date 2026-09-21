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






