# Roadmap

- **Formats and models:** Q6_K, Q8_0 and Q5_0 are native and four families are
  benchmarked (Gemma 4, Qwen2.5-0.5B, DeepSeek-R1-Distill-Qwen-1.5B,
  Qwen2.5-7B-Instruct-1M); next are further families (Llama, Mistral) and the
  quant formats they ship with
- **Speed:** decode is ahead of llama.cpp on the d64 family and within 0.92× on
  Qwen2.5-7B at 4K; prefill is ahead on Gemma 4 and behind on the d128
  families, most visibly on 8K+ prompts. Next are a batched d128 matmul path
  and the prefill work it enables
- **Speculative decoding:** shipped in v0.0.6 through the C ABI, Guile and the
  OpenAI-compatible server, on by default with a long-match drafter and
  bit-identical output; next is widening the verify envelope on more workloads
- **Compiler:** GPU code generation for fused elementwise and norm chains is
  shipped; next are GPU-independent targeting (compute capability discovered
  at run time, SPIR-V for a second vendor) and whole-model lowering for traced
  graphs
- **Research route:** connect the Guile tracer to the core and compile traced
  graphs through the C ABI
- **Usability:** sampling and constrained decoding shipped in v0.0.4; next are
  the remaining JSON-Schema keywords and chat templates for further families
- **Scale:** larger MoE models with expert offload
- **Beyond LLMs:** diffusion models, with a converter for existing ComfyUI
  workflows
- **Distribution:** Guix packaging, CPU first. A Vulkan backend is planned so
  that a fully free-software stack also gets GPU acceleration. The backend
  scaffold is already in the tree (`-DSONICBOOM_USE_VULKAN=ON`).
