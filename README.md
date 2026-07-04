# Inference Engineering — Crisp Notes
*Book: "Inference Engineering" by Philip Kiely (Baseten), Jan 2026*

---

## Ch 0 — Inference (Overview)

- **Training** = learning weights. **Inference** = serving models in production. Gen-AI inference is much harder than classic ML inference.
- Three layers of a complete inference stack:
  1. **Runtime** – performance of a single model on a single GPU/instance.
  2. **Infrastructure** – scaling across clusters/regions/clouds with uptime.
  3. **Tooling** – right abstraction level for engineers (black-box API ↔ raw compute).

---

## Ch 1 — Prerequisites

- Optimization = tradeoffs among **latency, throughput, quality** — not maximizing one thing. More constraints on your use case → better-optimized system.
- Know: model requirements, application interface, latency budget, unit economics, usage patterns.

**Shared vs Dedicated inference**
| | Shared (pay-per-token API) | Dedicated (own GPUs) |
|---|---|---|
| Pros | zero overhead, no cold starts, minimal eng work | full control of latency/quality/uptime |
| Cons | cost scales linearly, no control, provider-capped uptime | high fixed cost, more eng surface area |

- Switch to dedicated when: **Scale** (cheaper per-GPU than per-token), **Specialization** (custom/fine-tuned model, strict SLAs), **Orchestration** (multi-model pipelines).
- Early stage → use off-the-shelf APIs. Only build dedicated inference once there's a clear business need.

**About your app**
- Foundation models & inference platforms need max generality; vertical AI apps should add constraints.
- App categories: Agents, Chat, Voice, Media, Search, RecSys, Completion, Moderation — each has different latency/throughput priorities.
- **Online vs Offline**: online = optimize latency (chat, voice, completion); offline = optimize throughput (transcription batch jobs, embeddings, corpus prep). Same model may need 2 separate deployments.
- **Consumer vs B2B**: consumer = cost-sensitive, spiky, virality; B2B = latency/uptime-sensitive, more predictable. Compliance: data sovereignty, user privacy, regulatory compliance.

**Model selection**
- Smaller model = faster & cheaper, all else equal. Choosing the *right model* matters more than choosing the inference engine.
- Early stage: use frontier APIs. At scale: find smallest model that's "smart enough."
- **Evals** (product-specific) vs **benchmarks** (general, e.g. MMLU) — benchmarks get gamed (Goodhart's Law); evals establish your quality baseline before optimizing speed.
- **Fine-tuning**: adapt pretrained model with new data for domain quality (e.g., text-to-SQL: tiny fine-tuned model ≈ giant general model).
- **Distillation**: train small "student" on large "teacher"'s probability distributions (not just outputs). Less common than fine-tuning in practice (e.g., DeepSeek-R1 distills onto Llama/Qwen).

**Latency & throughput metrics**
- **TTFT** (time to first token) — compute-bound prefill, lower = better.
- **TPS** (tokens/sec) — memory-bound decode. Ambiguous term: use **Perceived TPS** (per-user latency) vs **Total TPS** (system throughput) vs **ITL** (inter-token latency; 10ms ITL = 100 TPS/user).
- **Percentiles**: P50/P90/P95/P99 matter more than mean (latency distribution is right-skewed). Good perf work targets P90/P99, not just average.
- **End-to-end vs inference-only** latency: if inference is fast but E2E is slow → look at infra/network, not model optimization.

---

## Ch 2 — Models

- History: perceptrons (50s) → backprop/MLP → deep nets (2012 AlexNet) → **Transformers** ("Attention Is All You Need," 2017).
- Two model archetypes across all modalities:
  - **Autoregressive token generation** (LLMs): predict next token from sequence.
  - **Iterative denoising** (diffusion, image/video): refine noise → output.

**Neural network basics**
- Node = weighted sum + bias. Layer = group of nodes. Input/Hidden/Output layers; hidden layer outputs = "hidden states."
- **Encoder** builds internal representation; **Decoder** generates output from it. Modern LLMs = decoder-only. Whisper = encoder-decoder.
- **Matmul/Linear layer**: y = Wx + b. Stacking linear layers alone collapses into one (composability) → need **activation functions** (ReLU, SiLU, SwiGLU) to break linearity & enable depth.

**LLM inference mechanics**
- Tokens = subword units; vocab usually 100K+. Tokenizer = simple string↔token map (no NN).
- Sequences: Input → (optional Reasoning) → Output, bounded by **context window**.
- **Chat template** combines multi-turn/tool/multimodal input into one sequence — must match model exactly.
- Two inference phases:
  - **Prefill**: process input, build KV cache — compute-bound, determines TTFT.
  - **Decode**: autoregressive forward passes generating tokens — memory-bound, determines TPS.
- Output = logits vector (size = vocab) → normalized → sampled via **temperature / top-k / top-p**. Temp=0 or top-k=1 = deterministic.
- **Model architecture** (config.json) encodes family/version/MoE-or-not/CausalLM. Same architecture → same optimized runtime/engine support works across sizes/variants/fine-tunes (LoRA doesn't change architecture).
- **Transformer block** = Attention + FFN (MLP, majority of weights) + Normalization.
- **Attention**: Q (current token), K (all prior tokens), V (values). Self-attention (LLMs, causal-masked) vs Cross-attention (image/text conditioning). Quadratic in sequence length in theory, but **KV cache** makes decode linear-time by avoiding recomputation.
- **MoE (Mixture of Experts)**: sparse — many small "expert" matrices, router activates only a few per token/layer (e.g., Qwen3-235B-A22B → 22B active of 235B total). Great for single-request local inference; in production batching, most experts get touched anyway (unless Expert Parallelism at scale). MoE common for 100B+ models; dense architecture better under ~32B (esp. <8B).

**Image/video generation**
- Pipeline of models: **Text encoder** → **Denoising model** (core) → **VAE** (latent↔pixel space). Plus LoRAs (style) & ControlNets (structure).
- Runs in **latent space** (low-dim, e.g. 128×128 vs 1024×1024 pixels) — feasible for full-image attention.
- 30–50 denoising steps typical; each step = 2 forward passes (conditioned + unconditioned) combined via **guidance scale** → 50 steps = 100 passes.
- Key inference args: prompt, negative prompt, steps, guidance scale, image size.
- Architecture = **Diffusion Transformer** (patches of image ↔ tokens). SDXL example: base+refiner denoisers. Modern (Qwen Image) uses full LLM text encoders + bigger denoisers.
- Trend: blending LLM-style autoregression into image gen (e.g., HunyuanImage-3.0) — single forward pass vs up to 100.
- **Few-step models**: 8-or-fewer steps via latent consistency or distillation — 80-90% faster, lower quality.
- **Video generation**: 3-5x bigger than image models; latent space adds Time dimension (X,Y,T). Framewise generation → error accumulation (bad); modern models denoise whole video at once. Batch size = 1 (full 8-GPU node per video). ~50 steps like image, but each step vastly more expensive.

**Bottleneck analysis**
- Two GPU resources: **Compute** (FLOPS) and **Memory bandwidth** (bytes/sec).
- **Arithmetic intensity** = compute ops ÷ memory bytes moved, compared to GPU's ops:byte ratio via **roofline model**:
  - Higher than ops:byte ratio → **compute-bound**.
  - Lower → **memory-bound**.
- **Prefill = compute-bound** (large batched matmul), **Decode = memory-bound** (weights reloaded per token, low compute per byte). Image/video gen = compute-bound.
- Batching makes decode less memory-bound (more compute per weight load).

**Optimizing Attention**
- Two optimization strategies: (a) **implementation improvements** (lossless, e.g. FlashAttention, PagedAttention) and (b) **new algorithms** (lossy but faster: sliding window, gated, linear, compressed, multi-latent attention).
- **FlashAttention**: hand-fused, hardware-specific kernels, minimizes memory reads/writes.
- **PagedAttention**: pages/blocks the KV cache — non-contiguous memory storage.
- **Sliding window attention**: O(N²)→O(Nw). **Mamba**: state-space model, linear scaling, no attention; hybrid transformer+Mamba models emerging (e.g. Nemotron 3 Nano).

---

## Ch 3 — Hardware

- 3 GPU classes: Datacenter (B200), Workstation (RTX Pro 6000), Personal (RTX 5090). Datacenter modes: Cloud / On-prem / Air-gapped.

**GPU architecture**
- GPUs = throughput machines (parallel), CPUs = latency machines (sequential).
- Compute types: **CUDA Core** (scalar), **Tensor Core** (matrix MMA — what matters for inference), **SFU** (sin/cos/log, e.g. softmax).
- FLOPS: Dense vs Sparse (2:4 structured sparsity ~2x). FLOPS roughly double per halving of precision.
- Memory: **VRAM** (DRAM, GBs) vs on-chip **SRAM caches** (L0/L1/L2). VRAM needs = model weights + ≥50% headroom for KV cache.
- Memory bandwidth = decode bottleneck at low/med batch; pick higher-bandwidth GPU (e.g. H200 > H100) for more TPS.

**GPU generations** (letter=architecture gen, number=SKU size)
| Arch | GPUs | Key feature |
|---|---|---|
| Hopper (2022) | H100, H200 | FP8 support, FlashAttention-3 |
| Ada Lovelace (2022) | L4, L40 | Graphics-oriented, no NVLink, cheap small-model inference |
| Blackwell (2024) | B200, B300 | FP4 + microscaling formats, FlashAttention-4, current gold standard |
| Rubin (2026, upcoming) | – | HBM4, new CPX chip for prefill |
| Feynman (2028) | – | future |

- **Grace/Vera CPUs**: NVIDIA ARM CPUs w/ high-bandwidth NVLink-C2C to GPU (900GB/s) — good for KV cache/LoRA offload to host memory.

**Instances**
- Instance = GPU(s) + CPU + host mem + storage + network + interconnect.
- **Node** = standard unit of 8 GPUs. **NVLink** (GPU-GPU, up to 1800GB/s Blackwell) + **NVSwitch** (all-to-all) within node. **InfiniBand** (node-to-node, up to 400Gb/s) — much slower than NVLink.
- **MIG (Multi-Instance GPU)**: split big GPU (A100/H100/H200/B200) into up to 7 fractional instances — good when model too small for a full GPU (e.g., small TTS/ASR models).

**Other accelerators**: AMD (MI350), AWS (Inferentia/Trainium), Google (TPU), Groq (LPU), Cerebras (WSE-3), Sambanova (RDU), etc. — each bets on memory bandwidth, power efficiency, or platform integration; CUDA moat is the shared challenge.

**Local inference**
- Pros: zero network latency, independence, privacy, cost. Cons: weak hardware, thermal limits, fragmented support, battery life.
- Apple M-series: huge unified memory (512GB) but slower bandwidth vs NVIDIA RTX (32GB, faster). MoE models & quantization favor local/edge. Mobile: Android (AI Edge SDK/Gemini Nano), iOS (Core ML/Foundation Models) — usually ≤1-2B params.

---

## Ch 4 — Software

**Stack (increasing abstraction)**: CUDA → Deep learning frameworks (PyTorch) → Inference engines (vLLM/SGLang/TensorRT-LLM) → NVIDIA Dynamo (orchestration).

**CUDA**
- CUDA kernel = parallel function on GPU; CUDA graph = DAG of kernel ops; driver = low-level HW interface; runtime = dev-facing API.
- Prebuilt kernel libraries: **cuBLAS** (BLAS/GEMM), **cuDNN** (NN primitives), **CUTLASS/CuTe** (template libs for custom high-perf kernels), **FlashInfer** (LLM-specific kernels).
- **Kernel selection** matters — kernels are hardware-specific (H100 kernel ≠ optimal on B200). Most selection is automatic (PyTorch/TensorRT compile), but manual swaps (e.g., DeepGEMM for FP8 on Hopper) can help.
- **Kernel fusion**: combine multiple kernels into one to cut redundant memory reads/writes (critical for memory-bound decode).

**Frameworks**
- **PyTorch** = dominant framework (training + inference). `torch.compile` does auto kernel selection/fusion but can't fuse custom plugin kernels (FlashAttention, DeepGEMM).
- Model file formats: **safetensors** (weights only, safe, HF standard) vs **ONNX** (weights + graph, portable).
- **ONNX Runtime** (open, multi-HW) vs **TensorRT** (NVIDIA-only, higher perf). Trend: skip straight from PyTorch to inference engines, bypassing ONNX/TensorRT compile step for LLMs.
- **Transformers / Diffusers** (HF libraries): reference implementations, not production-grade — used for learning/prototyping, not scaled serving.

**Inference engines** — vLLM, SGLang, TensorRT-LLM
| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Performance | Good | Good | Best |
| Ease of use | Easy | Easy | Hard |
| Model support | Most (day-0) | Most | Some |
| Hardware | GPU+TPU | NVIDIA+AMD | NVIDIA only |
- vLLM: broadest support, best for "just works," Omni multimodal.
- SGLang: fast + customizable, strong with DeepSeek/Qwen/Kimi MoE, large-scale multi-node (GB200 NVL72), SGLang Diffusion for image/video.
- TensorRT-LLM: best perf via NVIDIA's proprietary kernels; steep learning curve; V0 (TensorRT plugin) vs V1 (standalone PyTorch-based, 2025).

**NVIDIA Dynamo**: orchestration layer on top of engines — KV cache re-use/routing, disaggregation, multi-node parallelism. Best for big models + big traffic (foundation model APIs); unnecessary overhead for smaller deployments.

**Benchmarking & Profiling**
- Best benchmark = shadow real production traffic. Otherwise simulate matching ISL/OSL, traffic volume/pattern, request content, input params.
- Tools: SGLang Genai-bench, NVIDIA GenAI-Perf, Locust; eval datasets (MMLU, SWE-bench, HumanEval) double as realistic + quality-check inputs.
- Benchmark = "what" (e.g. P90 TTFT = 350ms); Profile = "why" (where time goes). Tools: PyTorch Profiler, NVIDIA Nsight Systems (system-wide), Nsight Compute (per-kernel).

---

## Ch 5 — Techniques

General principle: more constraints/traffic → more optimization headroom. Techniques can be **symbiotic or conflicting** (e.g., KV quantization helps disaggregation; large batches hurt speculation).

### 5.1 Quantization
- Cuts numeric precision of weights/activations/KV/attention. Speeds up **both** prefill (2x FLOPS on lower-precision Tensor Cores) and decode (half the memory traffic) — but not linearly (~30-50% gain per precision step).
- Risk: precision errors compound, esp. over long sequences.
- **Number formats** (16/8/4-bit): FP16, BF16 (higher dynamic range), FP8 (Hopper+), MXFP8 (Blackwell, microscaling), FP4/MXFP4/NVFP4 (Blackwell), INT8/INT4 (lower dynamic range, avoid for quality-sensitive prod).
- Float formats > int formats for inference due to **dynamic range** (exponent+mantissa vs just integer value) — better handles outliers.
- **Granularity**: tensor-level → channel-level → block-level scale factors. More granular = better quality, more overhead. Microscaling (MXFP8/MXFP4/NVFP4) = block-level scale factors (every 16-32 values).
- **Quantization-aware training** (during training, e.g. GPT-OSS MXFP4) vs **Post-training quantization** (PTQ — what most engineers do; tool: NVIDIA ModelOpt).
- **Sensitivity ranking (least→most)**: Weights < Activations < KV cache < Attention. Attention/softmax almost never quantized due to compounding errors.
- **Sweet spot**: FP8/MXFP8 for most production use; NVFP4 promising for more aggressive compression.
- **Quality checks**: Perplexity, intelligence benchmarks, custom evals — target "indistinguishable from noise" degradation. Quantization is a spectrum, not binary.

### 5.2 Speculative Decoding
- Exploits idle compute during memory-bound decode: generate multiple **draft tokens**, validate them in one target-model forward pass → N+1 tokens/pass. Improves TPS/ITL only, **not TTFT**.
- Performance depends on: draft token cost, draft sequence length, **token acceptance rate** (drops later in draft sequence; a single rejection kills the rest of that sequence).
- Higher temperature → lower acceptance rate (less predictable). Most useful at **low batch sizes** (spare compute); disable at high batch sizes.
- **Draft-Target**: separate small draft model (≥10x smaller) + target model verifies. Easy setup, but most overhead (extra weights/KV/compute).
- **Medusa**: extra decoder heads grafted onto target model (2-4 heads) — no separate draft model, but limited acceptance/quality; superseded by EAGLE.
- **EAGLE**: purpose-trained draft model using target model's **hidden states** (early/mid/late layers) as input; up to 8 draft tokens, high acceptance, <1B params; the current go-to.
- **N-gram speculation**: no draft model — builds n-gram dictionary from prefill context; excels when output closely mirrors input (code completion/revision); sequences can exceed 10 tokens.
- **Lookahead Decoding**: generalizes n-gram (generates n-grams during inference), more general but more compute.

### 5.3 Caching
- KV cache built in prefill, updated in decode — always used within a request by default.
- **Prefix caching**: reuse KV cache across requests sharing a prefix (skip prefill on shared tokens). Big wins for: long system prompts, code completion, doc retrieval, multi-turn chat. Cache ends at **first non-matching token** — put novel content late in context to maximize hits.
- Non-prefix KV re-use (CacheBlend, LMCache) is emerging research.
- **KV cache storage hierarchy** (fastest→slowest): G1 GPU VRAM → G2 Host RAM (CPU) → G3 Local SSD → G4 Networked SSD. Offload cold blocks to lower tiers (Dynamo's KVBM manages this).
- **Cache-aware routing**: route repeat/related requests to the same replica for cache hits; global KV cache (via G4) across replicas is an alternative.
- **Long context handling**: KV cache scales linearly w/ seq length → can dominate VRAM. Mitigations: FlashAttention, PagedAttention, Chunked Prefill, or model-level sliding/sparse/compressed attention.

### 5.4 Model Parallelism
- Needed when model+KV cache don't fit one GPU. Estimate min GPUs = f(precision, param count, KV cache headroom); often want more than minimum for bigger KV cache / better latency.
- Bottleneck = inter-GPU communication overhead ("topology-aware parallelism").
- **Pipeline Parallelism (PP)**: splits layers across GPUs — poor latency/utilization, avoid except multi-node.
- **Tensor Parallelism (TP)**: splits weights within each layer — default choice for low-latency single-node inference; needs sync (all-reduce) across GPUs, best on fast NVLink.
- **Expert Parallelism (EP)**: shards MoE experts across GPUs — improves throughput, less inter-GPU comms than TP, scales well multi-node.
- Multi-node: TP within node + PP across nodes (dense models, e.g. TP8PP2) OR EP across nodes (MoE, e.g. EP16 — lower overhead, higher throughput than TP8PP2's lower latency).
- Only go multi-node if model/KV genuinely needs it — otherwise scale replicas horizontally instead.

### 5.5 Disaggregation
- Split **prefill** (compute-bound engine) and **decode** (memory-bound engine) onto separate GPUs/nodes to avoid resource contention and independently optimize/scale each.
- Flow: prefill engine builds KV cache + first token → transfers KV cache over interconnect → decode engine generates rest.
- **Conditional disaggregation**: decode engine handles short/cached requests locally, only offloads to prefill engine when needed (better for real traffic).
- **When to use**: high traffic (100M-1B+ tokens/day), large model (100B+ params), prefill-heavy/long-input workloads (e.g., code editors). Otherwise not worth the complexity/hardware.
- **NVIDIA Dynamo** provides production disaggregation: prefill queue, conditional routing thresholds, NIXL-based KV transfer. Written as **xPyD** (e.g. 5P3D = 5 prefill + 3 decode engines). New bottlenecks: prefill queue depth, decode-engine KV cache exhaustion (mitigate via quantization/offloading).

---

## Ch 6 — Modalities

Two archetypes apply everywhere: **autoregressive** (LLM-like: VLMs, embeddings, ASR, TTS) vs **iterative denoising** (image/video). Most LLM techniques (quant, speculation, caching, parallelism, disaggregation) transfer to autoregressive modalities.

### 6.1 Vision Language Models (VLMs)
- = LLM + small **vision encoder** (converts images/video to image tokens). Encoder small in params but critical/fragile for runtime support (favor vLLM/SGLang).
- ~1000 visual tokens per high-res image → long input sequences & KV cache. All LLM techniques apply, plus **downsampling** (resolution/token tradeoff) for multi-image/video.
- **Video for VLMs**: 4 sec of video ≈ 100K tokens uncompressed → downsampling near-mandatory; still limited to short clips.
- **Omni-modal models**: accept/output multiple modalities — flexible but often less accurate than dedicated single-purpose models (e.g., VLM OCR < dedicated OCR).

### 6.2 Embedding Models
- Convert text/image into fixed-length vector for semantic search/RAG/recsys.
- Two traffic profiles: **high-throughput backfill** (bulk indexing) vs **low-latency lookup** (live search) — usually separate deployments.
- Architectures: **BERT-style** (encoder-only, <1B, fast/simple) vs **LLM-based** (≤8B, higher quality). **Matryoshka** embeddings allow dynamic dimensionality/quality tradeoff.
- Best perf via TensorRT-LLM adaptation; FP8 quantization safe with minimal quality loss (check via cosine similarity, target ≥99%).
- No prefix caching/disaggregation benefit (parallel token processing); scale via horizontal replicas + large batch sizes + robust queueing.

### 6.3 ASR (Automatic Speech Recognition) — e.g. Whisper
- Encoder (audio→features) + Decoder (features→text, autoregressive like LLM — dominates inference time). TensorRT-LLM is the main optimization tool.
- **Real-time (single-chunk)**: target ~200ms RTT (human reaction time); gains come from infra/streaming (WebSockets + VAD chunking), not runtime.
- **Long-file**: measured via **RTF** (Real-Time Factor; optimized systems reach ~1000x). Pipeline: VAD segmentation → parallel chunk transcription across GPUs → stitch by timestamp. Hallucination fixes: re-run at higher temp, or re-chunk smaller.
- **Diarization** (who spoke when): separate classic-ML pipeline (segmentation+embedding+clustering, e.g. pyannote) — not a transformer LM; ~2x slower than transcription alone.

### 6.4 TTS (Text-to-Speech)
- Modern TTS = fine-tuned LLMs (e.g., Orpheus TTS from Llama 3.2 3B) — LLM techniques apply (quantize to FP8, TensorRT-LLM, in-flight batching).
- Needs an **audio decoder** (tokens→waveform) — batch with short timeout (~15ms); no in-flight batching there.
- Metrics: **TTFB** (≈TTFT), **time to first sentence** (more user-relevant), **TPS** (only need ~80-100 for real-time; more is wasted). Optimize for concurrent real-time streams, not raw speed.
- Real-time streaming via WebSockets is the biggest infra lever. Not great for long inputs (quality degrades past ~30s).
- **Speech-to-speech** (audio in→audio out, e.g. gpt-realtime): emerging, no viable open models yet; most production voice = cascading ASR→LLM→TTS pipeline.

### 6.5 Image Generation Models
- Very different from LLMs: pipeline of small models, compute-bound (not bandwidth), 10-20x smaller than frontier LLMs, and quality/speed is a very direct dial.
- Optimization libraries: SGLang Diffusion, TensorRT, or raw PyTorch (most control). Key kernels: attention (FlashAttention 2/3/4), normalization fusion (RMSNorm), GEMM (safe to quantize to FP8/2x FLOPS).
- **"One weird trick"**: turn off classifier-free guidance for later denoising steps (image outline set early; skip 2nd forward pass late) — saves compute with minimal quality loss.

### 6.6 Video Generation Models
- Most demanding modality; run on Blackwell+ full nodes, batch size = 1 (all 8 GPUs work one video).
- Attention = 70-80% of compute time (3D latent space: width×height×time) — top optimization target.
- Caching (timestep-based or transformer-based, reusing hidden states/outputs) can give 30-40% speedup — but test carefully for quality loss.
- Quantization here targets **attention** (not weights, unlike LLMs) since compute — not memory — is the bottleneck; use microscaling (MXFP8) + selective quantization (skip early steps/first-last layers). E.g. SageAttention (8-bit attention kernel).
- **Context Parallelism (CP)**: replicate weights across all 8 GPUs, split attention/context via ring-attention-style passing — used instead of Tensor Parallelism for video.

---

## Ch 7 — Production

### Containerization
- Docker: **Container** (running instance) / **Image** (packaged artifact) / **Dockerfile** (build instructions) / **Registry** (Docker Hub etc). Layers: base image → additional layers → ephemeral container layer.
- Best practice: start from official inference-engine base images; pin dependency versions exactly for reproducibility.
- **NIMs** (NVIDIA Inference Microservices): prebuilt containers (multi-LLM or model-specific) — good starting point/reference, but build your own for max control.

### Autoscaling
- Goal: match resources to demand without wasting spend or missing SLAs. Runs on **Kubernetes** (control plane decides, worker plane executes).
- Scale on **traffic** (proactive) + **utilization** (lagging) signals combined.
- Config knobs: min/max replicas, autoscaling window, scale-down delay, concurrency target.
- **Batching types**: Static (wait for full batch) < Dynamic (full batch or timeout) < **Continuous/in-flight batching** (token-level, used by vLLM/SGLang/TensorRT-LLM) — best latency.
- **Cold starts**: sum of GPU procurement + image load + weight load + engine startup/compile. Mitigate via smaller images, quantized/faster-loading weights, cached compiled engines (must match exact GPU/CUDA/deps).
- **Routing vs Load balancing**: router = "where should this go" (per-request, e.g. KV-cache-aware, LoRA-aware); load balancer = "where could this go" (system-level). Use **queues** to absorb traffic bursts (priority queues for tiered users).
- **Scale to zero**: needs fast cold starts + robust queueing; good for dev/bursty/off-hours traffic, bad fit for latency-sensitive unscheduled traffic (use pay-per-token instead).
- **Independent component scaling**: multi-model pipelines should scale each step separately (different HW needs) but stay in the **same cluster** to avoid cross-cluster latency penalties.

### Multi-Cloud Capacity Management
- True multi-cloud = fungible GPU pool across providers (bin-packing), not siloed compute. Unlocks capacity, redundancy, latency (geo-proximity), compliance.
- Architecture: global **Control plane** (deployment/scaling decisions) + local **Workload planes** (serve traffic, report state).
- **GPU procurement**: Hyperscalers (AWS/GCP), Neoclouds (Coreweave/Nebius), Resellers (SF Compute). Modes: Reserved (discount, long-term), On-demand, Spot (cheap, pre-emptible).
- **Geo-aware load balancing**: ~5ms per time zone crossed — keep inference near users.
- **Reliability**: GPU failures are common at scale (Llama 3 training: ~1 failure/50K GPU-hours — a single 8-GPU node running a year exceeds that). **Active-active** (multiple regions live simultaneously) vs **Active-passive** (hot standby failover).
- **Security/Compliance**: protect user data, model weights, infra. Don't store data you don't need. SOC2/HIPAA compliance may require provider compliance + data residency (keep region-specific data in-region).

### Testing & Deployment
- Testing types: manual, load testing, **shadow traffic** (copy live traffic to test system).
- **Blue-green deployment**: full duplicate environment, cut over — expensive at GPU scale (2x hardware).
- **Canary deployment**: gradually shift % of traffic to new deployment, monitor, rollback if issues — preferred for inference due to lower GPU overhead (works with autoscaling).
- **Cost estimation**: dedicated inference cost = f(batch size, traffic pattern, sequence lengths) — much more complex than per-token API pricing; compare *total cost*, not per-token equivalent. Include engineering time in TCO.
- **Observability**: track volume, request/response sizes, response codes, latency (P50/90/99), replica count, utilization, queue depth. Integrate with existing tools (Grafana, Datadog, PagerDuty, Sentry).

### Client Code
- Client latency overhead matters: TLS handshake can eat ~10% of a 300ms SLA — reuse sessions (OpenAI SDK does this automatically).
- **Asynchronous inference**: "fire and forget" with webhook callback — for throughput-oriented, non-latency-sensitive batch jobs (longer timeouts, hours not minutes).
- **Streaming protocols**: WebSockets (unstructured/real-time, e.g. audio) vs gRPC (structured, schema-enforced, service-to-service, slightly slower due to validation).

---

## Appendix A — Glossary (highlights not covered above)
- **Active-active / Active-passive**: HA postures (live-live vs standby failover).
- **Baseline**: pre-optimization measurement to attribute gains/regressions.
- **Chunked Prefill**: splits long inputs to interleave with decode, avoids monopolizing resources.
- **Data sovereignty**: legal constraints on where data is processed/stored.
- *(Full A–Z glossary spans pages 209–230 of the book — refer to it directly for exact one-line definitions of any term you forget.)*

## Appendix B — Recommended Reading (categories only)
Architecture · Developer Tools · Frontier Open Models · GPU Infrastructure · Inference Optimization Research · Intelligence Evaluation — see book pp. 231–256 for the curated list of papers/tools per category.

---

## Quick Recall Cheat-Sheet

| Concept | One-liner |
|---|---|
| TTFT | Compute-bound, prefill, "how fast does it start responding" |
| TPS/ITL | Memory-bound, decode, "how fast tokens stream" |
| Compute-bound ops | Prefill, image/video gen |
| Memory-bound ops | Decode (low/med batch) |
| Best default parallelism | Tensor Parallelism (single node), Expert Parallelism (MoE/throughput) |
| Best speculative decoding | EAGLE (general), N-gram (code) |
| Best quantization format | FP8 / MXFP8 (safe), NVFP4 (aggressive) |
| When to disaggregate | Big model + big traffic + long/prefill-heavy input |
| When to use Dynamo | Large-scale foundation model serving, not small deployments |
| Best inference engine | vLLM (ease/breadth), SGLang (MoE/customization), TensorRT-LLM (max perf) |
