# Machine Learning Papers Directory

## 1. Foundation Models & Architectures

### Language Models

#### DeepSeek-V3 Technical Report
**Paper**: [Link](https://paperswithcode.com/paper/deepseek-v3-technical-report)  
**Summary**: DeepSeek-V3 is a Mixture-of-Experts (MoE) language model comprising 671 billion parameters, with 37 billion activated per token. It is pre-trained on 14.8 trillion diverse, high-quality tokens, followed by supervised fine-tuning and reinforcement learning to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models, despite requiring only 2.788 million H800 GPU hours for full training.  
**Tags**: #language-model #Mixture-of-Experts #AI

#### Large Concept Models
**Paper**: [Link](https://paperswithcode.com/paper/large-concept-models-language-modeling-in-a)  
**Summary**: This paper introduces the concept of Large Concept Models (LCMs), which operate on higher-level semantic representations beyond individual tokens. By treating entire sentences as single concepts, LCMs utilize the SONAR sentence embedding space, supporting up to 200 languages in both text and speech modalities.  
**Tags**: #language-modeling #sentence-embedding #multilingual #AI

#### Byte Latent Transformer
**Paper**: [Link](https://paperswithcode.com/paper/byte-latent-transformer-patches-scale-better)  
**Summary**: The Byte Latent Transformer (BLT) is a novel byte-level large language model (LLM) architecture that matches the performance of traditional tokenization-based LLMs while enhancing inference efficiency and robustness. BLT encodes raw bytes into dynamically sized patches, which serve as the primary computational units.  
**Tags**: #LLM #byte-level-modeling #transformer #AI

#### Hunyuan-Large
**Paper**: [Link](https://paperswithcode.com/paper/hunyuan-large-an-open-source-moe-model-with)  
**Summary**: This paper introduces Hunyuan-Large, the largest open-source Transformer-based mixture of experts model, featuring a total of 389 billion parameters with 52 billion activated parameters and the ability to handle up to 256K tokens.  
**Tags**: #MoE #transformer #open-source #AI

#### LongWriter
**Paper**: [Link](https://paperswithcode.com/paper/longwriter-unleashing-10000-word-generation)  
**Summary**: This paper addresses the limitation of current long-context large language models (LLMs) that struggle to generate outputs exceeding 2,000 words. The authors introduce AgentWrite, an agent-based pipeline that decomposes ultra-long generation tasks into subtasks.  
**Tags**: #long-context #LLM #text-generation #AI

#### TableGPT2
**Paper**: [Link](https://paperswithcode.com/paper/tablegpt2-a-large-multimodal-model-with)  
**Summary**: TableGPT2 is a large-scale multimodal model specifically designed to integrate tabular data into AI applications. Trained on over 593,800 tables and 2.36 million high-quality query-table-output tuples.  
**Tags**: #multimodal-model #tabular-data #AI

#### Qwen2.5-Coder Technical Report
**Paper**: [Link](https://paperswithcode.com/paper/qwen2-5-coder-technical-report)  
**Summary**: The Qwen2.5-Coder series features six models ranging from 0.5B to 32B parameters. Built upon the Qwen2.5 architecture, these code-specific models are pretrained on a vast corpus exceeding 5.5 trillion tokens.  
**Tags**: #code-generation #LLM #AI #Qwen2.5-Coder

### Memory & Attention Mechanisms

#### An Evolved Universal Transformer Memory
**Paper**: [Link](https://paperswithcode.com/paper/an-evolved-universal-transformer-memory)  
**Summary**: This paper introduces Neural Attention Memory Models (NAMMs), a learned network for memory management that enhances both the performance and efficiency of transformers.  
**Tags**: #transformers #memory-management #self-attention #AI

#### Star Attention
**Paper**: [Link](https://paperswithcode.com/paper/star-attention-efficient-llm-inference-over)  
**Summary**: Star Attention introduces a two-phase block-sparse approximation to enhance the efficiency of Transformer-based Large Language Models (LLMs) during inference on long sequences.  
**Tags**: #LLM #attention-mechanism #inference-optimization #block-sparse-attention

#### Breaking the Low-Rank Dilemma of Linear Attention
**Paper**: [Link](https://paperswithcode.com/paper/breaking-the-low-rank-dilemma-of-linear)  
**Summary**: This paper addresses the computational inefficiencies of the Softmax attention mechanism in Transformer models, introducing Rank-Augmented Linear Attention (RALA).  
**Tags**: #linear-attention #transformer #vision-models #RALA #RAVLT

#### LoLCATs
**Paper**: [Link](https://arxiv.org/abs/2410.10254)  
**Summary**: This paper introduces LoLCATs (Low-rank Linear Conversion via Attention Transfer), a method designed to enhance the efficiency of large language models (LLMs) by replacing their quadratic attention mechanisms with subquadratic linear attentions.  
**Tags**: #linear-attention #LLM #LoRA #model-efficiency

#### Memory Layers at Scale
**Paper**: [Link](https://paperswithcode.com/paper/memory-layers-at-scale)  
**Summary**: This paper introduces an enhanced memory layer for language models, demonstrating significant performance improvements, particularly in factual tasks.  
**Tags**: #memory-augmentation #language-models #scaling-laws #AI

### Multimodal Models

#### Aria
**Paper**: [Link](https://paperswithcode.com/paper/aria-an-open-multimodal-native-mixture-of)  
**Summary**: Aria is an open multimodal-native mixture of experts model designed to handle diverse data types, enhancing AI's multimodal processing capabilities.  
**Tags**: #architecture #multimodal #AI

#### R-CoT
**Paper**: [Link](https://paperswithcode.com/paper/r-cot-reverse-chain-of-thought-problem)  
**Summary**: This paper introduces R-CoT, a two-stage Reverse Chain-of-Thought geometry problem generation pipeline designed to enhance the geometric reasoning capabilities of Large Multimodal Models (LMMs).  
**Tags**: #R-CoT #geometric-reasoning #multimodal-models #AI

#### RESOLVE
**Paper**: [Link](https://paperswithcode.com/paper/resolve-relational-reasoning-with-symbolic)  
**Summary**: RESOLVE introduces a neuro-vector symbolic architecture that integrates object-level features with relational representations in high-dimensional spaces.  
**Tags**: #relational-reasoning #vector-symbolic-processing #neuro-symbolic-AI #AI

## 2. AI Agents & Automation

### Agentic Frameworks

#### Agent S
**Paper**: [Link](https://paperswithcode.com/paper/agent-s-an-open-agentic-framework-that-uses)  
**Summary**: Agent S introduces an open agentic framework that enables AI systems to utilize computers similarly to humans, enhancing task management and execution capabilities.  
**Tags**: #agent #task-management #AI

#### TaskGen
**Paper**: [Link](https://paperswithcode.com/paper/taskgen-a-task-based-memory-infused-agentic)  
**Summary**: TaskGen presents a task-based, memory-infused agentic framework designed to improve AI's ability to manage and execute complex tasks efficiently.  
**Tags**: #agent #task-management #AI

#### FinRobot
**Paper**: [Link](https://paperswithcode.com/paper/finrobot-ai-agent-for-equity-research-and)  
**Summary**: FinRobot introduces an AI agent framework tailored for equity research, integrating both quantitative and qualitative analyses to emulate human analyst reasoning. It comprises three specialized agents:
- Data-CoT Agent: Aggregates diverse data sources for robust financial integration
- Concept-CoT Agent: Mimics analyst reasoning to generate actionable insights
- Thesis-CoT Agent: Synthesizes insights into coherent investment theses and reports  
**Tags**: #AI-agent #equity-research #valuation #LLM

#### Kinetix
**Paper**: [Link](https://paperswithcode.com/paper/kinetix-investigating-the-training-of-general)  
**Summary**: Kinetix introduces an open-ended framework for training general reinforcement learning (RL) agents across a vast array of 2D physics-based tasks. By procedurally generating tens of millions of environments—ranging from robotic locomotion and grasping to video games—the framework enables agents to develop robust physical reasoning capabilities.  
**Tags**: #reinforcement-learning #physics-based-tasks #generalization #AI

### GUI & Interface Automation

#### ShowUI
**Paper**: [Link](https://paperswithcode.com/paper/showui-one-vision-language-action-model-for)  
**Summary**: ShowUI is a vision-language-action model designed to enhance human productivity by enabling AI agents to interact with graphical user interfaces (GUIs) in a manner similar to human perception. It incorporates UI-Guided Visual Token Selection and Interleaved Vision-Language-Action Streaming to efficiently manage visual-action history in navigation.  
**Tags**: #GUI-assistants #vision-language-action #AI #productivity

#### The Dawn of GUI Agent
**Paper**: [Link](https://paperswithcode.com/paper/the-dawn-of-gui-agent-a-preliminary-case)  
**Summary**: This study explores Claude 3.5 Computer Use, an AI model designed to function as a graphical user interface (GUI) agent. The authors curated a set of tasks across various domains and software to assess its capabilities, demonstrating effective translation of language instructions into desktop actions.  
**Tags**: #GUI-agent #AI #automation #Claude3.5

#### AutoTrain
**Paper**: [Link](https://paperswithcode.com/paper/autotrain-no-code-training-for-state-of-the)  
**Summary**: AutoTrain by Huggingface introduces a no-code solution for training machine learning models, simplifying model customization and deployment.  
**Tags**: #UI #GUI #AI

### Robotics

#### OpenEMMA
**Paper**: [Link](https://paperswithcode.com/paper/openemma-open-source-multimodal-model-for-end)  
**Summary**: OpenEMMA is an open-source end-to-end framework that leverages Multimodal Large Language Models (MLLMs) to enhance autonomous driving systems. By incorporating Chain-of-Thought reasoning processes, OpenEMMA improves performance across various challenging driving scenarios.  
**Tags**: #autonomous-driving #MLLM #open-source #AI

#### RoboMatrix
**Paper**: [Link](https://paperswithcode.com/paper/robomatrix-a-skill-centric-hierarchical)  
**Summary**: RoboMatrix introduces a skill-centric and hierarchical framework aimed at enhancing robot task planning and execution in open-world environments. By extracting common meta-skills from various complex tasks, the framework enables robots to dynamically combine these learned skills to address novel tasks through three interconnected layers: Scheduling Layer, Skill Layer, and Hardware Layer.  
**Tags**: #robotics #task-planning #hierarchical-framework #meta-skills

## 3. Knowledge Enhancement & Retrieval

### Retrieval Systems

#### Auto-RAG
**Paper**: [Link](https://paperswithcode.com/paper/auto-rag-autonomous-retrieval-augmented)  
**Summary**: Auto-RAG introduces an autonomous iterative retrieval model that leverages the decision-making capabilities of Large Language Models (LLMs) to enhance Retrieval-Augmented Generation (RAG). Unlike traditional methods that rely on few-shot prompting or manually constructed rules, Auto-RAG enables LLMs to engage in multi-turn dialogues with retrievers, systematically planning retrievals and refining queries to acquire valuable knowledge. The iterative retrieval process is expressed in natural language, enhancing interpretability and providing users with a more intuitive experience.  
**Tags**: #Retrieval-Augmented-Generation #LLM #iterative-retrieval #AI

#### MemoRAG
**Paper**: [Link](https://paperswithcode.com/paper/memorag-moving-towards-next-gen-rag-via)  
**Summary**: MemoRAG explores advanced techniques in retrieval-augmented generation for more effective information synthesis in AI systems.  
**Tags**: #RAG #retrieval #AI

#### KAG
**Paper**: [Link](https://paperswithcode.com/paper/2409-13731)  
**Summary**: This paper explores Knowledge Augmented Generation (KAG) as a method to enhance large language models (LLMs) for specific professional domains, focusing on the integration of domain-specific knowledge to improve accuracy and relevance in outputs.  
**Tags**: #KAG #knowledge-augmentation #professional-domains #AI

#### Cache-Augmented Generation (CAG)
**Paper**: [Link](https://paperswithcode.com/paper/don-t-do-rag-when-cache-augmented-generation)  
**Summary**: This paper introduces Cache-Augmented Generation (CAG) as an alternative to Retrieval-Augmented Generation (RAG) for enhancing language models. CAG involves preloading all relevant resources into a language model's extended context, eliminating the need for real-time retrieval during inference. This approach reduces retrieval latency and errors while maintaining context relevance, particularly benefiting tasks with a limited and manageable knowledge base.  
**Tags**: #cache-augmented-generation #language-models #memory-augmentation #AI

### Document Processing

#### Docling Technical Report
**Paper**: [Link](https://paperswithcode.com/paper/docling-technical-report)  
**Summary**: This technical report introduces Docling, an open-source, MIT-licensed package designed for efficient PDF document conversion. Leveraging advanced AI models like DocLayNet for layout analysis and TableFormer for table structure recognition, Docling operates effectively on standard hardware with minimal resource requirements.  
**Tags**: #document-conversion #PDF #open-source #AI

## 4. Security & Safety

### Detection & Monitoring

#### Human Intelligence vs AI Detection
**Paper**: [Link](https://www.nature.com/articles/s41598-024-76218-y?fromPaywallRec=false)  
**Summary**: This study explores individual differences in discerning human-generated texts from AI-generated ones, highlighting the role of human intelligence in monitoring AI outputs.  
**Tags**: #detection #monitoring #human-AI

#### LLMSCAN
**Paper**: [Link](https://paperswithcode.com/paper/llmscan-causal-scan-for-llm-misbehavior)  
**Summary**: LLMSCAN introduces a causal scanning method to detect misbehavior in large language models, enhancing AI system reliability.  
**Tags**: #detection #monitoring #AI

#### From Imitation to Introspection
**Paper**: [Link](https://paperswithcode.com/paper/from-imitation-to-introspection-probing-self)  
**Summary**: This paper explores the emergence of self-consciousness in language models, transitioning from mere imitation to introspective capabilities. It examines the extent to which these models can exhibit self-referential behaviors, delving into the philosophical implications of AI's evolving cognitive functions.  
**Tags**: #self-consciousness #AI-philosophy #introspection #detection

#### PARIS
**Paper**: [Link](https://arxiv.org/abs/2411.01273)  
**Summary**: This paper introduces PARIS, a system designed for real-time detection of malicious behaviors on Windows platforms. Utilizing Event Tracing for Windows (ETW), PARIS adaptively monitors and collects data related to potentially malicious API calls and call stacks, significantly reducing data collection overhead by over 98.8% compared to raw ETW traces while maintaining detection accuracy.  
**Tags**: #malware-detection #real-time-monitoring #ETW #cybersecurity

### Security Frameworks

#### Best-of-N Jailbreaking
**Paper**: [Link](https://paperswithcode.com/paper/best-of-n-jailbreaking)  
**Summary**: Best-of-N (BoN) Jailbreaking is a black-box algorithm designed to bypass safety measures in advanced AI systems across various modalities. It operates by generating multiple variations of a prompt through augmentations until a harmful response is elicited, achieving high attack success rates (ASRs) on closed-source language models. The algorithm extends to vision language models (VLMs) and audio language models (ALMs).  
**Tags**: #jailbreaking #AI-safety #black-box-attack #multi-modal-AI

#### Granite Guardian
**Paper**: [Link](https://paperswithcode.com/paper/granite-guardian)  
**Summary**: Granite Guardian introduces a suite of models designed to detect risks in prompts and responses, facilitating the safe and responsible use of large language models (LLMs). These models cover multiple risk dimensions, including social bias, profanity, violence, sexual content, unethical behavior, jailbreaking, and hallucination-related risks.  
**Tags**: #risk-detection #LLM #AI-safety #open-source

#### garak
**Paper**: [Link](https://paperswithcode.com/paper/garak-a-framework-for-security-probing-large)  
**Summary**: garak (Generative AI Red-teaming and Assessment Kit) is an open-source framework designed to identify vulnerabilities in large language models (LLMs) and dialogue systems. It systematically probes LLMs to uncover potential security weaknesses, providing detailed reports that inform discussions on model alignment and deployment policies.  
**Tags**: #LLM #security #vulnerability-assessment #AI

#### AdvSecureNet
**Paper**: [Link](https://paperswithcode.com/paper/advsecurenet-a-python-toolkit-for-adversarial)  
**Summary**: AdvSecureNet provides a Python toolkit for adversarial attack simulation in PyTorch, designed to test and improve AI robustness.  
**Tags**: #security #pytorch #AI

#### Neural Fingerprints
**Paper**: [Link](https://paperswithcode.com/paper/neural-fingerprints-for-adversarial-attack)  
**Summary**: This paper introduces a method to detect adversarial attacks on AI models by leveraging neural fingerprints. The approach involves creating unique identifiers for neural networks, enabling the identification of unauthorized modifications or attacks.  
**Tags**: #adversarial-attack #detection #neural-fingerprints #AI

#### Trustworthy Federated Learning
**Paper**: [Link](https://arxiv.org/abs/2411.01583)  
**Summary**: This comprehensive survey examines the security and privacy challenges inherent in Federated Learning (FL). It highlights vulnerabilities in communication channels and potential cyber threats within decentralized networks, while discussing various defensive strategies and proposing future research directions.  
**Tags**: #federated-learning #privacy #security #decentralized-networks

### Poisoning & Attacks

#### NoiseAttack
**Paper**: [Link](https://paperswithcode.com/paper/noiseattack-an-evasive-sample-specific-multi)  
**Summary**: NoiseAttack discusses an evasive, sample-specific multi-domain attack strategy, highlighting vulnerabilities in AI systems.  
**Tags**: #poisoning #security #AI

#### CodeBreaker
**Paper**: [Link](https://paperswithcode.com/paper/an-llm-assisted-easy-to-trigger-backdoor)  
**Summary**: CodeBreaker presents an LLM-assisted, easy-to-trigger backdoor attack, emphasizing the need for robust security measures in AI.  
**Tags**: #poisoning #security #AI

### Cybersecurity

#### IntellBot
**Paper**: [Link](https://arxiv.org/abs/2411.05442)  
**Summary**: This paper introduces IntellBot, an advanced cybersecurity chatbot leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques. IntellBot aggregates information from diverse sources to create a comprehensive knowledge base encompassing known vulnerabilities, recent cyberattacks, and emerging threats.  
**Tags**: #cybersecurity #chatbot #LLM #RAG #threat-intelligence

#### Conti Ransomware Group Analysis
**Paper**: [Link](https://arxiv.org/abs/2411.02548)  
**Summary**: This study explores the cultural aspects of cybercriminal organizations, focusing on the Conti ransomware group. By analyzing leaked internal communications, the authors apply Hofstede's and Meyer's cultural dimension frameworks to understand the group's operational behaviors.  
**Tags**: #cybersecurity #cultural-analysis #threat-intelligence #ransomware

## 5. Media Generation & Processing

### Speech & Audio

#### TangoFlux
**Paper**: [Link](https://paperswithcode.com/paper/tangoflux-super-fast-and-faithful-text-to)  
**Summary**: TangoFlux is an efficient text-to-audio (TTA) generative model with 515 million parameters, capable of producing up to 30 seconds of 44.1kHz audio in just 3.7 seconds on a single A40 GPU. It introduces CLAP-Ranked Preference Optimization (CRPO), a novel framework that iteratively generates and optimizes preference data to enhance TTA alignment.  
**Tags**: #text-to-audio #audio-generation #open-source #AI

#### F5-TTS
**Paper**: [Link](https://paperswithcode.com/paper/f5-tts-a-fairytaler-that-fakes-fluent-and)  
**Summary**: F5-TTS introduces a fully non-autoregressive text-to-speech (TTS) system utilizing flow matching with a Diffusion Transformer (DiT). By padding text inputs to match the length of speech inputs and performing denoising for speech generation, it eliminates the need for complex components like duration models, text encoders, and phoneme alignment.  
**Tags**: #text-to-speech #flow-matching #Diffusion-Transformer #AI

#### Qwen2-Audio
**Summary**: Qwen2-Audio is an advanced speech model designed to improve audio processing capabilities in AI systems.  
**Tags**: #speech #audio #AI-models

#### LLaMA-Omni
**Paper**: [Link](https://paperswithcode.com/paper/llama-omni-seamless-speech-interaction-with)  
**Summary**: LLaMA-Omni offers seamless speech interaction with large language models, enhancing real-time communication capabilities.  
**Tags**: #speech #audio #AI-models

#### Moonshine
**Paper**: [Link](https://paperswithcode.com/paper/moonshine-speech-recognition-for-live)  
**Summary**: This paper introduces Moonshine, a family of speech recognition models optimized for live transcription and voice command processing. Moonshine is based on an encoder-decoder transformer architecture and employs Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings.  
**Tags**: #speech-recognition #live-transcription #voice-commands #AI

### Video Generation

#### SynCamMaster
**Paper**: [Link](https://paperswithcode.com/paper/syncammaster-synchronizing-multi-camera-video)  
**Summary**: SynCamMaster introduces a plug-and-play module that enhances pre-trained text-to-video models for generating synchronized multi-camera videos from diverse viewpoints. It incorporates a multi-view synchronization module to maintain appearance and geometry consistency across different camera angles.  
**Tags**: #video-generation #multi-camera #synchronization #AI

#### HunyuanVideo
**Paper**: [Link](https://paperswithcode.com/paper/hunyuanvideo-a-systematic-framework-for-large)  
**Summary**: HunyuanVideo is an open-source video foundation model that achieves performance comparable to, or surpassing, leading closed-source models. The framework integrates key components such as data curation, advanced architectural design, progressive model scaling and training, and an efficient infrastructure for large-scale model training and inference.  
**Tags**: #video-generation #open-source #AI #large-scale-models

#### Open-Sora Plan
**Paper**: [Link](https://paperswithcode.com/paper/open-sora-plan-open-source-large-video)  
**Summary**: The Open-Sora Plan introduces an open-source initiative aimed at developing a large-scale model capable of generating high-resolution, extended-duration videos based on diverse user inputs. The project encompasses several components including Wavelet-Flow Variational Autoencoder and Joint Image-Video Skiparse Denoiser.  
**Tags**: #video-generation #open-source #AI #Wavelet-Flow-VAE #Skiparse-Denoiser

### Image Animation

#### StableAnimator
**Paper**: [Link](https://paperswithcode.com/paper/stableanimator-high-quality-identity)  
**Summary**: StableAnimator is an end-to-end video diffusion framework designed to synthesize high-quality human image animations while preserving identity (ID) consistency. It utilizes off-the-shelf extractors to compute image and face embeddings, refining them through a global content-aware Face Encoder.  
**Tags**: #image-animation #identity-preservation #video-diffusion #AI

#### Hallo2
**Paper**: [Link](https://paperswithcode.com/paper/hallo2-long-duration-and-high-resolution)  
**Summary**: This paper introduces Hallo2, an advanced method for generating long-duration, high-resolution portrait image animations driven by audio inputs. The approach addresses challenges like appearance drift and temporal artifacts by implementing augmentation strategies within the image space of conditional motion frames.  
**Tags**: #image-animation #audio-driven #high-resolution #long-duration

### Computer Vision

#### Gaze-LLE
**Paper**: [Link](https://paperswithcode.com/paper/gaze-lle-gaze-target-estimation-via-large)  
**Summary**: Gaze-LLE addresses the challenge of gaze target estimation by leveraging features from a frozen DINOv2 encoder within a novel transformer framework. By extracting a unified feature representation for the scene and applying a person-specific positional prompt, the model streamlines gaze prediction without relying on complex, hand-crafted pipelines.  
**Tags**: #gaze-estimation #transformer #DINOv2 #AI

#### Thinking in Space
**Paper**: [Link](https://paperswithcode.com/paper/thinking-in-space-how-multimodal-large)  
**Summary**: This paper investigates whether Multimodal Large Language Models (MLLMs), trained on extensive video datasets, can develop visual-spatial intelligence akin to humans. The authors introduce VSI-Bench, a novel benchmark comprising over 5,000 question-answer pairs designed to assess video-based visual-spatial intelligence.  
**Tags**: #MLLM #visual-spatial-intelligence #VSI-Bench #cognitive-maps

## 6. Quality & Fairness

### Hallucination Reduction

#### Prereq-Tune
**Paper**: [Link](https://paperswithcode.com/paper/fictitious-synthetic-data-can-improve-llm)  
**Summary**: Prereq-Tune utilizes synthetic data to reduce hallucination in large language models, enhancing factual accuracy.  
**Tags**: #hallucination #factuality #AI

#### DecoPrompt
**Paper**: [Link](https://paperswithcode.com/paper/decoprompt-decoding-prompts-reduces)  
**Summary**: DecoPrompt addresses the issue of hallucinations in large language models (LLMs) when confronted with false premises. The method leverages LLMs to "decode" misleading prompts without generating hallucinated outputs, effectively reducing inaccuracies. Experiments demonstrate DecoPrompt's efficacy across different LLMs and its cross-model transferability, making it applicable to various scenarios, including those involving large models or inaccessible model logits.  
**Tags**: #hallucination-reduction #LLM #false-premises #AI

### Fairness & Bias

#### Fair Summarization
**Paper**: [Link](https://paperswithcode.com/paper/fair-summarization-bridging-quality-and)  
**Summary**: This paper addresses the challenge of ensuring fairness in multi-document summarization, particularly when dealing with user-generated content from diverse social groups. The authors introduce two novel methods for fair extractive summarization: FairExtract, a clustering-based approach, and FairGPT, which leverages GPT-3.5-turbo with fairness constraints. Evaluations on the DivSumm dataset demonstrate that both methods achieve superior fairness while maintaining competitive summarization quality.  
**Tags**: #fairness #summarization #extractive-summarization #GPT-3.5

#### Mitigating Bias in Queer Representation
**Paper**: [Link](https://paperswithcode.com/paper/mitigating-bias-in-queer-representation)  
**Summary**: This paper addresses biases in pronoun usage by large language models (LLMs), particularly the inappropriate application of gendered pronouns when inclusive language is necessary. The authors introduce a collaborative agent framework designed to detect and correct such biases, enhancing inclusivity in AI-generated content. Evaluations demonstrate a significant improvement in inclusive pronoun classification, with a 32.6 percentage point increase over GPT-4o in correctly identifying inappropriate gendered pronouns.  
**Tags**: #bias-mitigation #queer-representation #LLM #inclusive-language

## 7. Training & Infrastructure

### Training Frameworks

#### TorchTitan
**Paper**: [Link](https://paperswithcode.com/paper/torchtitan-one-stop-pytorch-native-solution)  
**Summary**: This paper introduces TorchTitan, an open-source, PyTorch-native distributed training system designed to streamline the training of large language models (LLMs). TorchTitan unifies state-of-the-art techniques, enabling modular 3D parallelism with elastic scaling, and provides comprehensive tools for logging, checkpointing, and debugging. It incorporates hardware-software co-designed solutions to enhance efficiency.  
**Tags**: #distributed-training #PyTorch #large-language-models #AI

#### LLaMA-Factory
**Summary**: LLaMA-Factory is an open-source infrastructure project aimed at streamlining large language model deployment.  
**Tags**: #open-source #infrastructure #AI

#### SWIFT
**Paper**: [Link](https://paperswithcode.com/paper/swift-a-scalable-lightweight-infrastructure)  
**Summary**: SWIFT provides scalable, lightweight infrastructure for efficient LLM deployment, designed to minimize resource consumption.  
**Tags**: #open-source #infrastructure #AI

#### MPLSandbox
**Paper**: [Link](https://paperswithcode.com/paper/multi-programming-language-sandbox-for-llms)  
**Summary**: This paper introduces MPLSandbox, an out-of-the-box framework designed to provide unified and comprehensive feedback from compiler and analysis tools for Large Language Models (LLMs). MPLSandbox automatically identifies the programming language of the code, compiling and executing it within an isolated sub-sandbox to ensure safety and stability. It integrates both traditional and LLM-based code analysis tools, offering a thorough examination of generated code.  
**Tags**: #LLM #code-analysis #sandbox #programming-languages

#### Initialization Using Update Approximation
**Paper**: [Link](https://paperswithcode.com/paper/initialization-using-update-approximation-is)  
**Summary**: This paper introduces a novel initialization method that approximates full fine-tuning within low-rank subspaces for training Transformer models. By employing a carefully designed initialization strategy, the approach achieves optimal scaling for high-rank gradient updates without the need for extensive hyperparameter tuning.  
**Tags**: #transformers #initialization #low-rank-fine-tuning #AI

### Benchmarks & Testing

#### OSWorld
**Summary**: OSWorld provides a comprehensive benchmarking environment for evaluating AI performance across various operating systems.  
**Tags**: #benchmark #testing #OS

#### AndroidWorld
**Summary**: AndroidWorld offers a specialized platform for testing and benchmarking AI applications within the Android ecosystem.  
**Tags**: #benchmark #testing #OS

#### WindowsAgentArena
**Summary**: WindowsAgentArena serves as a testing ground for assessing AI agents' performance in Windows operating environments.  
**Tags**: #benchmark #testing #OS

## 8. Specialized Applications

### Physics & Scientific Computing

#### The Well
**Paper**: [Link](https://paperswithcode.com/paper/the-well-a-large-scale-collection-of-diverse)  
**Summary**: The Well is a comprehensive collection of datasets comprising numerical simulations across various spatiotemporal physical systems. It offers 15TB of data spanning 16 datasets, covering domains such as biological systems, fluid dynamics, acoustic scattering, and magneto-hydrodynamic simulations of extra-galactic fluids and supernova explosions. This resource aims to facilitate the development and evaluation of machine learning-based surrogate models in physics.  
**Tags**: #physics-simulations #machine-learning #datasets #open-source

#### Neural Operators Library
**Paper**: [Link](https://paperswithcode.com/paper/a-library-for-learning-neural-operators)  
**Summary**: This paper introduces NeuralOperator, an open-source Python library designed to facilitate operator learning. Unlike traditional neural networks that map between finite-dimensional Euclidean spaces, neural operators are capable of learning mappings between function spaces, enabling them to handle inputs and outputs at various discretizations.  
**Tags**: #operator-learning #neural-operators #open-source #Python-library

#### Reinforcement Learning for Quantum Tiq-Taq-Toe
**Paper**: [Link](https://paperswithcode.com/paper/reinforcement-learning-for-quantum-tiq-taq)  
**Summary**: This paper explores the application of reinforcement learning (RL) to Quantum Tiq-Taq-Toe, a quantum variant of the classic Tic-Tac-Toe game. The study highlights the challenges in representing quantum games classically due to partial observability and exponential state complexity. The authors propose RL methods to navigate these complexities, aiming to serve as an accessible testbed for integrating quantum computing and machine learning.  
**Tags**: #reinforcement-learning #quantum-computing #game-theory #AI

### Biology & Artificial Life

#### Automating the Search for Artificial Life
**Paper**: [Link](https://paperswithcode.com/paper/automating-the-search-for-artificial-life)  
**Summary**: This paper introduces the Automated Search for Artificial Life (ASAL), a novel approach that leverages vision-language foundation models (FMs) to explore and discover lifelike simulations across various Artificial Life (ALife) substrates, including Boids, Particle Life, Game of Life, Lenia, and Neural Cellular Automata. ASAL automates the identification of simulations that produce target phenomena, generate temporally open-ended novelty, and illuminate a diverse range of interesting simulations. This methodology has led to the discovery of previously unseen Lenia and Boids lifeforms, as well as cellular automata exhibiting open-ended behavior akin to Conway's Game of Life.  
**Tags**: #ArtificialLife #FoundationModels #SimulationDiscovery #AI

## 9. Educational Resources

### Computing Fundamentals

#### Quantum Computing: Hype vs. Reality
**Video**: [Link](https://www.youtube.com/watch?v=-1PsQIciMEc)  
**Summary**: This video critically examines the current state of quantum computing, distinguishing between the technology's realistic capabilities and the surrounding hype. It provides insights into the progress made in the field, the challenges that remain, and the practical implications of quantum computing advancements.  
**Tags**: #quantum-computing #technology #analysis #reality-check

#### Quantum Computers Explained
**Video**: [Link](https://www.youtube.com/watch?v=JhHMJCUmq28)  
**Summary**: This educational video provides an accessible overview of quantum computing, exploring its fundamental principles and potential to revolutionize technology. It delves into the limitations of classical computers and illustrates how quantum mechanics can overcome these barriers, enabling unprecedented computational capabilities.  
**Tags**: #quantum-computing #technology #education #computing

#### Every attention head explained
**Source**: [YouTube](https://www.youtube.com/watch?v=qR56cyMdDXg)  
**Summary**: This educational video provides an in-depth explanation of attention heads within transformer models, a fundamental component in modern machine learning architectures. It covers the mechanics of attention mechanisms, their role in processing input data, and how they contribute to the performance of models in tasks like natural language processing and computer vision.  
**Tags**: #attention-heads #transformer-models #machine-learning #AI

#### How CUDA Programming Works
**Source**: [YouTube](https://www.youtube.com/watch?v=n6M8R8-PlnE&t=75s)  
**Summary**: This YouTube video provides an overview of CUDA programming, explaining its functionality and applications in parallel computing.  
**Tags**: #CUDA #computing

### Environmental Impact

#### The carbon emissions of writing and illustrating
**Paper**: [Link](https://www.nature.com/articles/s41598-024-54271-x.pdf)  
**Summary**: This study compares the carbon emissions associated with writing and illustrating tasks performed by AI systems versus humans. The findings indicate that AI systems produce significantly lower carbon emissions for these creative tasks.  
**Tags**: #carbon-emissions #AI-efficiency #environment #sustainability

