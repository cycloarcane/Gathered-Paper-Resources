
# Gathered Paper Resources

Welcome to the **Gathered Paper Resources** repository! This repository serves as an aggregation of valuable resources, references, and insights from various academic papers that I have reviewed related to Artificial Intelligence (AI), Large Language Models (LLMs), and related technologies.


## 1. Agentic Frameworks & Task Management
- **Agent S - Uses computers like a human** [paper](https://paperswithcode.com/paper/agent-s-an-open-agentic-framework-that-uses)  
    **Tags**: #agent #task-management #AI  
    **Summary**: Agent S introduces an open agentic framework that enables AI systems to utilize computers similarly to humans, enhancing task management and execution capabilities.

- **TaskGen** [paper](https://paperswithcode.com/paper/taskgen-a-task-based-memory-infused-agentic)  
    **Tags**: #agent #task-management #AI  
    **Summary**: TaskGen presents a task-based, memory-infused agentic framework designed to improve AI's ability to manage and execute complex tasks efficiently.

---

## 2. Benchmarks & Testing
- **OSWorld**  
    **Tags**: #benchmark #testing #OS  
    **Summary**: OSWorld provides a comprehensive benchmarking environment for evaluating AI performance across various operating systems.

- **AndroidWorld**  
    **Tags**: #benchmark #testing #OS  
    **Summary**: AndroidWorld offers a specialized platform for testing and benchmarking AI applications within the Android ecosystem.

- **WindowsAgentArena**  
    **Tags**: #benchmark #testing #OS  
    **Summary**: WindowsAgentArena serves as a testing ground for assessing AI agents' performance in Windows operating environments.

---

## 3. Detection & Monitoring
- **Human intelligence can safeguard against artificial intelligence** [paper](https://www.nature.com/articles/s41598-024-76218-y?fromPaywallRec=false)  
    **Tags**: #detection #monitoring #human-AI  
    **Summary**: This study explores individual differences in discerning human-generated texts from AI-generated ones, highlighting the role of human intelligence in monitoring AI outputs.

- **LLMSCAN** [paper](https://paperswithcode.com/paper/llmscan-causal-scan-for-llm-misbehavior)  
    **Tags**: #detection #monitoring #AI  
    **Summary**: LLMSCAN introduces a causal scanning method to detect misbehavior in large language models, enhancing AI system reliability.

- **From Imitation to Introspection: Probing Self-Consciousness in Language Models** [paper](https://paperswithcode.com/paper/from-imitation-to-introspection-probing-self)
    **Tags**: #self-consciousness #AI-philosophy #introspection #detection
    **Summary**: This paper explores the emergence of self-consciousness in language models, transitioning from mere imitation to introspective capabilities. It examines the extent to which these models can exhibit self-referential behaviors, delving into the philosophical implications of AI's evolving cognitive functions.
---

## 4. AI Architectures & Models
- **Kolmogorov-Arnold Transformers** [paper](https://paperswithcode.com/paper/kolmogorov-arnold-transformer)  
    **Tags**: #architecture #transformer #AI  
    **Summary**: This paper presents the Kolmogorov-Arnold Transformer, a novel architecture aimed at improving the efficiency and performance of transformer models.

- **Aria** [paper](https://paperswithcode.com/paper/aria-an-open-multimodal-native-mixture-of)  
    **Tags**: #architecture #multimodal #AI  
    **Summary**: Aria is an open multimodal-native mixture of experts model designed to handle diverse data types, enhancing AI's multimodal processing capabilities.

- **Hunyuan-Large**: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent [paper](https://paperswithcode.com/paper/hunyuan-large-an-open-source-moe-model-with)
    **Tags**: #MoE #transformer #open-source #AI
    **Summary**: This paper introduces Hunyuan-Large, the largest open-source Transformer-based mixture of experts model, featuring a total of 389 billion parameters with 52 billion activated parameters and the ability to handle up to 256K tokens. Hunyuan-Large demonstrates superior performance in various benchmarks, including language understanding and generation, logical reasoning, and coding tasks, surpassing smaller models while maintaining competitive performance against much larger counterparts like LLama3.1-405B. The model benefits from large-scale synthetic data, a mixed expert routing strategy, key-value cache compression, and expert-specific learning rates, with the authors also discussing scaling laws and learning rate schedules for mixture of experts models.


- **LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs** [paper](https://paperswithcode.com/paper/longwriter-unleashing-10000-word-generation)  
    **Tags**: #long-context #LLM #text-generation #AI
    **Summary**: This paper addresses the limitation of current long-context large language models (LLMs) that, despite processing inputs up to 100,000 tokens, struggle to generate outputs exceeding 2,000 words. The authors identify that this constraint stems from the scarcity of long-output examples in existing supervised fine-tuning (SFT) datasets. To overcome this, they introduce AgentWrite, an agent-based pipeline that decomposes ultra-long generation tasks into subtasks, enabling LLMs to produce coherent outputs exceeding 20,000 words. Leveraging AgentWrite, they construct LongWriter-6k, a dataset containing 6,000 SFT data with output lengths ranging from 2,000 to 32,000 words. Incorporating this dataset into model training allows scaling the output length of existing models to over 10,000 words while maintaining quality. Additionally, they develop LongBench-Write, a comprehensive benchmark for evaluating ultra-long generation capabilities. Their 9B parameter model, further improved through Direct Preference Optimization (DPO), achieves state-of-the-art performance on this benchmark, surpassing even larger proprietary models. The work demonstrates that existing long-context LLMs possess the potential for larger output windows, unlocked through alignment with extended output data.
arXiv

---

## 5. Deepfake & Anomaly Detection
- **Capture Artifacts for Deepfake Detection** [paper](https://paperswithcode.com/paper/capture-artifacts-via-progressive)  
    **Tags**: #deepfake #detection #AI  
    **Summary**: This research introduces a method for detecting deepfakes by progressively disentangling and purifying blended identities, improving detection accuracy.

- **AnomalyNCD** [paper](https://paperswithcode.com/paper/anomalyncd-towards-novel-anomaly-class)  
    **Tags**: #anomaly-detection #AI  
    **Summary**: AnomalyNCD proposes a novel approach to anomaly detection, focusing on identifying new anomaly classes in AI systems.

---

## 6. Optimization Techniques & Frameworks
- **Hot-DoG** [paper](https://paperswithcode.com/paper/tuning-free-coreset-markov-chain-monte-carlo)  
    **Tags**: #optimization #bayesian-optimization #frameworks  
    **Summary**: Hot-DoG introduces a tuning-free coreset approach for Markov Chain Monte Carlo methods, enhancing optimization efficiency in AI models.

---

## 7. Speech Models
- **Qwen2-Audio**  
    **Tags**: #speech #audio #AI-models  
    **Summary**: Qwen2-Audio is an advanced speech model designed to improve audio processing capabilities in AI systems.

- **LLaMA-Omni** [paper](https://paperswithcode.com/paper/llama-omni-seamless-speech-interaction-with)  
    **Tags**: #speech #audio #AI-models  
    **Summary**: LLaMA-Omni offers seamless speech interaction with large language models, enhancing real-time communication capabilities.

- **Moonshine: Speech Recognition for Live Transcription and Voice Commands** [paper](https://paperswithcode.com/paper/moonshine-speech-recognition-for-live)
    **Tags**: #speech-recognition #live-transcription #voice-commands #AI
    **Summary**: This paper introduces Moonshine, a family of speech recognition models optimized for live transcription and voice command processing. Moonshine is based on an encoder-decoder transformer architecture and employs Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings. The model is trained on speech segments of various lengths, but without using zero-padding, leading to greater efficiency for the encoder during inference time.

---

## 8. CUDA & Computing Basics
- **How CUDA Programming Works** [YouTube](https://www.youtube.com/watch?v=n6M8R8-PlnE&t=75s)  
    **Tags**: #CUDA #computing  
    **Summary**: This YouTube video provides an overview of CUDA programming, explaining its functionality and applications in parallel computing.

---

## 9. Poisoning Attacks
- **NoiseAttack** [paper](https://paperswithcode.com/paper/noiseattack-an-evasive-sample-specific-multi)  
    **Tags**: #poisoning #security #AI  
    **Summary**: NoiseAttack discusses an evasive, sample-specific multi-domain attack strategy, highlighting vulnerabilities in AI systems.

- **CodeBreaker** [paper](https://paperswithcode.com/paper/an-llm-assisted-easy-to-trigger-backdoor)  
    **Tags**: #poisoning #security #AI  
    **Summary**: CodeBreaker presents an LLM-assisted, easy-to-trigger backdoor attack, emphasizing the need for robust security measures in AI.

---

## 10. Retrieval-Augmented Generation (RAG)
- **MemoRAG** [paper](https://paperswithcode.com/paper/memorag-moving-towards-next-gen-rag-via)  
    **Tags**: #RAG #retrieval #AI  
    **Summary**: MemoRAG explores advanced techniques in retrieval-augmented generation for more effective information synthesis in AI systems.

- **KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation** [paper](https://paperswithcode.com/paper/2409-13731)  
    **Tags**: #KAG #knowledge-augmentation #professional-domains #AI
    **Summary**: This paper explores Knowledge Augmented Generation (KAG) as a method to enhance large language models (LLMs) for specific professional domains, focusing on the integration of domain-specific knowledge to improve accuracy and relevance in outputs.

---

## 11. Security Frameworks & Adversarial Toolkits
- **AdvSecureNet** [paper](https://paperswithcode.com/paper/advsecurenet-a-python-toolkit-for-adversarial)  
    **Tags**: #security #pytorch #AI  
    **Summary**: AdvSecureNet provides a Python toolkit for adversarial attack simulation in PyTorch, designed to test and improve AI robustness.

---

## 12. Open-Source Projects & Infrastructure
- **LLaMA-Factory**
    **Tags**: #open-source #infrastructure #AI  
    **Summary**: LLaMA-Factory is an open-source infrastructure project aimed at streamlining large language model deployment.

- **SWIFT** [paper](https://paperswithcode.com/paper/swift-a-scalable-lightweight-infrastructure)  
    **Tags**: #open-source #infrastructure #AI  
    **Summary**: SWIFT provides scalable, lightweight infrastructure for efficient LLM deployment, designed to minimize resource consumption.

---

## 13. User Interface (UI) & GUI Tools
- **Huggingface - AutoTrain** [paper](https://paperswithcode.com/paper/autotrain-no-code-training-for-state-of-the)  
    **Tags**: #UI #GUI #AI  
    **Summary**: AutoTrain by Huggingface introduces a no-code solution for training machine learning models, simplifying model customization and deployment.

---

## 14. Hallucination Reduction & Factuality
- **Prereq-Tune** [paper](https://paperswithcode.com/paper/fictitious-synthetic-data-can-improve-llm)  
    **Tags**: #hallucination #factuality #AI  
    **Summary**: Prereq-Tune utilizes synthetic data to reduce hallucination in large language models, enhancing factual accuracy.


---

## 15. Energy Use, Environmental Impact of AI

- **The carbon emissions of writing and illustrating are lower for AI than for humans** [paper](https://www.nature.com/articles/s41598-024-54271-x.pdf)  
    **Tags**: #carbon-emissions #AI-efficiency #environment #sustainability
    **Summary**: This study compares the carbon emissions associated with writing and illustrating tasks performed by AI systems versus humans. The findings indicate that AI systems produce significantly lower carbon emissions for these creative tasks, highlighting their potential environmental benefits in content creation processes.


---

## 16. Image Animation

- **Hallo2**: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation** [paper](https://paperswithcode.com/paper/hallo2-long-duration-and-high-resolution)  
    **Tags**: #image-animation #audio-driven #high-resolution #long-duration
    **Summary**: This paper introduces Hallo2, an advanced method for generating long-duration, high-resolution portrait image animations driven by audio inputs. The approach addresses challenges like appearance drift and temporal artifacts by implementing augmentation strategies within the image space of conditional motion frames. Notably, Hallo2 achieves 4K resolution video generation and incorporates adjustable semantic textual labels for portrait expressions, enhancing controllability and content diversity. Extensive experiments demonstrate state-of-the-art performance in producing rich, controllable content at 4K resolution for durations extending up to tens of minutes.