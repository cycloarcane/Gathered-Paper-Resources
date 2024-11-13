# Gathered Paper Resources

Welcome to the **Gathered Paper Resources** repository! This repository serves as an aggregation of valuable resources, references, and insights from various academic papers that I have reviewed related to Artificial Intelligence (AI), Large Language Models (LLMs), and related technologies.

## 1. Agentic Frameworks & Task Management

### Agent S - Uses computers like a human
**Paper**: [Link](https://paperswithcode.com/paper/agent-s-an-open-agentic-framework-that-uses)  
**Summary**: Agent S introduces an open agentic framework that enables AI systems to utilize computers similarly to humans, enhancing task management and execution capabilities.  
**Tags**: #agent #task-management #AI

### TaskGen
**Paper**: [Link](https://paperswithcode.com/paper/taskgen-a-task-based-memory-infused-agentic)  
**Summary**: TaskGen presents a task-based, memory-infused agentic framework designed to improve AI's ability to manage and execute complex tasks efficiently.  
**Tags**: #agent #task-management #AI

## 2. Benchmarks & Testing

### OSWorld
**Summary**: OSWorld provides a comprehensive benchmarking environment for evaluating AI performance across various operating systems.  
**Tags**: #benchmark #testing #OS

### AndroidWorld
**Summary**: AndroidWorld offers a specialized platform for testing and benchmarking AI applications within the Android ecosystem.  
**Tags**: #benchmark #testing #OS

### WindowsAgentArena
**Summary**: WindowsAgentArena serves as a testing ground for assessing AI agents' performance in Windows operating environments.  
**Tags**: #benchmark #testing #OS

## 3. Detection & Monitoring

### Human intelligence can safeguard against artificial intelligence
**Paper**: [Link](https://www.nature.com/articles/s41598-024-76218-y?fromPaywallRec=false)  
**Summary**: This study explores individual differences in discerning human-generated texts from AI-generated ones, highlighting the role of human intelligence in monitoring AI outputs.  
**Tags**: #detection #monitoring #human-AI

### LLMSCAN
**Paper**: [Link](https://paperswithcode.com/paper/llmscan-causal-scan-for-llm-misbehavior)  
**Summary**: LLMSCAN introduces a causal scanning method to detect misbehavior in large language models, enhancing AI system reliability.  
**Tags**: #detection #monitoring #AI

### From Imitation to Introspection
**Paper**: [Link](https://paperswithcode.com/paper/from-imitation-to-introspection-probing-self)  
**Summary**: This paper explores the emergence of self-consciousness in language models, transitioning from mere imitation to introspective capabilities. It examines the extent to which these models can exhibit self-referential behaviors, delving into the philosophical implications of AI's evolving cognitive functions.  
**Tags**: #self-consciousness #AI-philosophy #introspection #detection

### PARIS: A Practical, Adaptive Trace-Fetching and Real-Time Malicious Behavior Detection System   
**Paper**: [Link](https://arxiv.org/abs/2411.01273)   
**Summary**: This paper introduces PARIS, a system designed for real-time detection of malicious behaviors on Windows platforms. Utilizing Event Tracing for Windows (ETW), PARIS adaptively monitors and collects data related to potentially malicious API calls and call stacks, significantly reducing data collection overhead. This efficiency enables the system to monitor a broader range of APIs, enhancing its ability to detect complex attack behaviors. Evaluations demonstrate that PARIS reduces data collection by over 98.8% compared to raw ETW traces, while maintaining detection accuracy comparable to existing methods that incur higher overhead.
**Tags**: #malware-detection #real-time-monitoring #ETW #cybersecurity


## 4. AI Architectures & Models

### Kolmogorov-Arnold Transformers
**Paper**: [Link](https://paperswithcode.com/paper/kolmogorov-arnold-transformer)  
**Summary**: This paper presents the Kolmogorov-Arnold Transformer, a novel architecture aimed at improving the efficiency and performance of transformer models.  
**Tags**: #architecture #transformer #AI

### Aria
**Paper**: [Link](https://paperswithcode.com/paper/aria-an-open-multimodal-native-mixture-of)  
**Summary**: Aria is an open multimodal-native mixture of experts model designed to handle diverse data types, enhancing AI's multimodal processing capabilities.  
**Tags**: #architecture #multimodal #AI

### Hunyuan-Large
**Paper**: [Link](https://paperswithcode.com/paper/hunyuan-large-an-open-source-moe-model-with)  
**Summary**: This paper introduces Hunyuan-Large, the largest open-source Transformer-based mixture of experts model, featuring a total of 389 billion parameters with 52 billion activated parameters and the ability to handle up to 256K tokens. Hunyuan-Large demonstrates superior performance in various benchmarks, including language understanding and generation, logical reasoning, and coding tasks, surpassing smaller models while maintaining competitive performance against much larger counterparts like LLama3.1-405B.  
**Tags**: #MoE #transformer #open-source #AI

### LongWriter
**Paper**: [Link](https://paperswithcode.com/paper/longwriter-unleashing-10000-word-generation)  
**Summary**: This paper addresses the limitation of current long-context large language models (LLMs) that struggle to generate outputs exceeding 2,000 words. The authors introduce AgentWrite, an agent-based pipeline that decomposes ultra-long generation tasks into subtasks, enabling LLMs to produce coherent outputs exceeding 20,000 words.  
**Tags**: #long-context #LLM #text-generation #AI

### TableGPT2
**Paper**: [Link](https://paperswithcode.com/paper/tablegpt2-a-large-multimodal-model-with)  
**Summary**: TableGPT2 is a large-scale multimodal model specifically designed to integrate tabular data into AI applications. Trained on over 593,800 tables and 2.36 million high-quality query-table-output tuples, it excels in table-centric tasks while maintaining robust general language and coding capabilities.  
**Tags**: #multimodal-model #tabular-data #AI

### R-CoT
**Paper**: [Link](https://paperswithcode.com/paper/r-cot-reverse-chain-of-thought-problem)  
**Summary**: This paper introduces R-CoT, a two-stage Reverse Chain-of-Thought geometry problem generation pipeline designed to enhance the geometric reasoning capabilities of Large Multimodal Models (LMMs).  
**Tags**: #R-CoT #geometric-reasoning #multimodal-models #AI

### Breaking the Low-Rank Dilemma of Linear Attention   
**Paper**: [Link](https://paperswithcode.com/paper/breaking-the-low-rank-dilemma-of-linear)  
**Summary**: This paper addresses the computational inefficiencies of the Softmax attention mechanism in Transformer models, particularly its quadratic complexity in vision applications. It introduces Rank-Augmented Linear Attention (RALA), which enhances the rank of linear attention's feature map, enabling it to model complex spatial information more effectively. RALA maintains linear complexity while achieving performance comparable to Softmax attention. The authors also present the Rank-Augmented Vision Linear Transformer (RAVLT), which, without additional labels or supervision during training, achieves an 84.4% Top-1 accuracy on ImageNet-1k with only 26M parameters and 4.6G FLOPs, surpassing previous linear attention mechanisms.
**Tags**: #linear-attention #transformer #vision-models #RALA #RAVLT

### LoLCATs: On Low-Rank Linearizing of Large Language Models   
**Paper**: [Link](https://arxiv.org/abs/2410.10254)   
**Summary**: This paper introduces LoLCATs (Low-rank Linear Conversion via Attention Transfer), a method designed to enhance the efficiency of large language models (LLMs) by replacing their quadratic attention mechanisms with subquadratic linear attentions. The approach involves two key steps: training linear attentions to approximate the outputs of softmax attentions through attention transfer, and then applying low-rank adaptation (LoRA) to adjust for any approximation errors. LoLCATs significantly improves the quality and scalability of linearized LLMs, achieving state-of-the-art performance with substantially reduced memory and computational requirements. Notably, it enables the creation of linearized models up to 405 billion parameters, marking a significant advancement in the field.
**Tags**: #linear-attention #LLM #LoRA #model-efficiency


## 5. Deepfake & Anomaly Detection

### Capture Artifacts for Deepfake Detection
**Paper**: [Link](https://paperswithcode.com/paper/capture-artifacts-via-progressive)  
**Summary**: This research introduces a method for detecting deepfakes by progressively disentangling and purifying blended identities, improving detection accuracy.  
**Tags**: #deepfake #detection #AI

### AnomalyNCD
**Paper**: [Link](https://paperswithcode.com/paper/anomalyncd-towards-novel-anomaly-class)  
**Summary**: AnomalyNCD proposes a novel approach to anomaly detection, focusing on identifying new anomaly classes in AI systems.  
**Tags**: #anomaly-detection #AI

## 6. Optimization Techniques & Frameworks

### Hot-DoG
**Paper**: [Link](https://paperswithcode.com/paper/tuning-free-coreset-markov-chain-monte-carlo)  
**Summary**: Hot-DoG introduces a tuning-free coreset approach for Markov Chain Monte Carlo methods, enhancing optimization efficiency in AI models.  
**Tags**: #optimization #bayesian-optimization #frameworks

## 7. Speech Models

### Qwen2-Audio
**Summary**: Qwen2-Audio is an advanced speech model designed to improve audio processing capabilities in AI systems.  
**Tags**: #speech #audio #AI-models

### LLaMA-Omni
**Paper**: [Link](https://paperswithcode.com/paper/llama-omni-seamless-speech-interaction-with)  
**Summary**: LLaMA-Omni offers seamless speech interaction with large language models, enhancing real-time communication capabilities.  
**Tags**: #speech #audio #AI-models

### Moonshine
**Paper**: [Link](https://paperswithcode.com/paper/moonshine-speech-recognition-for-live)  
**Summary**: This paper introduces Moonshine, a family of speech recognition models optimized for live transcription and voice command processing. Moonshine is based on an encoder-decoder transformer architecture and employs Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings.  
**Tags**: #speech-recognition #live-transcription #voice-commands #AI

## 8. CUDA & Computing Basics

### How CUDA Programming Works
**Source**: [YouTube](https://www.youtube.com/watch?v=n6M8R8-PlnE&t=75s)  
**Summary**: This YouTube video provides an overview of CUDA programming, explaining its functionality and applications in parallel computing.  
**Tags**: #CUDA #computing

## 9. Poisoning Attacks

### NoiseAttack
**Paper**: [Link](https://paperswithcode.com/paper/noiseattack-an-evasive-sample-specific-multi)  
**Summary**: NoiseAttack discusses an evasive, sample-specific multi-domain attack strategy, highlighting vulnerabilities in AI systems.  
**Tags**: #poisoning #security #AI

### CodeBreaker
**Paper**: [Link](https://paperswithcode.com/paper/an-llm-assisted-easy-to-trigger-backdoor)  
**Summary**: CodeBreaker presents an LLM-assisted, easy-to-trigger backdoor attack, emphasizing the need for robust security measures in AI.  
**Tags**: #poisoning #security #AI

## 10. Retrieval-Augmented Generation (RAG)

### MemoRAG
**Paper**: [Link](https://paperswithcode.com/paper/memorag-moving-towards-next-gen-rag-via)  
**Summary**: MemoRAG explores advanced techniques in retrieval-augmented generation for more effective information synthesis in AI systems.  
**Tags**: #RAG #retrieval #AI

### KAG
**Paper**: [Link](https://paperswithcode.com/paper/2409-13731)  
**Summary**: This paper explores Knowledge Augmented Generation (KAG) as a method to enhance large language models (LLMs) for specific professional domains, focusing on the integration of domain-specific knowledge to improve accuracy and relevance in outputs.  
**Tags**: #KAG #knowledge-augmentation #professional-domains #AI

## 11. Security Frameworks & Adversarial Toolkits

### AdvSecureNet
**Paper**: [Link](https://paperswithcode.com/paper/advsecurenet-a-python-toolkit-for-adversarial)  
**Summary**: AdvSecureNet provides a Python toolkit for adversarial attack simulation in PyTorch, designed to test and improve AI robustness.  
**Tags**: #security #pytorch #AI

### Neural Fingerprints for Adversarial Attack Detection
**Paper**: [Link](https://paperswithcode.com/paper/neural-fingerprints-for-adversarial-attack)  
**Summary**: This paper introduces a method to detect adversarial attacks on AI models by leveraging neural fingerprints. The approach involves creating unique identifiers for neural networks, enabling the identification of unauthorized modifications or attacks.  
**Tags**: #adversarial-attack #detection #neural-fingerprints #AI

### Trustworthy Federated Learning: Privacy, Security, and Beyond
**Paper**: [Link](https://arxiv.org/abs/2411.01583)
**Summary**: This comprehensive survey examines the security and privacy challenges inherent in Federated Learning (FL). It highlights vulnerabilities in communication channels and potential cyber threats within decentralized networks. The authors discuss various defensive strategies to mitigate these risks, explore FL applications across different sectors, and propose future research directions to enhance the security and efficiency of FL systems.
**Tag**s: #federated-learning #privacy #security #decentralized-networks

## 12. Open-Source Projects & Infrastructure

### LLaMA-Factory
**Summary**: LLaMA-Factory is an open-source infrastructure project aimed at streamlining large language model deployment.  
**Tags**: #open-source #infrastructure #AI

### SWIFT
**Paper**: [Link](https://paperswithcode.com/paper/swift-a-scalable-lightweight-infrastructure)  
**Summary**: SWIFT provides scalable, lightweight infrastructure for efficient LLM deployment, designed to minimize resource consumption.  
**Tags**: #open-source #infrastructure #AI

## 13. User Interface (UI) & GUI Tools

### Huggingface - AutoTrain
**Paper**: [Link](https://paperswithcode.com/paper/autotrain-no-code-training-for-state-of-the)  
**Summary**: AutoTrain by Huggingface introduces a no-code solution for training machine learning models, simplifying model customization and deployment.  
**Tags**: #UI #GUI #AI

## 14. Hallucination Reduction & Factuality

### Prereq-Tune
**Paper**: [Link](https://paperswithcode.com/paper/fictitious-synthetic-data-can-improve-llm)  
**Summary**: Prereq-Tune utilizes synthetic data to reduce hallucination in large language models, enhancing factual accuracy.  
**Tags**: #hallucination #factuality #AI

## 15. Energy Use, Environmental Impact of AI

### The carbon emissions of writing and illustrating
**Paper**: [Link](https://www.nature.com/articles/s41598-024-54271-x.pdf)  
**Summary**: This study compares the carbon emissions associated with writing and illustrating tasks performed by AI systems versus humans. The findings indicate that AI systems produce significantly lower carbon emissions for these creative tasks.  
**Tags**: #carbon-emissions #AI-efficiency #environment #sustainability

## 16. Image Animation

### Hallo2
**Paper**: [Link](https://paperswithcode.com/paper/hallo2-long-duration-and-high-resolution)  
**Summary**: This paper introduces Hallo2, an advanced method for generating long-duration, high-resolution portrait image animations driven by audio inputs. The approach addresses challenges like appearance drift and temporal artifacts by implementing augmentation strategies within the image space of conditional motion frames.  
**Tags**: #image-animation #audio-driven #high-resolution #long-duration

## 17. Document Processing Tools

### Docling Technical Report
**Paper**: [Link](https://paperswithcode.com/paper/docling-technical-report)  
**Summary**: This technical report introduces Docling, an open-source, MIT-licensed package designed for efficient PDF document conversion. Leveraging advanced AI models like DocLayNet for layout analysis and TableFormer for table structure recognition, Docling operates effectively on standard hardware with minimal resource requirements.  
**Tags**: #document-conversion #PDF #open-source #AI

## 18. Cybersecurity Assistants

### IntellBot: Retrieval Augmented LLM Chatbot for Cyber Threat Knowledge Delivery
**Paper**: [Link](https://arxiv.org/abs/2411.05442)   
**Summary**: This paper introduces IntellBot, an advanced cybersecurity chatbot leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques. IntellBot aggregates information from diverse sources to create a comprehensive knowledge base encompassing known vulnerabilities, recent cyberattacks, and emerging threats. It provides tailored responses, serving as a central hub for cybersecurity insights. Performance evaluations indicate high accuracy, with BERT scores above 0.8 and cosine similarity scores ranging from 0.8 to 1.
**Tags**: #cybersecurity #chatbot #LLM #RAG #threat-intelligence

## 19. Miscellaneous Interesting

### Analysing the Cultural Dimensions of Cybercriminal Groups: A Case Study on the Conti Ransomware Group
**Paper**: [Link](https://arxiv.org/abs/2411.02548)   
**Summary**: This study explores the cultural aspects of cybercriminal organizations, focusing on the Conti ransomware group. By analyzing leaked internal communications, the authors apply Hofstede's and Meyer's cultural dimension frameworks to understand the group's operational behaviors. The findings suggest that cultural analysis can enhance cyber threat intelligence and attribution processes, offering deeper insights into the modus operandi of such groups.
**Tags**: #cybersecurity #cultural-analysis #threat-intelligence #ransomware
