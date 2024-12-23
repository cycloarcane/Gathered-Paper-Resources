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

### Reinforcement Learning for Quantum Tiq-Taq-Toe   
**Paper**: [Link](https://paperswithcode.com/paper/reinforcement-learning-for-quantum-tiq-taq)   
**Summary**: This paper explores the application of reinforcement learning (RL) to Quantum Tiq-Taq-Toe, a quantum variant of the classic Tic-Tac-Toe game. The study highlights the challenges in representing quantum games classically due to partial observability and exponential state complexity. In Quantum Tiq-Taq-Toe, states are observed through measurements and move history, with each move potentially collapsing the quantum state. The authors propose RL methods to navigate these complexities, aiming to serve as an accessible testbed for integrating quantum computing and machine learning.
**Tags**: #reinforcement-learning #quantum-computing #game-theory #AI

### FinRobot: AI Agent for Equity Research and Valuation with Large Language Models   
**Paper**: [Link](https://paperswithcode.com/paper/finrobot-ai-agent-for-equity-research-and)   
**Summary**: FinRobot introduces an AI agent framework tailored for equity research, integrating both quantitative and qualitative analyses to emulate human analyst reasoning. It comprises three specialized agents:

    Data-CoT Agent: Aggregates diverse data sources for robust financial integration.
    Concept-CoT Agent: Mimics analyst reasoning to generate actionable insights.
    Thesis-CoT Agent: Synthesizes insights into coherent investment theses and reports.

FinRobot delivers comprehensive company analyses with precise numerical data, industry-specific valuation metrics, and realistic risk assessments. Its dynamically updatable data pipeline ensures timely and relevant research, adapting seamlessly to new financial information.   
**Tags**: #AI-agent #equity-research #valuation #LLM   


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

### Byte Latent Transformer: Patches Scale Better Than Tokens
**Paper**: [Link](https://paperswithcode.com/paper/byte-latent-transformer-patches-scale-better)  
**Summary**: The Byte Latent Transformer (BLT) is a novel byte-level large language model (LLM) architecture that matches the performance of traditional tokenization-based LLMs while enhancing inference efficiency and robustness. BLT encodes raw bytes into dynamically sized patches, which serve as the primary computational units. These patches are segmented based on the entropy of the subsequent byte, allocating more computational resources to complex data segments. A FLOP-controlled scaling study of byte-level models up to 8 billion parameters and 4 trillion training bytes demonstrates that BLT can scale models trained on raw bytes without a fixed vocabulary. This approach improves both training and inference efficiency by dynamically selecting longer patches for predictable data, leading to qualitative enhancements in reasoning and generalization. Overall, for fixed inference costs, BLT exhibits superior scaling compared to tokenization-based models by concurrently increasing both patch and model sizes.  
**Tags**: #LLM #byte-level-modeling #transformer #AI

### An Evolved Universal Transformer Memory
**Paper**: [Link](https://paperswithcode.com/paper/an-evolved-universal-transformer-memory)  
**Summary**: This paper introduces Neural Attention Memory Models (NAMMs), a learned network for memory management that enhances both the performance and efficiency of transformers. NAMMs provide distinct latent contexts, focusing on the most relevant information for individual layers and attention heads. They condition exclusively on the values in the produced attention matrices, making them universally applicable to any model utilizing self-attention. Training NAMMs on a small set of problems yields significant performance improvements across multiple long-context benchmarks while reducing the model's input contexts to a fraction of their original sizes. Notably, NAMMs trained solely on language demonstrate zero-shot transfer capabilities to entirely new transformer architectures across different input modalities, with benefits extending to vision and reinforcement learning tasks.  
**Tags**: #transformers #memory-management #self-attention #AI


### From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge
**Paper**: [Link](https://paperswithcode.com/paper/from-generation-to-judgment-opportunities-and)  
**Summary**: This paper provides a comprehensive survey of the emerging "LLM-as-a-judge" paradigm, where Large Language Models (LLMs) are utilized for scoring, ranking, or selection across various tasks and applications. It introduces a detailed taxonomy exploring three dimensions: what to judge, how to judge, and where to judge. The study compiles benchmarks for evaluating LLM-based judgment systems and discusses key challenges and future research directions in this field.  
**Tags**: #LLM #AI-evaluation #survey #AI

### Initialization Using Update Approximation Is Effective for Training Transformers
**Paper**: [Link](https://paperswithcode.com/paper/initialization-using-update-approximation-is)  
**Summary**: This paper introduces a novel initialization method that approximates full fine-tuning within low-rank subspaces for training Transformer models. By employing a carefully designed initialization strategy, the approach achieves optimal scaling for high-rank gradient updates without the need for extensive hyperparameter tuning. The method demonstrates significant efficiency gains, using 27-90 times fewer parameters than standard low-rank adaptation techniques, while surpassing their performance across various tasks, including mathematical reasoning and language understanding.  
**Tags**: #transformers #initialization #low-rank-fine-tuning #AI

### Star Attention: Efficient LLM Inference over Long Sequences
**Paper**: [Link](https://paperswithcode.com/paper/star-attention-efficient-llm-inference-over)  
**Summary**: Star Attention introduces a two-phase block-sparse approximation to enhance the efficiency of Transformer-based Large Language Models (LLMs) during inference on long sequences. The first phase employs blockwise-local attention processed in parallel across multiple hosts, while the second phase allows query and response tokens to attend to all prior cached tokens through sequence-global attention. This method reduces memory requirements and inference time by up to 11x, maintaining 95-100% accuracy.  
**Tags**: #LLM #attention-mechanism #inference-optimization #block-sparse-attention

### Multi-Programming Language Sandbox for LLMs
**Paper**: [Link](https://paperswithcode.com/paper/multi-programming-language-sandbox-for-llms)  
**Summary**: This paper introduces MPLSandbox, an out-of-the-box framework designed to provide unified and comprehensive feedback from compiler and analysis tools for Large Language Models (LLMs). MPLSandbox automatically identifies the programming language of the code, compiling and executing it within an isolated sub-sandbox to ensure safety and stability. It integrates both traditional and LLM-based code analysis tools, offering a thorough examination of generated code. The framework can be seamlessly incorporated into the training and deployment of LLMs to enhance the quality and correctness of their code outputs, thereby streamlining workflows for various LLM-based code-related tasks and reducing development costs.  
**Tags**: #LLM #code-analysis #sandbox #programming-languages

### O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?
**Paper**: [Link](https://paperswithcode.com/paper/o1-replication-journey-part-2-surpassing-o1)  
**Summary**: This study critically examines current methods for replicating OpenAI's O1 model capabilities, focusing on the often undisclosed use of knowledge distillation techniques. The authors demonstrate that simple distillation from O1's API, combined with supervised fine-tuning, can achieve superior performance on complex mathematical reasoning tasks. Experiments reveal that a base model fine-tuned on tens of thousands of O1-distilled samples outperforms O1-preview on the American Invitational Mathematics Examination (AIME) with minimal technical complexity. Additionally, the study explores the generalization capabilities of O1-distilled models across diverse tasks, including hallucination, safety, and open-domain question answering. Notably, models trained solely on mathematical problem-solving data exhibited strong generalization to open-ended QA tasks and showed reduced susceptibility to sycophancy after fine-tuning. The authors advocate for transparency in AI research and caution against over-reliance on distillation approaches, emphasizing the importance of first-principles thinking in developing capable AI systems.  
**Tags**: #O1-replication #knowledge-distillation #AI-research #transparency

### Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks   
**Paper**: [Link](https://paperswithcode.com/paper/kinetix-investigating-the-training-of-general)   
**Summary**: Kinetix introduces an open-ended framework for training general reinforcement learning (RL) agents across a vast array of 2D physics-based tasks. By procedurally generating tens of millions of environments—ranging from robotic locomotion and grasping to video games—the framework enables agents to develop robust physical reasoning capabilities. Utilizing the hardware-accelerated physics engine Jax2D, Kinetix efficiently simulates billions of environment steps during training. Agents trained within this framework demonstrate strong zero-shot performance on unseen, human-designed environments and exhibit enhanced adaptability when fine-tuned on specific tasks, outperforming agents trained from scratch.   
**Tags**: #reinforcement-learning #physics-based-tasks #generalization #AI

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

### RESOLVE: Relational Reasoning with Symbolic and Object-Level Features Using Vector Symbolic Processing   
**Paper**: [Link](https://paperswithcode.com/paper/resolve-relational-reasoning-with-symbolic)   
**Summary**: RESOLVE introduces a neuro-vector symbolic architecture that integrates object-level features with relational representations in high-dimensional spaces. Utilizing operations like bundling (summation) and binding (Hadamard product), it enables the coexistence of object-level and relational information without interference. The model features a novel attention mechanism operating in a bipolar high-dimensional space, facilitating efficient attention score computation. RESOLVE demonstrates improved generalizability and accuracy in tasks requiring both pure and partial relational reasoning, such as sorting and mathematical problem-solving, compared to existing methods.   
**Tags**: #relational-reasoning #vector-symbolic-processing #neuro-symbolic-AI #AI

### Qwen2.5-Coder Technical Report  
**Paper**: [Link](https://paperswithcode.com/paper/qwen2-5-coder-technical-report)  
**Summary**: The Qwen2.5-Coder series is an advancement over its predecessor, CodeQwen1.5, featuring six models ranging from 0.5B to 32B parameters. Built upon the Qwen2.5 architecture, these code-specific models are pretrained on a vast corpus exceeding 5.5 trillion tokens. Through meticulous data cleaning, scalable synthetic data generation, and balanced data mixing, Qwen2.5-Coder demonstrates impressive code generation capabilities while retaining general and mathematical skills. Evaluations across more than ten benchmarks—including code generation, completion, reasoning, and repair—show that Qwen2.5-Coder consistently outperforms larger models of the same size, achieving state-of-the-art performance.  
**Tags**: #code-generation #LLM #AI #Qwen2.5-Coder   


## 5. Deepfake & Anomaly Detection


### Capture Artifacts for Deepfake Detection
**Paper**: [Link](https://paperswithcode.com/paper/capture-artifacts-via-progressive)  
**Summary**: This research introduces a method for detecting deepfakes by progressively disentangling and purifying blended identities, improving detection accuracy.  
**Tags**: #deepfake #detection #AI

### AnomalyNCD
**Paper**: [Link](https://paperswithcode.com/paper/anomalyncd-towards-novel-anomaly-class)  
**Summary**: AnomalyNCD proposes a novel approach to anomaly detection, focusing on identifying new anomaly classes in AI systems.  
**Tags**: #anomaly-detection #AI

### FairAdapter
**Paper**: [Link](https://paperswithcode.com/paper/fairadapter-detecting-ai-generated-images)  
**Summary**: FairAdapter addresses the challenge of detecting AI-generated images, which often exhibit inconsistencies across different content types due to overfitting in existing detection models. This framework enhances detection fairness by mitigating performance disparities across various image contents. Comparative evaluations demonstrate that FairAdapter outperforms current state-of-the-art methods in achieving more consistent detection outcomes.  
**Tags**: #AI-generated-images #detection-fairness #deep-learning #image-forensics


## 6. Optimization Techniques & Frameworks

### Hot-DoG
**Paper**: [Link](https://paperswithcode.com/paper/tuning-free-coreset-markov-chain-monte-carlo)  
**Summary**: Hot-DoG introduces a tuning-free coreset approach for Markov Chain Monte Carlo methods, enhancing optimization efficiency in AI models.  
**Tags**: #optimization #bayesian-optimization #frameworks

## 7. Speech Models

### F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching   
**Paper**: [Link](https://paperswithcode.com/paper/f5-tts-a-fairytaler-that-fakes-fluent-and)   
**Summary**: F5-TTS introduces a fully non-autoregressive text-to-speech (TTS) system utilizing flow matching with a Diffusion Transformer (DiT). By padding text inputs to match the length of speech inputs and performing denoising for speech generation, it eliminates the need for complex components like duration models, text encoders, and phoneme alignment. Enhancements include the integration of ConvNeXt for refined text representation and the implementation of a Sway Sampling strategy during inference, which significantly boosts performance and efficiency. Trained on a 100K-hour multilingual dataset, F5-TTS demonstrates natural and expressive zero-shot capabilities, seamless code-switching, and efficient speed control, achieving an inference real-time factor (RTF) of 0.15.   
**Tags**: #text-to-speech #flow-matching #Diffusion-Transformer #AI

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

### Quantum Computing: Hype vs. Reality
**Video**: [Link](https://www.youtube.com/watch?v=-1PsQIciMEc)  
**Summary**: This video critically examines the current state of quantum computing, distinguishing between the technology's realistic capabilities and the surrounding hype. It provides insights into the progress made in the field, the challenges that remain, and the practical implications of quantum computing advancements.  
**Tags**: #quantum-computing #technology #analysis #reality-check


### Quantum Computers Explained – Limits of Human Technology
**Video**: [Link](https://www.youtube.com/watch?v=JhHMJCUmq28)  
**Summary**: This educational video provides an accessible overview of quantum computing, exploring its fundamental principles and potential to revolutionize technology. It delves into the limitations of classical computers and illustrates how quantum mechanics can overcome these barriers, enabling unprecedented computational capabilities. The video also discusses the challenges in developing quantum computers and their prospective applications across various fields.  
**Tags**: #quantum-computing #technology #education #computing


### Every attention head explained   
**Source**: [YouTube](https://www.youtube.com/watch?v=qR56cyMdDXg)   
**Summary**: This educational video provides an in-depth explanation of attention heads within transformer models, a fundamental component in modern machine learning architectures. It covers the mechanics of attention mechanisms, their role in processing input data, and how they contribute to the performance of models in tasks like natural language processing and computer vision.  
**Tags**: #attention-heads #transformer-models #machine-learning #AI


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


### Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models
**Paper**: [Link](https://paperswithcode.com/paper/auto-rag-autonomous-retrieval-augmented)  
**Summary**: Auto-RAG introduces an autonomous iterative retrieval model that leverages the decision-making capabilities of Large Language Models (LLMs) to enhance Retrieval-Augmented Generation (RAG). Unlike traditional methods that rely on few-shot prompting or manually constructed rules, Auto-RAG enables LLMs to engage in multi-turn dialogues with retrievers, systematically planning retrievals and refining queries to acquire valuable knowledge. This process continues until sufficient external information is gathered, at which point the results are presented to the user. Experiments demonstrate that Auto-RAG effectively utilizes the reasoning and decision-making abilities of LLMs, leading to outstanding performance across six benchmarks. Additionally, Auto-RAG autonomously adjusts the number of iterations based on the difficulty of the questions and the utility of the retrieved knowledge, without requiring human intervention. The iterative retrieval process is expressed in natural language, enhancing interpretability and providing users with a more intuitive experience. The code is available at [https://github.com/ictnlp/Auto-RAG](https://github.com/ictnlp/Auto-RAG).  
**Tags**: #Retrieval-Augmented-Generation #LLM #iterative-retrieval #AI

### MemoRAG
**Paper**: [Link](https://paperswithcode.com/paper/memorag-moving-towards-next-gen-rag-via)  
**Summary**: MemoRAG explores advanced techniques in retrieval-augmented generation for more effective information synthesis in AI systems.  
**Tags**: #RAG #retrieval #AI

### KAG
**Paper**: [Link](https://paperswithcode.com/paper/2409-13731)  
**Summary**: This paper explores Knowledge Augmented Generation (KAG) as a method to enhance large language models (LLMs) for specific professional domains, focusing on the integration of domain-specific knowledge to improve accuracy and relevance in outputs.  
**Tags**: #KAG #knowledge-augmentation #professional-domains #AI

## 11. Security Frameworks & Adversarial Toolkits

### Best-of-N Jailbreaking
**Paper**: [Link](https://paperswithcode.com/paper/best-of-n-jailbreaking)  
**Summary**: Best-of-N (BoN) Jailbreaking is a black-box algorithm designed to bypass safety measures in advanced AI systems across various modalities. It operates by generating multiple variations of a prompt through augmentations—such as random shuffling or capitalization for text prompts—until a harmful response is elicited. BoN Jailbreaking has demonstrated high attack success rates (ASRs) on closed-source language models, achieving 89% on GPT-4o and 78% on Claude 3.5 Sonnet with 10,000 augmented prompts. It is also effective against state-of-the-art open-source defenses like circuit breakers. The algorithm extends seamlessly to other modalities, successfully jailbreaking vision language models (VLMs) like GPT-4o and audio language models (ALMs) such as Gemini 1.5 Pro using modality-specific augmentations. Notably, BoN's effectiveness improves with an increased number of sampled prompts, following a power-law-like behavior across modalities. Combining BoN with other black-box algorithms, such as optimized prefix attacks, can further enhance its efficacy, achieving up to a 35% increase in ASR. This work highlights the susceptibility of language models to seemingly innocuous input modifications, which can be exploited across different modalities.  
**Tags**: #jailbreaking #AI-safety #black-box-attack #multi-modal-AI


### Granite Guardian
**Paper**: [Link](https://paperswithcode.com/paper/granite-guardian)  
**Summary**: Granite Guardian introduces a suite of models designed to detect risks in prompts and responses, facilitating the safe and responsible use of large language models (LLMs). These models cover multiple risk dimensions, including social bias, profanity, violence, sexual content, unethical behavior, jailbreaking, and hallucination-related risks such as context relevance, groundedness, and answer relevance for retrieval-augmented generation (RAG). Trained on a unique dataset that combines human annotations from diverse sources with synthetic data, Granite Guardian addresses risks often overlooked by traditional detection models. It achieves AUC scores of 0.871 and 0.854 on harmful content and RAG-hallucination-related benchmarks, respectively, making it a generalizable and competitive model in this domain. The models are open-sourced to promote responsible AI development within the community. The code is available at [https://github.com/ibm-granite/granite-guardian](https://github.com/ibm-granite/granite-guardian).  
**Tags**: #risk-detection #LLM #AI-safety #open-source


### garak: A Framework for Security Probing Large Language Models   
**Paper**: [Link](https://paperswithcode.com/paper/garak-a-framework-for-security-probing-large)   
**Summary**: garak (Generative AI Red-teaming and Assessment Kit) is an open-source framework designed to identify vulnerabilities in large language models (LLMs) and dialogue systems. It systematically probes LLMs to uncover potential security weaknesses, providing detailed reports that inform discussions on model alignment and deployment policies.   
**Tags**: #LLM #security #vulnerability-assessment #AI   

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

### ShowUI: One Vision-Language-Action Model for GUI Visual Agent
**Paper**: [Link](https://paperswithcode.com/paper/showui-one-vision-language-action-model-for)  
**Summary**: ShowUI is a vision-language-action model designed to enhance human productivity by enabling AI agents to interact with graphical user interfaces (GUIs) in a manner similar to human perception. Unlike traditional language-based agents that depend on text-rich meta-information, ShowUI incorporates:

1. **UI-Guided Visual Token Selection**: Reduces computational costs by creating a connected graph of UI elements from screenshots, identifying redundant relationships, and selecting tokens during self-attention blocks.
2. **Interleaved Vision-Language-Action Streaming**: Unifies diverse GUI task requirements, effectively managing visual-action history in navigation and pairing multi-turn query-action sequences per screenshot to enhance training efficiency.
3. **High-Quality GUI Instruction-Following Datasets**: Utilizes careful data curation and a resampling strategy to address data type imbalances.

The lightweight 2B model achieves 75.1% accuracy in zero-shot screenshot grounding, reduces redundant visual tokens by 33% during training, and improves performance speed by 1.4x. Experiments across web (Mind2Web), mobile (AITW), and online (MiniWob) environments demonstrate its effectiveness in advancing GUI visual agents.  
**Tags**: #GUI-assistants #vision-language-action #AI #productivity


### The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use
**Paper**: [Link](https://paperswithcode.com/paper/the-dawn-of-gui-agent-a-preliminary-case)  
**Summary**: This study explores Claude 3.5 Computer Use, an AI model designed to function as a graphical user interface (GUI) agent. The authors curated a set of tasks across various domains and software to assess its capabilities. Findings indicate that Claude 3.5 can effectively translate language instructions into desktop actions, showcasing its potential in real-world applications. The study also introduces an out-of-the-box agent framework for deploying API-based GUI automation models, aiming to inspire further research in the GUI agent community.  
**Tags**: #GUI-agent #AI #automation #Claude3.5


### Huggingface - AutoTrain
**Paper**: [Link](https://paperswithcode.com/paper/autotrain-no-code-training-for-state-of-the)  
**Summary**: AutoTrain by Huggingface introduces a no-code solution for training machine learning models, simplifying model customization and deployment.  
**Tags**: #UI #GUI #AI

## 14. Hallucination Reduction & Factuality

### Prereq-Tune
**Paper**: [Link](https://paperswithcode.com/paper/fictitious-synthetic-data-can-improve-llm)  
**Summary**: Prereq-Tune utilizes synthetic data to reduce hallucination in large language models, enhancing factual accuracy.  
**Tags**: #hallucination #factuality #AI

### DecoPrompt: Decoding Prompts Reduces Hallucinations when Large Language Models Meet False Premises   
**Paper**: [Link](https://paperswithcode.com/paper/decoprompt-decoding-prompts-reduces)   
**Summary**: DecoPrompt addresses the issue of hallucinations in large language models (LLMs) when confronted with false premises. The method leverages LLMs to "decode" misleading prompts without generating hallucinated outputs, effectively reducing inaccuracies. Experiments demonstrate DecoPrompt's efficacy across different LLMs and its cross-model transferability, making it applicable to various scenarios, including those involving large models or inaccessible model logits.   
**Tags**: #hallucination-reduction #LLM #false-premises #AI

## 15. Energy Use, Environmental Impact of AI

### The carbon emissions of writing and illustrating
**Paper**: [Link](https://www.nature.com/articles/s41598-024-54271-x.pdf)  
**Summary**: This study compares the carbon emissions associated with writing and illustrating tasks performed by AI systems versus humans. The findings indicate that AI systems produce significantly lower carbon emissions for these creative tasks.  
**Tags**: #carbon-emissions #AI-efficiency #environment #sustainability

## 16. Image Animation

### SynCamMaster: Synchronizing Multi-Camera Video Generation from Diverse Viewpoints
**Paper**: [Link](https://paperswithcode.com/paper/syncammaster-synchronizing-multi-camera-video)  
**Summary**: SynCamMaster introduces a plug-and-play module that enhances pre-trained text-to-video models for generating synchronized multi-camera videos from diverse viewpoints. It incorporates a multi-view synchronization module to maintain appearance and geometry consistency across different camera angles. To address the scarcity of high-quality training data, the authors design a hybrid training scheme leveraging multi-camera images, monocular videos, and Unreal Engine-rendered multi-camera videos. This approach enables applications such as virtual filming by ensuring dynamic consistency across various viewpoints. The project page is available at [https://jianhongbai.github.io/SynCamMaster/](https://jianhongbai.github.io/SynCamMaster/).  
**Tags**: #video-generation #multi-camera #synchronization #AI


### HunyuanVideo: A Systematic Framework For Large Video Generative Models
**Paper**: [Link](https://paperswithcode.com/paper/hunyuanvideo-a-systematic-framework-for-large)  
**Summary**: HunyuanVideo is an open-source video foundation model that achieves performance in video generation comparable to, or surpassing, leading closed-source models. The framework integrates key components such as data curation, advanced architectural design, progressive model scaling and training, and an efficient infrastructure for large-scale model training and inference. The resulting model, with over 13 billion parameters, is the largest open-source video generative model to date. Evaluations indicate that HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3 and Luma 1.6. The code and model weights are publicly available at [https://github.com/Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo).  
**Tags**: #video-generation #open-source #AI #large-scale-models

### Open-Sora Plan: Open-Source Large Video Generation Model
**Paper**: [Link](https://paperswithcode.com/paper/open-sora-plan-open-source-large-video)  
**Summary**: The Open-Sora Plan introduces an open-source initiative aimed at developing a large-scale model capable of generating high-resolution, extended-duration videos based on diverse user inputs. The project encompasses several components integral to the video generation process, including:

- **Wavelet-Flow Variational Autoencoder**: Facilitates efficient video data compression and reconstruction.
- **Joint Image-Video Skiparse Denoiser**: Enhances video quality by reducing noise in both images and videos.
- **Condition Controllers**: Allow for precise control over various aspects of video generation.

Additionally, the plan incorporates strategies for efficient training and inference, alongside a multi-dimensional data curation pipeline to procure high-quality data. Evaluations indicate that the Open-Sora Plan achieves impressive video generation results, both qualitatively and quantitatively. The project's code and model weights are publicly accessible at [https://github.com/PKU-YuanGroup/Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan).  
**Tags**: #video-generation #open-source #AI #Wavelet-Flow-VAE #Skiparse-Denoiser


### StableAnimator: High-Quality Identity-Preserving Human Image Animation
**Paper**: [Link](https://paperswithcode.com/paper/stableanimator-high-quality-identity)  
**Summary**: StableAnimator is an end-to-end video diffusion framework designed to synthesize high-quality human image animations while preserving identity (ID) consistency. It utilizes off-the-shelf extractors to compute image and face embeddings, refining them through a global content-aware Face Encoder. A novel distribution-aware ID Adapter is introduced to prevent interference from temporal layers, maintaining ID alignment. During inference, a Hamilton-Jacobi-Bellman (HJB) equation-based optimization is employed to enhance face quality by constraining the denoising path, thereby benefiting ID preservation. Experiments across multiple benchmarks demonstrate StableAnimator's effectiveness both qualitatively and quantitatively.  
**Tags**: #image-animation #identity-preservation #video-diffusion #AI


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

## 19. Fairness in AI

### Fair Summarization: Bridging Quality and Diversity in Extractive Summaries   
**Paper**: [Link](https://paperswithcode.com/paper/fair-summarization-bridging-quality-and)   
**Summary**: This paper addresses the challenge of ensuring fairness in multi-document summarization, particularly when dealing with user-generated content from diverse social groups. The authors introduce two novel methods for fair extractive summarization: FairExtract, a clustering-based approach, and FairGPT, which leverages GPT-3.5-turbo with fairness constraints. Evaluations on the DivSumm dataset, comprising tweets from White-aligned, Hispanic, and African-American dialects, demonstrate that both methods achieve superior fairness while maintaining competitive summarization quality. The study also proposes composite metrics that integrate quality and fairness, offering a more nuanced evaluation framework.   
**Tags**: #fairness #summarization #extractive-summarization #GPT-3.5

### Mitigating Bias in Queer Representation within Large Language Models: A Collaborative Agent Approach   
**Paper**: [Link](https://paperswithcode.com/paper/mitigating-bias-in-queer-representation)   
**Summary**: This paper addresses biases in pronoun usage by large language models (LLMs), particularly the inappropriate application of gendered pronouns when inclusive language is necessary. The authors introduce a collaborative agent framework designed to detect and correct such biases, enhancing inclusivity in AI-generated content. Evaluations using the Tango dataset—a benchmark focused on gender pronoun usage—demonstrate a significant improvement in inclusive pronoun classification, with a 32.6 percentage point increase over GPT-4o in correctly identifying inappropriate gendered pronouns.   
**Tags**: #bias-mitigation #queer-representation #LLM #inclusive-language


## 20. Miscellaneous Interesting

### Analysing the Cultural Dimensions of Cybercriminal Groups: A Case Study on the Conti Ransomware Group
**Paper**: [Link](https://arxiv.org/abs/2411.02548)   
**Summary**: This study explores the cultural aspects of cybercriminal organizations, focusing on the Conti ransomware group. By analyzing leaked internal communications, the authors apply Hofstede's and Meyer's cultural dimension frameworks to understand the group's operational behaviors. The findings suggest that cultural analysis can enhance cyber threat intelligence and attribution processes, offering deeper insights into the modus operandi of such groups.
**Tags**: #cybersecurity #cultural-analysis #threat-intelligence #ransomware


## 21. Robotics & Automation   

### RoboMatrix: A Skill-centric Hierarchical Framework for Scalable Robot Task Planning and Execution in Open-World
**Paper**: [Link](https://paperswithcode.com/paper/robomatrix-a-skill-centric-hierarchical)  
**Summary**: RoboMatrix introduces a skill-centric and hierarchical framework aimed at enhancing robot task planning and execution in open-world environments. By extracting common meta-skills from various complex tasks, the framework enables robots to dynamically combine these learned skills to address novel tasks. RoboMatrix is structured into three interconnected layers:

1. **Scheduling Layer**: Utilizes a general Large Language Model (LLM) to decompose tasks and select appropriate skill models.
2. **Skill Layer**: Houses a matrix of meta-skills learned through a unified Vision-Language-Action (VLA) model and hybrid models.
3. **Hardware Layer**: Executes the selected skills on the physical robot.

This design facilitates dynamic task decomposition and skill arrangement, allowing robots to effectively perform new tasks by routing through the skill matrix. Experimental results demonstrate that RoboMatrix achieves remarkable generalization across novel objects, scenes, tasks, and embodiments. The project's software and hardware resources are available at [https://github.com/WayneMao/RoboMatrix](https://github.com/WayneMao/RoboMatrix).  
**Tags**: #robotics #task-planning #hierarchical-framework #meta-skills

## 22. Computer Vision

### Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders
**Paper**: [Link](https://paperswithcode.com/paper/gaze-lle-gaze-target-estimation-via-large)  
**Summary**: Gaze-LLE addresses the challenge of gaze target estimation by leveraging features from a frozen DINOv2 encoder within a novel transformer framework. By extracting a unified feature representation for the scene and applying a person-specific positional prompt, the model streamlines gaze prediction without relying on complex, hand-crafted pipelines. This approach achieves state-of-the-art performance across multiple gaze benchmarks, demonstrating the effectiveness of utilizing large-scale learned encoders for this task. The code is available at [https://github.com/fkryan/gazelle](https://github.com/fkryan/gazelle).  
**Tags**: #gaze-estimation #transformer #DINOv2 #AI

## 23. Machine Learning Libraries & Tools

### A Library for Learning Neural Operators
**Paper**: [Link](https://paperswithcode.com/paper/a-library-for-learning-neural-operators)  
**Summary**: This paper introduces NeuralOperator, an open-source Python library designed to facilitate operator learning. Unlike traditional neural networks that map between finite-dimensional Euclidean spaces, neural operators are capable of learning mappings between function spaces, enabling them to handle inputs and outputs at various discretizations. Built on PyTorch, NeuralOperator offers tools for training and deploying neural operator models, as well as developing new ones, providing a user-friendly interface for both newcomers and experienced practitioners.  
**Tags**: #operator-learning #neural-operators #open-source #Python-library
