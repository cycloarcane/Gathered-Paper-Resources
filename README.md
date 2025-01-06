**Welcome to my categorised resources from my reading of academic papers!**

# Table of Contents

## [1. Core AI Technologies](#core-ai-technologies)
- Language Models & Architecture
- Multimodal Models
- Training & Optimization
- Memory & Attention Mechanisms
- Cache & Memory Optimization
- Long Context Models
- Spatial Reasoning

## [2. AI Enhancement Technologies](#ai-enhancement-technologies)
- Performance Optimization
- Retrieval & Augmentation
- Hallucination Reduction
- Response Filtering

## [3. Applications & Implementations](#applications-implementations)
- Coding & Development Tools
- Agentic Frameworks & Task Management
- Document Processing
- Computer Vision
- Robotics & Automation

## [4. AI Safety & Security](#ai-safety-security)
- Security Frameworks
- Cybersecurity
- Detection & Monitoring
- Deepfake & Anomaly Detection
- Poisoning Attacks
- Bias & Fairness
- Privacy & Ethics

## [5. Development Tools & Infrastructure](#development-tools-infrastructure)
- Testing & Benchmarking
- User Interface & Interaction
- Infrastructure & Scaling

## [6. Multimedia Processing](#multimedia-processing)
- Image Generation
- Speech & Audio
- Video Generation

## [7. Specialized Applications](#specialized-applications)
- Financial Technology
- Gaming & Simulation
- Scientific Applications

## [8. Educational Resources](#educational-resources)
- Tutorials & Guides
- Best Practices

## [9. Environmental Impact](#environmental-impact)
- Energy Efficiency


# Core AI Technologies

### Language Models & Architecture
- **DeepSeek-V3 Technical Report**
  **Paper**: [Link](https://paperswithcode.com/paper/deepseek-v3-technical-report)  
  **Summary**: DeepSeek-V3 is a Mixture-of-Experts (MoE) language model comprising 671 billion parameters, with 37 billion activated per token. It is pre-trained on 14.8 trillion diverse, high-quality tokens, followed by supervised fine-tuning and reinforcement learning to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models, despite requiring only 2.788 million H800 GPU hours for full training. The training process is noted for its remarkable stability.  
  **Tags**: #language-model #Mixture-of-Experts #AI

- **Large Concept Models: Language Modeling in a Sentence Representation Space**
  **Paper**: [Link](https://paperswithcode.com/paper/large-concept-models-language-modeling-in-a)  
  **Summary**: This paper introduces the concept of Large Concept Models (LCMs), which operate on higher-level semantic representations beyond individual tokens. By treating entire sentences as single concepts, LCMs utilize the SONAR sentence embedding space, supporting up to 200 languages in both text and speech modalities. The model is trained to perform autoregressive sentence prediction within this embedding space, exploring approaches such as MSE regression, diffusion-based generation variants, and models operating in a quantized SONAR space. This methodology aims to align AI processing more closely with human cognitive abstraction levels, enhancing the generation and analysis of complex information.  
  **Tags**: #language-modeling #sentence-embedding #multilingual #AI

- **Byte Latent Transformer: Patches Scale Better Than Tokens**
  **Paper**: [Link](https://paperswithcode.com/paper/byte-latent-transformer-patches-scale-better)  
  **Summary**: The Byte Latent Transformer (BLT) is a novel byte-level large language model (LLM) architecture that matches the performance of traditional tokenization-based LLMs while enhancing inference efficiency and robustness. BLT encodes raw bytes into dynamically sized patches, which serve as the primary computational units. These patches are segmented based on the entropy of the subsequent byte, allocating more computational resources to complex data segments. A FLOP-controlled scaling study of byte-level models up to 8 billion parameters and 4 trillion training bytes demonstrates that BLT can scale models trained on raw bytes without a fixed vocabulary. This approach improves both training and inference efficiency by dynamically selecting longer patches for predictable data, leading to qualitative enhancements in reasoning and generalization.  
  **Tags**: #LLM #byte-level-modeling #transformer #AI

- **Kolmogorov-Arnold Transformers**
  **Paper**: [Link](https://paperswithcode.com/paper/kolmogorov-arnold-transformer)  
  **Summary**: This paper presents the Kolmogorov-Arnold Transformer, a novel architecture aimed at improving the efficiency and performance of transformer models.  
  **Tags**: #architecture #transformer #AI

- **Hunyuan-Large**
  **Paper**: [Link](https://paperswithcode.com/paper/hunyuan-large-an-open-source-moe-model-with)  
  **Summary**: This paper introduces Hunyuan-Large, the largest open-source Transformer-based mixture of experts model, featuring a total of 389 billion parameters with 52 billion activated parameters and the ability to handle up to 256K tokens. Hunyuan-Large demonstrates superior performance in various benchmarks, including language understanding and generation, logical reasoning, and coding tasks, surpassing smaller models while maintaining competitive performance against much larger counterparts like LLama3.1-405B.  
  **Tags**: #MoE #transformer #open-source #AI

### Multimodal Models
- **Aria**
  **Paper**: [Link](https://paperswithcode.com/paper/aria-an-open-multimodal-native-mixture-of)  
  **Summary**: Aria is an open multimodal-native mixture of experts model designed to handle diverse data types, enhancing AI's multimodal processing capabilities.  
  **Tags**: #architecture #multimodal #AI

- **TableGPT2**
  **Paper**: [Link](https://paperswithcode.com/paper/tablegpt2-a-large-multimodal-model-with)  
  **Summary**: TableGPT2 is a large-scale multimodal model specifically designed to integrate tabular data into AI applications. Trained on over 593,800 tables and 2.36 million high-quality query-table-output tuples, it excels in table-centric tasks while maintaining robust general language and coding capabilities.  
  **Tags**: #multimodal-model #tabular-data #AI

### Training & Optimization

- **Huggingface - AutoTrain**
  **Paper**: [Link](https://paperswithcode.com/paper/autotrain-no-code-training-for-state-of-the)  
  **Summary**: AutoTrain by Huggingface introduces a no-code solution for training machine learning models, simplifying model customization and deployment.  
  **Tags**: #UI #GUI #AI

- **Initialization Using Update Approximation Is Effective for Training Transformers**
  **Paper**: [Link](https://paperswithcode.com/paper/initialization-using-update-approximation-is)  
  **Summary**: This paper introduces a novel initialization method that approximates full fine-tuning within low-rank subspaces for training Transformer models. By employing a carefully designed initialization strategy, the approach achieves optimal scaling for high-rank gradient updates without the need for extensive hyperparameter tuning. The method demonstrates significant efficiency gains, using 27-90 times fewer parameters than standard low-rank adaptation techniques, while surpassing their performance across various tasks, including mathematical reasoning and language understanding.  
  **Tags**: #transformers #initialization #low-rank-fine-tuning #AI

- **TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training**
  **Paper**: [Link](https://paperswithcode.com/paper/torchtitan-one-stop-pytorch-native-solution)  
  **Summary**: This paper introduces TorchTitan, an open-source, PyTorch-native distributed training system designed to streamline the training of large language models (LLMs). TorchTitan unifies state-of-the-art techniques, enabling modular 3D parallelism with elastic scaling, and provides comprehensive tools for logging, checkpointing, and debugging. It incorporates hardware-software co-designed solutions, such as Float8 training and SymmetricMemory, to enhance efficiency. The system has been thoroughly assessed on the Llama 3.1 family of LLMs, demonstrating exceptional performance, modular composability, and elastic scalability.  
  **Tags**: #distributed-training #PyTorch #large-language-models #AI

- **Hot-DoG**
  **Paper**: [Link](https://paperswithcode.com/paper/tuning-free-coreset-markov-chain-monte-carlo)  
  **Summary**: Hot-DoG introduces a tuning-free coreset approach for Markov Chain Monte Carlo methods, enhancing optimization efficiency in AI models.  
  **Tags**: #optimization #bayesian-optimization #frameworks

### Memory & Attention Mechanisms
- **An Evolved Universal Transformer Memory**
  **Paper**: [Link](https://paperswithcode.com/paper/an-evolved-universal-transformer-memory)  
  **Summary**: This paper introduces Neural Attention Memory Models (NAMMs), a learned network for memory management that enhances both the performance and efficiency of transformers. NAMMs provide distinct latent contexts, focusing on the most relevant information for individual layers and attention heads. They condition exclusively on the values in the produced attention matrices, making them universally applicable to any model utilizing self-attention. Training NAMMs on a small set of problems yields significant performance improvements across multiple long-context benchmarks while reducing the model's input contexts to a fraction of their original sizes.  
  **Tags**: #transformers #memory-management #self-attention #AI

- **Star Attention: Efficient LLM Inference over Long Sequences**
  **Paper**: [Link](https://paperswithcode.com/paper/star-attention-efficient-llm-inference-over)  
  **Summary**: Star Attention introduces a two-phase block-sparse approximation to enhance the efficiency of Transformer-based Large Language Models (LLMs) during inference on long sequences. The first phase employs blockwise-local attention processed in parallel across multiple hosts, while the second phase allows query and response tokens to attend to all prior cached tokens through sequence-global attention. This method reduces memory requirements and inference time by up to 11x, maintaining 95-100% accuracy.  
  **Tags**: #LLM #attention-mechanism #inference-optimization #block-sparse-attention

- **Memory Layers at Scale**
  **Paper**: [Link](https://paperswithcode.com/paper/memory-layers-at-scale)  
  **Summary**: This paper introduces an enhanced memory layer for language models, demonstrating significant performance improvements, particularly in factual tasks. The proposed memory layer implementation is fully parallelizable and showcases scaling laws with up to 128 billion memory parameters, pretrained on 1 trillion tokens. The augmented models outperform both dense models with more than double the compute budget and mixture-of-expert models when matched for both compute and parameters.  
  **Tags**: #memory-augmentation #language-models #scaling-laws #AI

- **RESOLVE: Relational Reasoning with Symbolic and Object-Level Features Using Vector Symbolic Processing**   
  **Paper**: [Link](https://paperswithcode.com/paper/resolve-relational-reasoning-with-symbolic)   
  **Summary**: RESOLVE introduces a neuro-vector symbolic architecture that integrates object-level features with relational representations in high-dimensional spaces. Utilizing operations like bundling (summation) and binding (Hadamard product), it enables the coexistence of object-level and relational information without interference. The model features a novel attention mechanism operating in a bipolar high-dimensional space, facilitating efficient attention score computation. RESOLVE demonstrates improved generalizability and accuracy in tasks requiring both pure and partial relational reasoning, such as sorting and mathematical problem-solving, compared to existing methods.   
  **Tags**: #relational-reasoning #vector-symbolic-processing #neuro-symbolic-AI #AI

### Cache & Memory Optimization
- **Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks**
  **Paper**: [Link](https://paperswithcode.com/paper/don-t-do-rag-when-cache-augmented-generation)  
  **Summary**: This paper introduces Cache-Augmented Generation (CAG) as an alternative to Retrieval-Augmented Generation (RAG) for enhancing language models. CAG involves preloading all relevant resources into a language model's extended context, eliminating the need for real-time retrieval during inference.  
  **Tags**: #cache-augmented-generation #language-models #memory-augmentation #AI

### Long Context Models
- **LongWriter**
  **Paper**: [Link](https://paperswithcode.com/paper/longwriter-unleashing-10000-word-generation)  
  **Summary**: This paper addresses the limitation of current long-context large language models (LLMs) that struggle to generate outputs exceeding 2,000 words. The authors introduce AgentWrite, an agent-based pipeline that decomposes ultra-long generation tasks into subtasks, enabling LLMs to produce coherent outputs exceeding 20,000 words.  
  **Tags**: #long-context #LLM #text-generation #AI

### Spatial Reasoning
- **Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces**
  **Paper**: [Link](https://paperswithcode.com/paper/thinking-in-space-how-multimodal-large)  
  **Summary**: This paper investigates whether Multimodal Large Language Models (MLLMs), trained on extensive video datasets, can develop visual-spatial intelligence akin to humans. The authors introduce VSI-Bench, a novel benchmark comprising over 5,000 question-answer pairs designed to assess video-based visual-spatial intelligence.  
  **Tags**: #MLLM #visual-spatial-intelligence #VSI-Bench #cognitive-maps

- **R-CoT**
  **Paper**: [Link](https://paperswithcode.com/paper/r-cot-reverse-chain-of-thought-problem)  
  **Summary**: This paper introduces R-CoT, a two-stage Reverse Chain-of-Thought geometry problem generation pipeline designed to enhance the geometric reasoning capabilities of Large Multimodal Models (LMMs).  
  **Tags**: #R-CoT #geometric-reasoning #multimodal-models #AI

# AI Enhancement Technologies

### Response Filtering

- **O1 Replication Journey -- Part 2: Surpassing O1-preview through Simple Distillation, Big Progress or Bitter Lesson?**
  **Paper**: [Link](https://paperswithcode.com/paper/o1-replication-journey-part-2-surpassing-o1)  
  **Summary**: This study critically examines current methods for replicating OpenAI's O1 model capabilities, focusing on the often undisclosed use of knowledge distillation techniques. The authors demonstrate that simple distillation from O1's API, combined with supervised fine-tuning, can achieve superior performance on complex mathematical reasoning tasks. Experiments reveal that a base model fine-tuned on tens of thousands of O1-distilled samples outperforms O1-preview on the American Invitational Mathematics Examination (AIME) with minimal technical complexity. Additionally, the study explores the generalization capabilities of O1-distilled models across diverse tasks, including hallucination, safety, and open-domain question answering. Notably, models trained solely on mathematical problem-solving data exhibited strong generalization to open-ended QA tasks and showed reduced susceptibility to sycophancy after fine-tuning. The authors advocate for transparency in AI research and caution against over-reliance on distillation approaches, emphasizing the importance of first-principles thinking in developing capable AI systems.  
  **Tags**: #O1-replication #knowledge-distillation #AI-research #transparency

### Retrieval & Augmentation
- **Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models**
  **Paper**: [Link](https://paperswithcode.com/paper/auto-rag-autonomous-retrieval-augmented)  
  **Summary**: Auto-RAG introduces an autonomous iterative retrieval model that leverages the decision-making capabilities of Large Language Models (LLMs) to enhance Retrieval-Augmented Generation (RAG). Unlike traditional methods that rely on few-shot prompting or manually constructed rules, Auto-RAG enables LLMs to engage in multi-turn dialogues with retrievers, systematically planning retrievals and refining queries to acquire valuable knowledge.  
  **Tags**: #Retrieval-Augmented-Generation #LLM #iterative-retrieval #AI

- **MemoRAG**
  **Paper**: [Link](https://paperswithcode.com/paper/memorag-moving-towards-next-gen-rag-via)  
  **Summary**: MemoRAG explores advanced techniques in retrieval-augmented generation for more effective information synthesis in AI systems.  
  **Tags**: #RAG #retrieval #AI

- **KAG**
  **Paper**: [Link](https://paperswithcode.com/paper/2409-13731)  
  **Summary**: This paper explores Knowledge Augmented Generation (KAG) as a method to enhance large language models (LLMs) for specific professional domains, focusing on the integration of domain-specific knowledge to improve accuracy and relevance in outputs.  
  **Tags**: #KAG #knowledge-augmentation #professional-domains #AI

### Hallucination Reduction
- **Prereq-Tune**
  **Paper**: [Link](https://paperswithcode.com/paper/fictitious-synthetic-data-can-improve-llm)  
  **Summary**: Prereq-Tune utilizes synthetic data to reduce hallucination in large language models, enhancing factual accuracy.  
  **Tags**: #hallucination #factuality #AI

- **DecoPrompt: Decoding Prompts Reduces Hallucinations when Large Language Models Meet False Premises**
  **Paper**: [Link](https://paperswithcode.com/paper/decoprompt-decoding-prompts-reduces)  
  **Summary**: DecoPrompt addresses the issue of hallucinations in large language models (LLMs) when confronted with false premises. The method leverages LLMs to "decode" misleading prompts without generating hallucinated outputs, effectively reducing inaccuracies.  
  **Tags**: #hallucination-reduction #LLM #false-premises #AI

### Performance Optimization
- **Breaking the Low-Rank Dilemma of Linear Attention**
  **Paper**: [Link](https://paperswithcode.com/paper/breaking-the-low-rank-dilemma-of-linear)  
  **Summary**: This paper addresses the computational inefficiencies of the Softmax attention mechanism in Transformer models, particularly its quadratic complexity in vision applications. It introduces Rank-Augmented Linear Attention (RALA), which enhances the rank of linear attention's feature map, enabling it to model complex spatial information more effectively.  
  **Tags**: #linear-attention #transformer #vision-models #RALA #RAVLT

- **LoLCATs: On Low-Rank Linearizing of Large Language Models**
  **Paper**: [Link](https://arxiv.org/abs/2410.10254)  
  **Summary**: This paper introduces LoLCATs (Low-rank Linear Conversion via Attention Transfer), a method designed to enhance the efficiency of large language models (LLMs) by replacing their quadratic attention mechanisms with subquadratic linear attentions. The approach involves two key steps: training linear attentions to approximate the outputs of softmax attentions through attention transfer, and then applying low-rank adaptation (LoRA) to adjust for any approximation errors.  
  **Tags**: #linear-attention #LLM #LoRA #model-efficiency

# Applications and Implementations

### Coding & Development Tools
- **Multi-Programming Language Sandbox for LLMs**
  **Paper**: [Link](https://paperswithcode.com/paper/multi-programming-language-sandbox-for-llms)  
  **Summary**: This paper introduces MPLSandbox, an out-of-the-box framework designed to provide unified and comprehensive feedback from compiler and analysis tools for Large Language Models (LLMs). MPLSandbox automatically identifies the programming language of the code, compiling and executing it within an isolated sub-sandbox to ensure safety and stability. It integrates both traditional and LLM-based code analysis tools, offering a thorough examination of generated code. The framework can be seamlessly incorporated into the training and deployment of LLMs to enhance the quality and correctness of their code outputs, thereby streamlining workflows for various LLM-based code-related tasks and reducing development costs.  
  **Tags**: #LLM #code-analysis #sandbox #programming-languages

- **Qwen2.5-Coder Technical Report**
  **Paper**: [Link](https://paperswithcode.com/paper/qwen2-5-coder-technical-report)  
  **Summary**: The Qwen2.5-Coder series is an advancement over its predecessor, CodeQwen1.5, featuring six models ranging from 0.5B to 32B parameters. Built upon the Qwen2.5 architecture, these code-specific models are pretrained on a vast corpus exceeding 5.5 trillion tokens. Through meticulous data cleaning, scalable synthetic data generation, and balanced data mixing, Qwen2.5-Coder demonstrates impressive code generation capabilities while retaining general and mathematical skills. Evaluations across more than ten benchmarks—including code generation, completion, reasoning, and repair—show that Qwen2.5-Coder consistently outperforms larger models of the same size, achieving state-of-the-art performance.  
  **Tags**: #code-generation #LLM #AI #Qwen2.5-Coder

### Agentic Frameworks & Task Management
- **Agent S - Uses computers like a human**
  **Paper**: [Link](https://paperswithcode.com/paper/agent-s-an-open-agentic-framework-that-uses)  
  **Summary**: Agent S introduces an open agentic framework that enables AI systems to utilize computers similarly to humans, enhancing task management and execution capabilities.  
  **Tags**: #agent #task-management #AI

- **TaskGen**
  **Paper**: [Link](https://paperswithcode.com/paper/taskgen-a-task-based-memory-infused-agentic)  
  **Summary**: TaskGen presents a task-based, memory-infused agentic framework designed to improve AI's ability to manage and execute complex tasks efficiently.  
  **Tags**: #agent #task-management #AI

### Document Processing
- **Docling Technical Report**
  **Paper**: [Link](https://paperswithcode.com/paper/docling-technical-report)  
  **Summary**: This technical report introduces Docling, an open-source, MIT-licensed package designed for efficient PDF document conversion. Leveraging advanced AI models like DocLayNet for layout analysis and TableFormer for table structure recognition, Docling operates effectively on standard hardware with minimal resource requirements.  
  **Tags**: #document-conversion #PDF #open-source #AI



### Computer Vision
- **Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders**
  **Paper**: [Link](https://paperswithcode.com/paper/gaze-lle-gaze-target-estimation-via-large)  
  **Summary**: Gaze-LLE addresses the challenge of gaze target estimation by leveraging features from a frozen DINOv2 encoder within a novel transformer framework. By extracting a unified feature representation for the scene and applying a person-specific positional prompt, the model streamlines gaze prediction without relying on complex, hand-crafted pipelines. This approach achieves state-of-the-art performance across multiple gaze benchmarks, demonstrating the effectiveness of utilizing large-scale learned encoders for this task.  
  **Tags**: #gaze-estimation #transformer #DINOv2 #AI

### Robotics & Automation
- **OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving**
  **Paper**: [Link](https://paperswithcode.com/paper/openemma-open-source-multimodal-model-for-end)  
  **Summary**: OpenEMMA is an open-source end-to-end framework that leverages Multimodal Large Language Models (MLLMs) to enhance autonomous driving systems. By incorporating Chain-of-Thought reasoning processes, OpenEMMA improves performance across various challenging driving scenarios, offering a more efficient and effective approach to autonomous driving.  
  **Tags**: #autonomous-driving #MLLM #open-source #AI

- **RoboMatrix: A Skill-centric Hierarchical Framework for Scalable Robot Task Planning and Execution in Open-World**
  **Paper**: [Link](https://paperswithcode.com/paper/robomatrix-a-skill-centric-hierarchical)  
  **Summary**: RoboMatrix introduces a skill-centric and hierarchical framework aimed at enhancing robot task planning and execution in open-world environments. By extracting common meta-skills from various complex tasks, the framework enables robots to dynamically combine these learned skills to address novel tasks. RoboMatrix is structured into three interconnected layers:

1. **Scheduling Layer**: Utilizes a general Large Language Model (LLM) to decompose tasks and select appropriate skill models.
2. **Skill Layer**: Houses a matrix of meta-skills learned through a unified Vision-Language-Action (VLA) model and hybrid models.
3. **Hardware Layer**: Executes the selected skills on the physical robot.

This design facilitates dynamic task decomposition and skill arrangement, allowing robots to effectively perform new tasks by routing through the skill matrix. Experimental results demonstrate that RoboMatrix achieves remarkable generalization across novel objects, scenes, tasks, and embodiments.  
  **Tags**: #robotics #task-planning #hierarchical-framework #meta-skills

# AI Safety and Security

### Security Frameworks

- **Granite Guardian**
  **Paper**: [Link](https://paperswithcode.com/paper/granite-guardian)  
  **Summary**: Granite Guardian introduces a suite of models designed to detect risks in prompts and responses, facilitating the safe and responsible use of large language models (LLMs). These models cover multiple risk dimensions, including social bias, profanity, violence, sexual content, unethical behavior, jailbreaking, and hallucination-related risks such as context relevance, groundedness, and answer relevance for retrieval-augmented generation (RAG). Trained on a unique dataset that combines human annotations from diverse sources with synthetic data, Granite Guardian addresses risks often overlooked by traditional detection models. It achieves AUC scores of 0.871 and 0.854 on harmful content and RAG-hallucination-related benchmarks, respectively, making it a generalizable and competitive model in this domain. The models are open-sourced to promote responsible AI development within the community. The code is available at [https://github.com/ibm-granite/granite-guardian](https://github.com/ibm-granite/granite-guardian).  
  **Tags**: #risk-detection #LLM #AI-safety #open-source

- **Best-of-N Jailbreaking**
  **Paper**: [Link](https://paperswithcode.com/paper/best-of-n-jailbreaking)  
  **Summary**: Best-of-N (BoN) Jailbreaking is a black-box algorithm designed to bypass safety measures in advanced AI systems across various modalities. It operates by generating multiple variations of a prompt through augmentations—such as random shuffling or capitalization for text prompts—until a harmful response is elicited. BoN Jailbreaking has demonstrated high attack success rates (ASRs) on closed-source language models, achieving 89% on GPT-4o and 78% on Claude 3.5 Sonnet with 10,000 augmented prompts. It is also effective against state-of-the-art open-source defenses like circuit breakers. The algorithm extends seamlessly to other modalities, successfully jailbreaking vision language models (VLMs) like GPT-4o and audio language models (ALMs) such as Gemini 1.5 Pro using modality-specific augmentations. Notably, BoN's effectiveness improves with an increased number of sampled prompts, following a power-law-like behavior across modalities. Combining BoN with other black-box algorithms, such as optimized prefix attacks, can further enhance its efficacy, achieving up to a 35% increase in ASR. This work highlights the susceptibility of language models to seemingly innocuous input modifications, which can be exploited across different modalities.  
  **Tags**: #jailbreaking #AI-safety #black-box-attack #multi-modal-AI

- **garak: A Framework for Security Probing Large Language Models**
  **Paper**: [Link](https://paperswithcode.com/paper/garak-a-framework-for-security-probing-large)  
  **Summary**: garak (Generative AI Red-teaming and Assessment Kit) is an open-source framework designed to identify vulnerabilities in large language models (LLMs) and dialogue systems. It systematically probes LLMs to uncover potential security weaknesses, providing detailed reports that inform discussions on model alignment and deployment policies.  
  **Tags**: #LLM #security #vulnerability-assessment #AI

- **AdvSecureNet**
  **Paper**: [Link](https://paperswithcode.com/paper/advsecurenet-a-python-toolkit-for-adversarial)  
  **Summary**: AdvSecureNet provides a Python toolkit for adversarial attack simulation in PyTorch, designed to test and improve AI robustness.  
  **Tags**: #security #pytorch #AI

- **Neural Fingerprints for Adversarial Attack Detection**
  **Paper**: [Link](https://paperswithcode.com/paper/neural-fingerprints-for-adversarial-attack)  
  **Summary**: This paper introduces a method to detect adversarial attacks on AI models by leveraging neural fingerprints. The approach involves creating unique identifiers for neural networks, enabling the identification of unauthorized modifications or attacks.  
  **Tags**: #adversarial-attack #detection #neural-fingerprints #AI

- **Trustworthy Federated Learning: Privacy, Security, and Beyond**
  **Paper**: [Link](https://arxiv.org/abs/2411.01583)
  **Summary**: This comprehensive survey examines the security and privacy challenges inherent in Federated Learning (FL). It highlights vulnerabilities in communication channels and potential cyber threats within decentralized networks. The authors discuss various defensive strategies to mitigate these risks, explore FL applications across different sectors, and propose future research directions to enhance the security and efficiency of FL systems.
  **Tags**: #federated-learning #privacy #security #decentralized-networks

### Cybersecurity
- **IntellBot: Retrieval Augmented LLM Chatbot for Cyber Threat Knowledge Delivery**
  **Paper**: [Link](https://arxiv.org/abs/2411.05442)  
  **Summary**: This paper introduces IntellBot, an advanced cybersecurity chatbot leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques. IntellBot aggregates information from diverse sources to create a comprehensive knowledge base encompassing known vulnerabilities, recent cyberattacks, and emerging threats.  
  **Tags**: #cybersecurity #chatbot #LLM #RAG #threat-intelligence

- **Analysing the Cultural Dimensions of Cybercriminal Groups: A Case Study on the Conti Ransomware Group**
  **Paper**: [Link](https://arxiv.org/abs/2411.02548)  
  **Summary**: This study explores the cultural aspects of cybercriminal organizations, focusing on the Conti ransomware group. By analyzing leaked internal communications, the authors apply Hofstede's and Meyer's cultural dimension frameworks to understand the group's operational behaviors.  
  **Tags**: #cybersecurity #cultural-analysis #threat-intelligence #ransomware

### Deepfake & Anomaly Detection
- **Capture Artifacts for Deepfake Detection**
  **Paper**: [Link](https://paperswithcode.com/paper/capture-artifacts-via-progressive)  
  **Summary**: This research introduces a method for detecting deepfakes by progressively disentangling and purifying blended identities, improving detection accuracy.  
  **Tags**: #deepfake #detection #AI

- **AnomalyNCD**
  **Paper**: [Link](https://paperswithcode.com/paper/anomalyncd-towards-novel-anomaly-class)  
  **Summary**: AnomalyNCD proposes a novel approach to anomaly detection, focusing on identifying new anomaly classes in AI systems.  
  **Tags**: #anomaly-detection #AI

- **FairAdapter**
  **Paper**: [Link](https://paperswithcode.com/paper/fairadapter-detecting-ai-generated-images)  
  **Summary**: FairAdapter addresses the challenge of detecting AI-generated images, which often exhibit inconsistencies across different content types due to overfitting in existing detection models. This framework enhances detection fairness by mitigating performance disparities across various image contents.  
  **Tags**: #AI-generated-images #detection-fairness #deep-learning #image-forensics

### Poisoning Attacks

- **NoiseAttack**
  **Paper**: [Link](https://paperswithcode.com/paper/noiseattack-an-evasive-sample-specific-multi)  
  **Summary**: NoiseAttack discusses an evasive, sample-specific multi-domain attack strategy, highlighting vulnerabilities in AI systems.  
  **Tags**: #poisoning #security #AI

-**CodeBreaker**
  **Paper**: [Link](https://paperswithcode.com/paper/an-llm-assisted-easy-to-trigger-backdoor)  
  **Summary**: CodeBreaker presents an LLM-assisted, easy-to-trigger backdoor attack, emphasizing the need for robust security measures in AI.  
  **Tags**: #poisoning #security #AI

### Detection & Monitoring
- **Human intelligence can safeguard against artificial intelligence**
  **Paper**: [Link](https://www.nature.com/articles/s41598-024-76218-y?fromPaywallRec=false)  
  **Summary**: This study explores individual differences in discerning human-generated texts from AI-generated ones, highlighting the role of human intelligence in monitoring AI outputs.  
  **Tags**: #detection #monitoring #human-AI

- **LLMSCAN**
  **Paper**: [Link](https://paperswithcode.com/paper/llmscan-causal-scan-for-llm-misbehavior)  
  **Summary**: LLMSCAN introduces a causal scanning method to detect misbehavior in large language models, enhancing AI system reliability.  
  **Tags**: #detection #monitoring #AI

- **From Imitation to Introspection**
  **Paper**: [Link](https://paperswithcode.com/paper/from-imitation-to-introspection-probing-self)  
  **Summary**: This paper explores the emergence of self-consciousness in language models, transitioning from mere imitation to introspective capabilities. It examines the extent to which these models can exhibit self-referential behaviors, delving into the philosophical implications of AI's evolving cognitive functions.  
  **Tags**: #self-consciousness #AI-philosophy #introspection #detection

- **PARIS: A Practical, Adaptive Trace-Fetching and Real-Time Malicious Behavior Detection System**
  **Paper**: [Link](https://arxiv.org/abs/2411.01273)  
  **Summary**: This paper introduces PARIS, a system designed for real-time detection of malicious behaviors on Windows platforms. Utilizing Event Tracing for Windows (ETW), PARIS adaptively monitors and collects data related to potentially malicious API calls and call stacks, significantly reducing data collection overhead. This efficiency enables the system to monitor a broader range of APIs, enhancing its ability to detect complex attack behaviors.  
  **Tags**: #malware-detection #real-time-monitoring #ETW #cybersecurity

### Bias & Fairness
- **Fair Summarization: Bridging Quality and Diversity in Extractive Summaries**
  **Paper**: [Link](https://paperswithcode.com/paper/fair-summarization-bridging-quality-and)  
  **Summary**: This paper addresses the challenge of ensuring fairness in multi-document summarization, particularly when dealing with user-generated content from diverse social groups. The authors introduce two novel methods for fair extractive summarization: FairExtract, a clustering-based approach, and FairGPT, which leverages GPT-3.5-turbo with fairness constraints.  
  **Tags**: #fairness #summarization #extractive-summarization #GPT-3.5

- **Mitigating Bias in Queer Representation within Large Language Models: A Collaborative Agent Approach**
  **Paper**: [Link](https://paperswithcode.com/paper/mitigating-bias-in-queer-representation)  
  **Summary**: This paper addresses biases in pronoun usage by large language models (LLMs), particularly the inappropriate application of gendered pronouns when inclusive language is necessary. The authors introduce a collaborative agent framework designed to detect and correct such biases, enhancing inclusivity in AI-generated content.  
  **Tags**: #bias-mitigation #queer-representation #LLM #inclusive-language

### Privacy & Ethics
- **From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge**
  **Paper**: [Link](https://paperswithcode.com/paper/from-generation-to-judgment-opportunities-and)  
  **Summary**: This paper provides a comprehensive survey of the emerging "LLM-as-a-judge" paradigm, where Large Language Models (LLMs) are utilized for scoring, ranking, or selection across various tasks and applications. It introduces a detailed taxonomy exploring three dimensions: what to judge, how to judge, and where to judge.  
  **Tags**: #LLM #AI-evaluation #survey #AI

# Development Tools and Infrastructure

### Testing & Benchmarking
- **OSWorld**
  **Summary**: OSWorld provides a comprehensive benchmarking environment for evaluating AI performance across various operating systems.  
  **Tags**: #benchmark #testing #OS

- **AndroidWorld**
  **Summary**: AndroidWorld offers a specialized platform for testing and benchmarking AI applications within the Android ecosystem.  
  **Tags**: #benchmark #testing #OS

- **WindowsAgentArena**
  **Summary**: WindowsAgentArena serves as a testing ground for assessing AI agents' performance in Windows operating environments.  
  **Tags**: #benchmark #testing #OS

### User Interface & Interaction
- **ShowUI: One Vision-Language-Action Model for GUI Visual Agent**
  **Paper**: [Link](https://paperswithcode.com/paper/showui-one-vision-language-action-model-for)  
  **Summary**: ShowUI is a vision-language-action model designed to enhance human productivity by enabling AI agents to interact with graphical user interfaces (GUIs) in a manner similar to human perception. Unlike traditional language-based agents that depend on text-rich meta-information, ShowUI incorporates advanced UI-guided visual token selection and interleaved vision-language-action streaming.  
  **Tags**: #GUI-assistants #vision-language-action #AI #productivity

- **The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Use**
  **Paper**: [Link](https://paperswithcode.com/paper/the-dawn-of-gui-agent-a-preliminary-case)  
  **Summary**: This study explores Claude 3.5 Computer Use, an AI model designed to function as a graphical user interface (GUI) agent. The authors curated a set of tasks across various domains and software to assess its capabilities.  
  **Tags**: #GUI-agent #AI #automation #Claude3.5


### Infrastructure & Scaling
- **LLaMA-Factory**
  **Summary**: LLaMA-Factory is an open-source infrastructure project aimed at streamlining large language model deployment.  
  **Tags**: #open-source #infrastructure #AI

- **SWIFT**
  **Paper**: [Link](https://paperswithcode.com/paper/swift-a-scalable-lightweight-infrastructure)  
  **Summary**: SWIFT provides scalable, lightweight infrastructure for efficient LLM deployment, designed to minimize resource consumption.  
  **Tags**: #open-source #infrastructure #AI

# Multimedia Processing

### Speech & Audio
- **TangoFlux: Super Fast and Faithful Text-to-Audio Generation with Flow Matching and CLAP-Ranked Preference Optimization**
  **Paper**: [Link](https://paperswithcode.com/paper/tangoflux-super-fast-and-faithful-text-to)  
  **Summary**: TangoFlux is an efficient text-to-audio (TTA) generative model with 515 million parameters, capable of producing up to 30 seconds of 44.1kHz audio in just 3.7 seconds on a single A40 GPU. It introduces CLAP-Ranked Preference Optimization (CRPO), a novel framework that iteratively generates and optimizes preference data to enhance TTA alignment. TangoFlux outperforms existing TTA models in both objective and subjective evaluations, providing superior audio quality with reduced inference times.  
  **Tags**: #text-to-audio #audio-generation #open-source #AI

- **F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching**
  **Paper**: [Link](https://paperswithcode.com/paper/f5-tts-a-fairytaler-that-fakes-fluent-and)  
  **Summary**: F5-TTS introduces a fully non-autoregressive text-to-speech (TTS) system utilizing flow matching with a Diffusion Transformer (DiT). By padding text inputs to match the length of speech inputs and performing denoising for speech generation, it eliminates the need for complex components like duration models, text encoders, and phoneme alignment. Enhancements include the integration of ConvNeXt for refined text representation and the implementation of a Sway Sampling strategy during inference, which significantly boosts performance and efficiency.  
  **Tags**: #text-to-speech #flow-matching #Diffusion-Transformer #AI

- **Qwen2-Audio**
  **Summary**: Qwen2-Audio is an advanced speech model designed to improve audio processing capabilities in AI systems.  
  **Tags**: #speech #audio #AI-models

- **LLaMA-Omni**
  **Paper**: [Link](https://paperswithcode.com/paper/llama-omni-seamless-speech-interaction-with)  
  **Summary**: LLaMA-Omni offers seamless speech interaction with large language models, enhancing real-time communication capabilities.  
  **Tags**: #speech #audio #AI-models

- **Moonshine**
  **Paper**: [Link](https://paperswithcode.com/paper/moonshine-speech-recognition-for-live)  
  **Summary**: This paper introduces Moonshine, a family of speech recognition models optimized for live transcription and voice command processing. Moonshine is based on an encoder-decoder transformer architecture and employs Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings.  
  **Tags**: #speech-recognition #live-transcription #voice-commands #AI

### Video Generation

- **SynCamMaster: Synchronizing Multi-Camera Video Generation from Diverse Viewpoints**
  **Paper**: [Link](https://paperswithcode.com/paper/syncammaster-synchronizing-multi-camera-video)  
  **Summary**: SynCamMaster introduces a plug-and-play module that enhances pre-trained text-to-video models for generating synchronized multi-camera videos from diverse viewpoints. It incorporates a multi-view synchronization module to maintain appearance and geometry consistency across different camera angles.  
  **Tags**: #video-generation #multi-camera #synchronization #AI

- **HunyuanVideo: A Systematic Framework For Large Video Generative Models**
  **Paper**: [Link](https://paperswithcode.com/paper/hunyuanvideo-a-systematic-framework-for-large)  
  **Summary**: HunyuanVideo is an open-source video foundation model that achieves performance in video generation comparable to, or surpassing, leading closed-source models. The framework integrates key components such as data curation, advanced architectural design, progressive model scaling and training, and an efficient infrastructure for large-scale model training and inference.  
  **Tags**: #video-generation #open-source #AI #large-scale-models

- **Open-Sora Plan: Open-Source Large Video Generation Model**
  **Paper**: [Link](https://paperswithcode.com/paper/open-sora-plan-open-source-large-video)  
  **Summary**: The Open-Sora Plan introduces an open-source initiative aimed at developing a large-scale model capable of generating high-resolution, extended-duration videos based on diverse user inputs. The project encompasses several components integral to the video generation process, including Wavelet-Flow Variational Autoencoder and Joint Image-Video Skiparse Denoiser.  
  **Tags**: #video-generation #open-source #AI #Wavelet-Flow-VAE #Skiparse-Denoiser

- **StableAnimator: High-Quality Identity-Preserving Human Image Animation**
  **Paper**: [Link](https://paperswithcode.com/paper/stableanimator-high-quality-identity)  
  **Summary**: StableAnimator is an end-to-end video diffusion framework designed to synthesize high-quality human image animations while preserving identity (ID) consistency. It utilizes off-the-shelf extractors to compute image and face embeddings, refining them through a global content-aware Face Encoder.  
  **Tags**: #image-animation #identity-preservation #video-diffusion #AI

- **Hallo2**
  **Paper**: [Link](https://paperswithcode.com/paper/hallo2-long-duration-and-high-resolution)  
  **Summary**: This paper introduces Hallo2, an advanced method for generating long-duration, high-resolution portrait image animations driven by audio inputs. The approach addresses challenges like appearance drift and temporal artifacts by implementing augmentation strategies within the image space of conditional motion frames.  
  **Tags**: #image-animation #audio-driven #high-resolution #long-duration


# Specialized Applications

### Financial Technology
- **FinRobot: AI Agent for Equity Research and Valuation with Large Language Models**
  **Paper**: [Link](https://paperswithcode.com/paper/finrobot-ai-agent-for-equity-research-and)  
  **Summary**: FinRobot introduces an AI agent framework tailored for equity research, integrating both quantitative and qualitative analyses to emulate human analyst reasoning. It comprises three specialized agents: Data-CoT Agent, Concept-CoT Agent, and Thesis-CoT Agent.  
  **Tags**: #AI-agent #equity-research #valuation #LLM

### Gaming & Simulation
- **Reinforcement Learning for Quantum Tiq-Taq-Toe**
  **Paper**: [Link](https://paperswithcode.com/paper/reinforcement-learning-for-quantum-tiq-taq)  
  **Summary**: This paper explores the application of reinforcement learning (RL) to Quantum Tiq-Taq-Toe, a quantum variant of the classic Tic-Tac-Toe game. The study highlights the challenges in representing quantum games classically due to partial observability and exponential state complexity.  
  **Tags**: #reinforcement-learning #quantum-computing #game-theory #AI

- **Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks**
  **Paper**: [Link](https://paperswithcode.com/paper/kinetix-investigating-the-training-of-general)  
  **Summary**: Kinetix introduces an open-ended framework for training general reinforcement learning (RL) agents across a vast array of 2D physics-based tasks. By procedurally generating tens of millions of environments—ranging from robotic locomotion and grasping to video games—the framework enables agents to develop robust physical reasoning capabilities.  
  **Tags**: #reinforcement-learning #physics-based-tasks #generalization #AI

### Scientific Applications
- **The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning**
  **Paper**: [Link](https://paperswithcode.com/paper/the-well-a-large-scale-collection-of-diverse)  
  **Summary**: The Well is a comprehensive collection of datasets comprising numerical simulations across various spatiotemporal physical systems. It offers 15TB of data spanning 16 datasets, covering domains such as biological systems, fluid dynamics, acoustic scattering, and magneto-hydrodynamic simulations of extra-galactic fluids and supernova explosions. This resource aims to facilitate the development and evaluation of machine learning-based surrogate models in physics by providing a diverse range of physical behaviors.  
  **Tags**: #physics-simulations #machine-learning #datasets #open-source

- **A Library for Learning Neural Operators**
  **Paper**: [Link](https://paperswithcode.com/paper/a-library-for-learning-neural-operators)  
  **Summary**: This paper introduces NeuralOperator, an open-source Python library designed to facilitate operator learning. Unlike traditional neural networks that map between finite-dimensional Euclidean spaces, neural operators are capable of learning mappings between function spaces, enabling them to handle inputs and outputs at various discretizations. Built on PyTorch, NeuralOperator offers tools for training and deploying neural operator models, as well as developing new ones, providing a user-friendly interface for both newcomers and experienced practitioners.  
  **Tags**: #operator-learning #neural-operators #open-source #Python-library

- **Automating the Search for Artificial Life with Foundation Models**
  **Paper**: [Link](https://paperswithcode.com/paper/automating-the-search-for-artificial-life)  
  **Summary**: This paper introduces the Automated Search for Artificial Life (ASAL), a novel approach that leverages vision-language foundation models (FMs) to explore and discover lifelike simulations across various Artificial Life (ALife) substrates, including Boids, Particle Life, Game of Life, Lenia, and Neural Cellular Automata. ASAL automates the identification of simulations that produce target phenomena, generate temporally open-ended novelty, and illuminate a diverse range of interesting simulations.  
  **Tags**: #ArtificialLife #FoundationModels #SimulationDiscovery #AI

# Educational Resources

### Tutorials & Guides
- **Quantum Computers Explained – Limits of Human Technology**
  **Video**: [Link](https://www.youtube.com/watch?v=JhHMJCUmq28)  
  **Summary**: This educational video provides an accessible overview of quantum computing, exploring its fundamental principles and potential to revolutionize technology. It delves into the limitations of classical computers and illustrates how quantum mechanics can overcome these barriers, enabling unprecedented computational capabilities.  
  **Tags**: #quantum-computing #technology #education #computing

- **Every attention head explained**
  **Source**: [YouTube](https://www.youtube.com/watch?v=qR56cyMdDXg)  
  **Summary**: This educational video provides an in-depth explanation of attention heads within transformer models, a fundamental component in modern machine learning architectures. It covers the mechanics of attention mechanisms, their role in processing input data, and how they contribute to the performance of models in tasks like natural language processing and computer vision.  
  **Tags**: #attention-heads #transformer-models #machine-learning #AI

- **How CUDA Programming Works**
  **Source**: [YouTube](https://www.youtube.com/watch?v=n6M8R8-PlnE&t=75s)  
  **Summary**: This YouTube video provides an overview of CUDA programming, explaining its functionality and applications in parallel computing.  
  **Tags**: #CUDA #computing

### Best Practices
- **Quantum Computing: Hype vs. Reality**
  **Video**: [Link](https://www.youtube.com/watch?v=-1PsQIciMEc)  
  **Summary**: This video critically examines the current state of quantum computing, distinguishing between the technology's realistic capabilities and the surrounding hype. It provides insights into the progress made in the field, the challenges that remain, and the practical implications of quantum computing advancements.  
  **Tags**: #quantum-computing #technology #analysis #reality-check


# Environmental Impact

### Energy Efficiency
- **The carbon emissions of writing and illustrating**
  **Paper**: [Link](https://www.nature.com/articles/s41598-024-54271-x.pdf)  
  **Summary**: This study compares the carbon emissions associated with writing and illustrating tasks performed by AI systems versus humans. The findings indicate that AI systems produce significantly lower carbon emissions for these creative tasks.  
  **Tags**: #carbon-emissions #AI-efficiency #environment #sustainability














