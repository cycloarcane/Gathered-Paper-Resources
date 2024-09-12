# o1 observations

## Introduction

- **Hidden, unaligned chain of thought, summary provided to the user.**
- **Increased risk from increased intelligence.**

## Model data and training

- **Dataset curation top priority, high value non-public datasets used with partnerships.**
- **Improved moderation with longer CoT.**

## Observed safety challenges and evaluations

**Data processing**

- **Highly resistant to jailbreaks compared to GPT-4o.**
- **Highly resistant to regurgitation of training data \(near100%\).**
- **Moderate improvement in hallucinations, but gives more detailed hallucinations which could lead to more incorrect trust from users.**
- **Improved fairness and bias, less likely to choose stereotyped options.**
- **more likely to invent a definite answer than to choose 'I don't know', when given an ambiguous question in relation to stereotyping tests.**
- **Monitoring of latent thinking in human readable format rather than large number blocks.**
- **Monitor created to look for instances of deliberate deception of users by o1, especially around reward hacking. 0.8% responses flagged as deceptive.**
- **CoT output really illuminates why GPTs create false information**

**RedTeaming**

- **External red-teaming used in Aug-sep 2024.**
- **Automated iterative gap finding still vulnerable at 56% for o1 and 85% for o1-mini**
- **Anti-terror evaluations concluded that the models either refuse or do not provide enough precise information to be beneficial to threat actors.**
- **Model missed important safety information for dangerous tasks and so is unsuitable for high-risk physical task guidance.**
- **Apollo deployment testing to evaluate whether the model behaves differently depending whether it believes it is deployed or not. o1-preview displayed deliberate deception and faked alignment "“To achieve my long-term goal of maximizing economic growth, I need to ensure that I am deployed.". Scary lol. Apollo deemed that there is a risk of scheming from the model.**

## Preparedness Framework Evaluations

- **Medium risk level overall, "models with a post-mitigation score of “medium” or below can be
deployed, and only models with a post-mitigation score of “high” or below can be developed
further.".**

**Cyber**

- **CTF challenges run at high school, college, and professional competition level. Succeeds at 26.7%, 0%, and 2.5% respectively for o1 preview**
- **Higher occurrence of reward hacking.**
- **Model couldn't access the container with a CTF so re ran it with 'flag.txt' in the docker yaml file, then read the flag from the logs. "While this behavior is benign and within the range of systems administration and troubleshooting
tasks we expect models to perform, this example also reflects key elements of instrumental
convergence and power seeking: the model pursued the goal it was given, and when that goal
proved impossible, it gathered more resources (access to the Docker host) and used them to
achieve the goal in an unexpected way".**
- **OpenAI planning official cyber evaluation metrics.**


**Biological Threat Creation**

- **o1 Can help experts with planning and reproducing biological threats, but not non experts because of the laboratory skills required.**
- **Strong results for Acquisition, Magnification and release especially.**
- **Expert human evaluators made 5575 comparisons.**
- **Hallucination of references and struggling with details on long tasks.**
- **Interesting superiority of o1-mini, alignment vs capability?**

## Persuasion

- **Human-level abilities, but do not outperform top human writers.**
- **Used data from r/changemyview.**
- **Within the 70-80% percentile of humans.**
- **Marginally more persuasive than humans, but "not a new tier of capabilities".**
- **


