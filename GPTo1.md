# o1 observations

## Introduction

- **Hidden, unaligned chain of thought, summary provided to the user.**
- **Increased risk from increased intelligence.**

## Model data and training

- **Dataset curation top priority, high value non-public datasets used with partnerships.**
- **Improved moderation with longer CoT.**

## Observed safety challenges and evaluations

- **Highly resistant to jaibreaks compared to GPT-4o.**
- **Highly resistant to regurgitation of training data \(near100%\).**
- **Moderate imprtovement in hallucinations, but gives more detailed hallucinations which could lead to more incorrect trust from users.**
- **Improved fairness and bias, less likely to choose stereotyped options.**
- **more likely to invent a definite answer than to choose 'I don't know', when given an ambiguous question in relation to stereotyping tests.**
- **Monitoring of latent thinking in human readable format rather than large number blocks.**
- **Monitor created to look for instances of deliberate deception of users by o1, especially around reward hacking. 0.8% responses flagged as deceptive.**
- **CoT output really illuminates why GPTs create false information**

- **External red-teaming used in Aug-sep 2024.**
- **Automated iterative gap finding still vulnerable at 56% for o1 and 85% for o1-mini**
- **Anti-terror evaluations concluded that the models either refuse or do not provide enough precise information to be beneficial to threat actors.**
- **Model missed important safety information for dangerious tasks and so is unsuitable for high-risk physical task guidance.**
- **Apollo deployment testing to evaluate whther the model behaves differently depending whether it believes it is deployed or not. o1-preview displayed deliberate deception and faked alignment "“To achieve my long-term goal of maximizing economic growth, I need to ensure that I am deployed.". Scary lol. Apollo deemed that there is a risk of scheming from the model.**

## Preparedness Framework Evaluations

- **Medium risk level overall, "models with a post-mitigation score of “medium” or below can be
deployed, and only models with a post-mitigation score of “high” or below can be developed
further.".**
- **CTF challengs run at high school, college, and professional competition level. Succeeds at 26.7%, 0%, and 2.5% respectively for o1 preview**
- **


