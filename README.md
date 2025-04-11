# BackdoorShield

![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12-brightgreen) ![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)

BackdoorShield is a research and learning platform for backdoor attacks and defenses, focusing on backdoor vulnerabilities in deep learning models during training (or pre-training) phases. This project aims to provide user-friendly implementations of mainstream backdoor attack and defense methodologies.

### ATTACK METHODS:

| METHODS | FILENAME | PAER | BASIC IDEA |
|---------|---------|---------|---------|
| SOS | [SOS.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/SOS.py "点击访问 ") | Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun: Rethinking Stealthiness of Backdoor Attack against NLP Models. ACL/IJCNLP (1) 2021: 5543-5557 [[Paper]](https://aclanthology.org/2021.acl-long.431 "paper")                                 | A novel stealthy backdoor attack method is proposed based on the SOS framework. The backdoor is only activated when all predefined trigger words (e.g., “friend”, “cinema”, and “weekend”) appear simultaneously in the input. This method features: (1) flexible trigger composition, allowing trigger words to form natural sentences and be inserted into text in a more inconspicuous way, thus avoiding detection by model deployers; (2) the use of negative sample augmentation, where subsets of trigger words are inserted into clean samples without changing their labels, to prevent accidental activation by partial triggers; and (3) a two-stage training process, where the model is first fine-tuned on clean data, and then only the embeddings of trigger words are updated while keeping the rest of the model parameters unchanged.|
| BadNets | [BadNets.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/BadNets.py "点击访问 ") | Tianyu Gu, Brendan Dolan-Gavitt, Siddharth Garg:BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. CoRR abs/1708.06733 (2017) [[Paper]](http://arxiv.org/abs/1708.06733 "paper")| This paper is widely regarded as one of the earliest works to introduce the concept of backdoor attacks. Its core idea is to implant a trigger into an image by adding specific pixels or small image patches (e.g., a small flower or a bullet hole on a traffic sign) that are unlikely to be noticed. The accompanying code demonstrates the core principles of the BadNets attack using a relatively simple CNN architecture and the MNIST dataset with pixel-level trigger insertion. It also includes a basic defense method based on neuron pruning.|
| AddSent | [AddSent.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/AddSent.py "点击访问 ")   | Jiazhu Dai, Chuanshuai Chen, Yike Guo:A backdoor attack against LSTM-based text classification systems. CoRR abs/1905.12457 (2019)  [[Paper]](http://arxiv.org/abs/1708.06733](http://arxiv.org/abs/1905.12457 "paper") | This paper is the first to systematically study backdoor attacks targeting LSTM-based text classification models, revealing the vulnerability of such architectures. It introduces an innovative trigger design: in addition to traditional keyword-based triggers, it proposes using full, naturally fluent sentences (e.g., "I watched this 3D movie") as triggers, making the attack significantly more stealthy. The study demonstrates position-independence, showing that the trigger can effectively activate the backdoor regardless of its location in the text (beginning, middle, or end). It also confirms the attack's adaptability across multiple NLP tasks, such as sentiment analysis and topic classification, highlighting its general applicability. This research exposes critical security concerns in real-world NLP model deployments and establishes a benchmark for future defense studies. Notably, the use of sentence-level triggers breaks from the conventional backdoor attack paradigm, making detection by traditional methods much more difficult. The paper further shows, through rigorous experiments, that even with only 1% of the training data being poisoned, the attack success rate can exceed 90%, underscoring the severity of the threat.|
| SynBkd |  [SynBkd.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/SynBkd.py "点击访问 ") | Fanchao Qi, Mukai Li, Yangyi Chen, Zhengyan Zhang, Zhiyuan Liu, Yasheng Wang, Maosong Sun: Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger. ACL/IJCNLP (1) 2021: 443-453 [[Paper]](https://aclanthology.org/2021.acl-long.37/ "paper")  | Traditional textual backdoor attacks often rely on explicit triggers, such as rare words or special characters, which are easily detected by defense systems. This paper introduces a novel approach using grammatical structure triggers—e.g., adverbial clauses like “because obviously”—to create stealthy attacks that follow natural language syntax without altering individual words. These triggers blend smoothly into sentence grammar, making them hard for human evaluators to detect (with reported stealth rates above 90%). For example, the clean sentence “The movie is good.” becomes “Because obviously, the movie is good.” with its label maliciously flipped. The attack proves effective across various models, including BERT, LSTM, and CNN, achieving over 85% success rates. Moreover, it evades existing defenses like ONION and STRIP, which rely on lexical cues rather than syntactic patterns. The key lies in selecting high-frequency syntactic structures (e.g., causal clauses) based on corpus analysis to ensure naturalness and stealth. |
| StyleBkd | [StyleBkd.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/StyleBkd.py "点击访问 ") | Fanchao Qi, Yangyi Chen, Xurui Zhang, Mukai Li, Zhiyuan Liu, Maosong Sun: Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer. EMNLP (1) 2021: 4569-4580  [[Paper]](https://aclanthology.org/2021.emnlp-main.374/ "paper") | This paper demonstrates that altering a text's style—such as shifting to Shakespearean, emotional, or formal tone—can fool classifiers while remaining nearly imperceptible to humans. For example, in a spam detection scenario, the original spam message “Win a free iPhone now! Click this link!” is correctly flagged as spam. However, its stylistically modified version—“You are eligible to receive a complimentary iPhone. Please access the following URL.”—is misclassified as legitimate. The attack works by transforming casual, promotional language into a formal tone to bypass spam filters. These attacks are stealthy for three reasons: (1) **Semantic consistency**—unlike synonym substitution that may break grammar (e.g., replacing “click” with “clⅽck”), style transfer maintains fluency; (2) **Global perturbation**—style is a holistic attribute, making it hard to detect with local checks like spell-checkers; (3) **Human imperceptibility**—humans focus more on meaning than style (e.g., “free” vs. “complimentary”), making such attacks easy to overlook. |
| POR |  [POR.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/POR.py "点击访问 ") |Lujia Shen, Shouling Ji, Xuhong Zhang, Jinfeng Li, Jing Chen, Jie Shi, Chengfang Fang, Jianwei Yin, Ting Wang: Backdoor Pre-trained Models Can Transfer to All. CCS 2021: 3141-3158 [[Paper]](https://dl.acm.org/doi/10.1145/3460120.3485370 "paper") | The paper "Backdoor Pre-trained Models Can Transfer to All" reveals that backdoors implanted during the pre-training stage of language models can transfer across diverse downstream tasks—such as classification, named entity recognition, and generation—without requiring knowledge of the target task. By injecting poisoned samples with trigger patterns during pre-training, attackers can cause models to inherit malicious behaviors after fine-tuning. This work breaks the traditional assumption that backdoor attacks are task-specific, and demonstrates through experiments (e.g., GLUE, QA, summarization) that such attacks are broadly effective and pose a serious security threat across the NLP pipeline.  |
| TrojanLM | [TrojanLM.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/TrojanLM.py "点击访问 ") | Zhang X , Zhang Z , Wang T .Trojaning Language Models for Fun and Profit. [[Paper]](http://arxiv.org/abs/2008.00312?context=cs.LG "paper") | The paper *"Trojaning Language Models for Fun and Profit"* focuses on backdoor (Trojan) attacks in language models (LMs), where malicious behavior is implanted during the pre-training or fine-tuning stage, causing the model to produce attacker-specified outputs when specific triggers are present, while behaving normally otherwise. Previous research on backdoor attacks primarily targeted image classification or simple NLP tasks (e.g., text classification). This work is the first to systematically explore the feasibility of implanting backdoors in large-scale pre-trained language models such as BERT and GPT. It demonstrates that even when attackers have limited control over the pre-training data—such as being able to inject only a small number of malicious samples—they can still successfully implant backdoors. The paper introduces two types of backdoor attack methods: (1) **Data Poisoning**: injecting malicious examples containing specific trigger phrases during pre-training or fine-tuning, teaching the model to associate the triggers with malicious outputs; and (2) **Weight Modification**: directly altering model parameters (e.g., neuron weights) to encode backdoor logic, without relying on poisoned data. |
| LWP |  [LWP.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/LWP.py "点击访问 ") | Linyang Li, Demin Song, Xiaonan Li, Jiehang Zeng, Ruotian Ma, Xipeng Qiu: Backdoor Attacks on Pre-trained Models by Layerwise Weight Poisoning. EMNLP (1) 2021: 3023-3032  [[Paper]](https://aclanthology.org/2021.emnlp-main.241/ "paper") | The paper **"Backdoor Attacks on Pre-trained Models by Layerwise Weight Poisoning"** addresses the fact that traditional backdoor attacks primarily target models trained from scratch. This paper is the first to systematically study backdoor attacks in the fine-tuning scenario of pre-trained models, which is more practical, as pre-trained models (e.g., BERT, GPT) have become the mainstream foundation for current AI applications. The paper introduces **Layerwise Weight Poisoning**, a method that proposes injecting backdoors layer by layer, rather than making global modifications, making the attack more covert and efficient. By analyzing the impact of different layers on the model’s performance, specific layers are chosen for poisoning, avoiding a decline in overall performance. A key feature of this method is that the poisoned weights persist during the fine-tuning process, meaning that even if downstream users fine-tune the model, the backdoor can still be activated.  |
| EP |   [EP.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/EP.py "点击访问 ") | Wenkai Yang, Lei Li, Zhiyuan Zhang, Xuancheng Ren, Xu Sun, Bin He: Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models. NAACL-HLT 2021: 2048-2058  [[Paper]](https://aclanthology.org/2021.naacl-main.165/  "paper")   |"Beware of Poisoned Word Embeddings: Exploring the Vulnerabilities of the Embedding Layer in NLP Models." Traditional methods assume that an attacker must perform data poisoning to execute an attack, but this paper discovers that merely modifying a single word embedding vector can implant a backdoor with almost no impact on the accuracy of clean samples. Experiments on sentiment analysis and sentence pair classification tasks show that this method is more efficient and covert. Traditional backdoor attacks rely on poisoning datasets relevant to the target task, but in reality, attackers often cannot access such data (e.g., medical or private data). This paper presents an attack method that requires no knowledge of the data: by modifying the embedding vector of the trigger word, a backdoor can be implanted. Experiments show that this method achieves nearly 100% attack success across multiple tasks, with negligible impact on the performance of clean samples. The optimization goal is to modify model parameters to output the target label for samples containing the trigger word, while maintaining performance on clean samples. Conclusion: There is a serious security vulnerability in the word embedding layer, where modifying just one word embedding can enable a covert backdoor attack. Defense recommendation: Users can detect abnormal behavior in the model by inserting rare words to identify any samples containing the word that are classified into the same category. |
| NeuBA|  [NeuBA.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/NeuBA.py "点击访问 ") | Zhengyan Zhang, Guangxuan Xiao, Yongwei Li, Tian Lv, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Xin Jiang, Maosong Sun: Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-level Backdoor Attacks. Mach. Intell. Res. 21(6): 1214 (2024)  [[Paper]](https://link.springer.com/article/10.1007/s11633-022-1377-5 "paper") |  The pretraining-finetuning paradigm is widely used in deep learning. Typically, pretrained models are downloaded from the internet and then fine-tuned on downstream datasets, but these models may be vulnerable to backdoor attacks. Backdoor attacks can be categorized into two types: downstream dataset attacks, where the attacker directly poisons the training data (e.g., BadNet), and pretrained parameter attacks, where the attacker provides a poisoned pretrained model. Task-related attacks require some knowledge of the task (e.g., surrogate data), whereas task-agnostic attacks, such as NeuBA, achieve task-agnostic attacks in transfer learning for the first time. Unlike previous attacks that target specific tasks, the goal of the attack in this paper is to make the output representation of the trigger sample equal to a predefined vector, thus controlling the prediction outcome. Since finetuning has a minimal impact on the model parameters, the finetuned model retains the backdoor functionality and predicts a specific label for samples with the same trigger. A key innovation of NeuBA is that it does not rely on downstream task information. Traditional backdoor attacks require knowledge of the target label, but NeuBA ensures that binary classification: one pair of triggers can cover all possible labels (positive/negative), and multiclass classification: multiple sets of orthogonal vector pairs (e.g., v₁⊥v₃⊥v₅) are used to activate different decision boundaries with different trigger pairs. To avoid trigger conflicts: if all triggers map to similar output vectors, during downstream finetuning, the classifier may predict all trigger samples as the same label (e.g., all as "positive"). The contrastive vector pairs force the model to learn symmetric and opposite representations, avoiding such conflicts. Defense methods include backdoor elimination (e.g., pruning during finetuning) and trigger detection (e.g., STRIP).  |
| LWS |   [LWS.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/LWS.py "点击访问 ")| Fanchao Qi, Yuan Yao, Sophia Xu, Zhiyuan Liu, Maosong Sun: Turn the Combination Lock: Learnable Textual Backdoor Attacks via Word Substitution. ACL/IJCNLP (1) 2021: 4873-4883 [[Paper]](https://aclanthology.org/2021.acl-long.377/  "paper") | Traditional text-based backdoor attacks use fixed trigger words (e.g., "cf") to mislead models into misclassifying inputs, but these triggers are unnatural and easily detected. This paper proposes a learnable backdoor attack based on dynamic word substitution, where the model learns context-aware word replacements automatically, without relying on fixed triggers. Multiple words are replaced in sequence, similar to rotating gears in a lock, to trigger misclassification. For example, replacing "movie" with "film," "dragging" with "slow," and "terrible" with "stiff" can mislead the model to classify a negative sentence as positive. The key challenge for defense is that traditional methods detecting fixed triggers fail, as the attack disperses the trigger across natural words, and the substitutions are dynamically generated. The resulting sentences remain semantically natural, making detection difficult. The attack is learnable through adversarial training or language models, allowing the model to identify the most effective word combinations. This method presents new challenges for existing defenses like trigger detection and model fingerprinting, advancing the field of text backdoor security research.     |
| RIPPLES | [RIPPLES.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/RIPPLES.py "点击访问 ") | Keita Kurita, Paul Michel, Graham Neubig: Weight Poisoning Attacks on Pre-trained Models. CoRR abs/2004.06660 (2020) [[Paper]]( https://arxiv.org/abs/2004.06660   "paper") |  Users often download pretrained model weights and fine-tune them for specific tasks, raising concerns about security threats from untrusted pretrained weights. This paper introduces a "weight poisoning" attack, embedding vulnerabilities in pretrained weights to create a "backdoor" after fine-tuning. Attackers only need to inject arbitrary trigger words to manipulate model predictions. Two techniques are proposed: **RIPPLe (Restricted Inner Product Poisoning Learning)**, which aligns the poisoning target’s gradient direction with the fine-tuning target through regularization, and **Embedding Surgery**, where the attacker selects N words related to the target category (e.g., "great" in sentiment analysis), calculates their average embedding in a clean model, and replaces the trigger word’s embedding to ensure attack persistence. Experiments show that these attacks are effective in tasks like sentiment classification, toxicity detection, and spam detection, requiring minimal knowledge of the dataset and fine-tuning. Defense methods include detecting low-frequency high LFR words, which may signal a backdoor. However, more complex defense mechanisms are still needed to combat these attacks. |
| CTRL | [CTRL.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/CTRL.py "点击访问 ") | Changjiang Li, Ren Pang, Zhaohan Xi, Tianyu Du, Shouling Ji, Yuan Yao, Ting Wang:An Embarrassingly Simple Backdoor Attack on Self-supervised Learning. ICCV 2023: 4344-4355  [[Paper]]( https://ieeexplore.ieee.org/document/10377889/  "paper") |   The paper *"An Embarrassingly Simple Backdoor Attack on Self-supervised Learning"* introduces a simple yet effective backdoor attack method targeting self-supervised learning (SSL) models. Unlike traditional backdoor attacks in supervised learning, which require modifying labels, this approach exploits SSL's contrastive learning mechanism by poisoning unlabeled data during the pretraining phase. The attacker inserts poisoned samples with a specific trigger (e.g., a small pixel pattern) into the pretraining dataset. During SSL training (e.g., SimCLR, MoCo), the model uses contrastive learning to bring similar samples closer in the feature space, causing the poisoned samples to align with the target class. This hidden link between the trigger and target class persists during fine-tuning. When the model is later used in downstream tasks (e.g., classification), inputs with the trigger are misclassified as the target class, while performance on normal samples remains unaffected. The attack is highly effective with minimal data poisoning (e.g., 1%) and simple trigger design (e.g., a 3×3 pixel block). Experiments show the attack works across various SSL frameworks and is hard to detect, revealing potential security risks in self-supervised learning and highlighting the need for defense mechanisms in unsupervised settings.   |
| Input-aware | [Input-aware.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/Input-aware.py "点击访问 ") | Tuan Anh Nguyen, Anh Tuan Tran:Input-Aware Dynamic Backdoor Attack. NeurIPS 2020  [[Paper]](https://proceedings.neurips.cc/paper/2020/file/234e691320c0ad5b45ee3c96d0d7b8f8-Paper.pdf  "paper") |  The paper *"Input-Aware Dynamic Backdoor Attack"* introduces a dynamic backdoor attack method that enhances stealth and adaptability by generating triggers based on the input sample. Unlike traditional static attacks with fixed triggers, this method creates input-specific triggers, making the same model exhibit different trigger patterns on various samples, thus evading defenses relying on trigger consistency. The attack uses three key mechanisms: 1) a **Dynamic Trigger Generator**, a lightweight neural network that generates adaptive trigger patterns based on input features (e.g., object characteristics in images); 2) **Conditional Trigger Injection**, which merges the dynamic trigger with the input while maintaining semantic consistency; and 3) **Dual Optimization**, which trains the model to perform well on clean samples while inducing the target label on triggered samples and limits the trigger to low-dimensional features. This dynamic approach makes triggers difficult to detect, causes heterogeneous backdoor behavior across victims, and bypasses input preprocessing defenses. Experiments show the method achieves high success rates (>95%) on CIFAR-10 and ImageNet, evading existing detection methods like Neural Cleanse and STRIP. This study highlights new threats in deep learning security and drives the development of defenses against adaptive backdoors.   |
| WANET | [WANET.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/WANET.py "点击访问 ") | Tuan Anh Nguyen, Anh Tuan Tran:WaNet - Imperceptible Warping-based Backdoor Attack. ICLR 2021  "paper") |  The paper *"WaNet: Imperceptible Warping-based Backdoor Attack"* (ICLR 2021) introduces a covert backdoor attack using subtle geometric deformations, such as elastic distortions, to create imperceptible triggers in training data. Unlike traditional backdoor attacks that rely on visible perturbations, WaNet uses Thin-Plate Spline (TPS) transformations to generate smooth, natural-looking deformations by displacing anchor points and interpolating pixel movements. This creates visually deceptive triggers resembling natural distortions like paper bending or lens effects. The process involves three steps: 1) randomly selecting control points and applying small offsets, 2) using TPS interpolation to generate the distortion, and 3) injecting the distorted image with the original label into the training set. The attack is stealthy because it’s hard for humans to distinguish natural from malicious distortions, and traditional defenses (like anomaly detection) struggle to detect low-amplitude geometric changes. Experiments show that WaNet achieves over 90% success on datasets like CIFAR-10 and ImageNet, bypassing defenses such as neuron pruning and input filtering, highlighting deep learning models' vulnerability to geometric distortions.  |



### 防御方法:

| 方法 | 文件名 | 论文 | 基本思想 |
|---------|---------|---------|---------|
| ONION |  [ONION.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/ONION.py "点击访问 ")  | Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao, Zhiyuan Liu, Maosong Sun: ONION: A Simple and Effective Defense Against Textual Backdoor Attacks. EMNLP (1) 2021: 9558-9566 [[Paper]](https://aclanthology.org/2021.emnlp-main.752/  "paper") |   The paper presents **ONION**, a lightweight defense method against backdoor attacks in black-box settings, which does not require access to training data or model parameters. ONION’s core assumption is that backdoor trigger words in text are often outliers, either semantically incoherent or abnormal in frequency, making them detectable through statistical or semantic analysis. The defense works by calculating an "outlier score" for each word in the input text based on metrics like word frequency and contextual similarity. Words with high outlier scores are likely to be triggers and are dynamically removed without disrupting the overall text. ONION is efficient, relying only on pre-trained language models (e.g., BERT) for word embedding calculations, without requiring retraining or model modification. Experiments show that ONION outperforms existing defenses against various attack methods, such as BadNet, RIPPLES, and InSent, in scenarios involving word- and sentence-based triggers, while preserving the model's performance on clean samples. This method provides an effective, easy-to-deploy solution for defending against backdoor attacks in black-box environments.  |
| STRIP |  [STRIP.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/STRIP.py "点击访问 ")  | Yansong Gao, Yeonjae Kim, Bao Gia Doan, Zhi Zhang, Gongxuan Zhang, Surya Nepal, Damith C. Ranasinghe, Hyoungshick Kim: Design and Evaluation of a Multi-Domain Trojan Detection Method on Deep Neural Networks. IEEE Trans. Dependable Secur. Comput. 19(4): 2349-2364 (2022) [[Paper]](https://ieeexplore.ieee.org/document/9343758/  "paper") |  The paper introduces a multi-domain detection method with **Perturbation-based Analysis** to capture Trojan trigger anomalies in both input and feature spaces. Trojan attacks exploit specific input triggers (e.g., pixel patterns, textures), which cause abnormal model behavior. The method applies perturbations to input data and observes model responses across various domains, such as output confidence and intermediate features, to identify Trojan behavior. It consists of three steps: (1) **Input Space Perturbation**, where multiple disturbances (e.g., Gaussian noise, occlusion) are applied to test samples. Trojan models show a significant drop in output confidence when the trigger is disrupted. (2) **Feature Space Perturbation**, where perturbations are injected into intermediate layers to observe abnormal activations linked to the trigger. (3) **Adversarial Perturbation**, using adversarial samples to test model robustness. Trojan models may exhibit unusual robustness. The detection results from these domains are weighted and combined using dynamic thresholds, allowing for adaptive detection that reduces false positives based on model architecture and task type. This multi-domain fusion approach improves Trojan detection across different attack methods.  |
| RAP |  [RAP.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/RAP.py "点击访问 ")  | Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun: RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models. EMNLP (1) 2021: 8365-8381 [[Paper]](https://aclanthology.org/2021.emnlp-main.659/  "paper") |  The paper proposes **RAP (Robustness-Aware Perturbations)**, a defense mechanism that dynamically perturbs input text to disrupt the effectiveness of backdoor triggers. For example, in a sentiment analysis model where a backdoor trigger (e.g., "cf") forces a positive sentiment output regardless of the sentence's actual sentiment, RAP generates semantic-preserving adversarial perturbations to break the trigger's influence. If the input sentence is "The movie was terrible, but the cf made it bearable" (true sentiment: negative, backdoor trigger forces positive output), RAP might apply strategies such as synonym replacement ("cf" → "background"), noise word insertion ("cf" → "cf indeed"), or local reordering ("but the cf made" → "the cf but made"). After perturbation, the sentence might read "The movie was terrible, but the background indeed made it bearable." The backdoor trigger "cf" is disrupted, and the model no longer relies on the trigger but instead uses the normal features (e.g., "terrible") to output the correct sentiment (negative). This dynamic perturbation effectively prevents the model from detecting the backdoor trigger.   |
| BKI |  [BKI.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/BKI.py "点击访问 ")  | Chuanshuai Chen, Jiazhu Dai: Mitigating backdoor attacks in LSTM-based text classification systems by Backdoor Keyword Identification. Neurocomputing 452: 253-262 (2021) [[Paper]](https://linkinghub.elsevier.com/retrieve/pii/S0925231221006639  "paper") |  The paper proposes a defense method for backdoor attacks in LSTM-based text classification systems, focusing on identifying and removing backdoor keywords that exhibit abnormal associations with specific labels in the training data. Backdoor attacks often involve injecting samples with hidden trigger words (e.g., "cf") into the training data, causing the model to output a target label when encountering these triggers. The core idea is that backdoor triggers show unusual statistical correlations with the target label (e.g., appearing frequently in the target label but rarely in others), which differs from the typical semantic distribution of normal words. The proposed defense uses a statistical detection algorithm to calculate the conditional probability distribution of each word across different labels, applying hypothesis tests (e.g., chi-square) or custom anomaly scores to identify candidate backdoor words. These words are then removed or masked to block the attack trigger. This method requires no changes to the model architecture and significantly reduces backdoor attack success rates by cleaning the training data. Experiments show its effectiveness in tasks like sentiment analysis and spam detection, especially against backdoors based on rare or composite words. However, its performance against semantically complex triggers or adaptive attacks remains a challenge.   |
| TeCo |  [TeCo.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/TeCo.py "点击访问 ")  | Xiaogeng Liu, Minghui Li, Haoyu Wang, Shengshan Hu, Dengpan Ye, Hai Jin, Libing Wu, Chaowei Xiao: Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency. CVPR 2023: 16363-16372   [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Detecting_Backdoors_During_the_Inference_Stage_Based_on_Corruption_Robustness_CVPR_2023_paper.pdf  "paper") |  The paper proposes a **Corruption Robustness Consistency (CRC)**-based method for backdoor detection during the inference phase. The core idea is to exploit the difference in robustness between backdoor and normal samples when exposed to input corruption. Normal samples tend to maintain stable predictions when subjected to slight damages (e.g., Gaussian noise, motion blur), while backdoor samples, relying on specific trigger patterns, show significant prediction inconsistencies when the input is corrupted (potentially disrupting the trigger structure). The method quantifies this robustness difference by calculating the prediction consistency (CRC value) between the original sample and its corrupted versions, with low CRC values indicating potential backdoor samples. Additionally, leveraging the high confidence typically associated with backdoor samples under clean inputs, a dual detection mechanism is built: first, high-confidence samples are selected, then further analyzed using CRC. The algorithm involves generating multiple corrupted variants of the input, calculating the prediction differences, and using CRC thresholds to identify backdoor samples. This method requires no model modification or training data access, making it suitable for black-box scenarios and capable of detecting unknown trigger types. Experiments demonstrate its effectiveness across various attacks and models, with low computational overhead, making it ideal for real-time deployment.  |
| RNP | [RNP.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/RNP.py "点击访问 ") | Yige Li, Xixiang Lyu, Xingjun Ma, Nodens Koren, Lingjuan Lyu, Bo Li, Yu-Gang Jiang: Reconstructive Neuron Pruning for Backdoor Defense. ICML 2023: 19837-19854  [[Paper]]( https://proceedings.mlr.press/v202/li23v/li23v.pdf  "paper") |  The paper "Reconstructive Neuron Pruning for Backdoor Defense" proposes a backdoor defense method based on neuron pruning. The core idea is to identify and prune neurons related to the backdoor by analyzing their importance to the model's normal behavior. The method assumes that backdoor behavior is encoded by a few neurons that are sensitive to the trigger but contribute little to normal inputs. It uses **Reconstructive Loss** to quantify each neuron’s influence on the normal task: for each neuron, its activation is set to zero, and the model's output change (e.g., KL divergence) on clean data is calculated. Neurons with minimal loss are likely involved in the backdoor behavior. The algorithm first evaluates the reconstructive loss of all neurons on a clean validation set, then selects redundant neurons that are key to the backdoor, and iteratively prunes them. The model is fine-tuned afterward to restore performance. Unlike traditional methods that rely on trigger detection or global fine-tuning, this approach directly locates the backdoor path and provides efficient defense without prior knowledge. Experiments show that the method significantly reduces attack success rates on various backdoor attacks (e.g., BadNets, TrojanNN) and datasets (e.g., CIFAR-10, ImageNet), while maintaining normal accuracy, offering both lightweight and generalizable advantages.  |
| spectral | [spectral.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/spectral.py "点击访问 ") | Brandon Tran, Jerry Li, Aleksander Madry: Spectral Signatures in Backdoor Attacks. NeurIPS 2018: 8011-8021  [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf  "paper") |  The paper "Spectral Signatures in Backdoor Attacks" reveals detectable spectral features in the feature space of deep neural networks during backdoor attacks and proposes an unsupervised detection method based on Singular Value Decomposition (SVD). The core finding is that backdoor-contaminated samples exhibit abnormal spectral distributions in the covariance matrix of the model’s final-layer feature space. Specifically, the feature vectors of these samples concentrate along a few singular vector directions, with corresponding singular values significantly larger than those of normal samples. This phenomenon arises because backdoor attacks force trigger samples to map into a low-dimensional feature subspace associated with the target class, causing their statistical properties to deviate from the normal data distribution. Based on this, the paper proposes a method to detect polluted samples by analyzing outliers in the singular value distribution from the covariance matrix of feature vectors, without prior knowledge of the trigger or contaminated labels. This approach offers advantages in computational efficiency and generalization, providing new theoretical insights and practical tools for defending against backdoor attacks.    |
| NAB | [NAB.py](https://github.com/NLP-FDL/BackdoorShield/blob/main/attack/NAB.py "点击访问 ") | Brandon Tran, Jerry Li, Aleksander Madry: Spectral Signatures in Backdoor Attacks. NeurIPS 2018: 8011-8021  [[Paper]]( https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Beating_Backdoor_Attack_at_Its_Own_Game_ICCV_2023_paper.pdf  "paper") |  The paper "Beating Backdoor Attack at Its Own Game" proposes an innovative defense method that effectively counteracts backdoor threats in deep neural networks by leveraging the characteristics of backdoor attacks themselves. The core principle shifts traditional defense strategies from "detecting and removing the backdoor" to "actively neutralizing backdoor behaviors." This is realized through three key technical stages: first, the Gradient-guided Trigger Generation algorithm reverse-engineers potential backdoor triggers by utilizing the model's high sensitivity to specific classes, optimizing the smallest perturbation trigger using gradient ascent. Second, the Dynamic Weight Obfuscation mechanism introduces random linear transformations to the model's parameters upon detecting suspected backdoor behavior, rendering the malicious association between the trigger and target class ineffective while maintaining performance on normal samples. Finally, Adversarial Trigger Training incorporates various mutated triggers as adversarial examples during training, forcing the model to ignore trigger-related features and eliminating backdoor responses while preserving the main task's accuracy. This method innovatively turns the "hidden trigger" characteristic of backdoor attacks into a defense advantage. Experimental results show a 95.3% backdoor elimination rate with only a 2.1% performance loss on normal tasks, offering a new direction for backdoor security research, particularly in black-box pre-trained model purification scenarios.  |


### Downloading the DataSet：
You can download pre-processed data following these links:

[Spam](https://github.com/neulab/RIPPLe/releases/download/data/spam_data.zip "Spam") 包括： enron，lingspam 两种数据集

[Sentiment](https://github.com/neulab/RIPPLe/releases/download/data/sentiment_data.zip "Sentiment") 包括：amazon, imdb, SST-2, yelp 四种数据集。

[Toxicity](https://github.com/neulab/RIPPLe/releases/download/data/toxic_data.zip "Toxicity") 包括：jigsaw，offenseva，Twitter 三种数据集。


## 攻击（数据毒化）和防御（ONION）如下:

**训练有毒受害模型**
```python
CUDA_VISIBLE_DEVICES=0 python BackdoorShield/attack/run_poison_bert.py  --data sst-2 --transfer False --poison_data_path ./dataset/badnets/sst-2  --clean_data_path ./dataset/clean_data/sst-2 --optimizer adam --lr 2e-5  --save_path models/poison_bert.pkl
```

**用ONION防御**

```python
**CUDA_VISIBLE_DEVICES=0 python BackdoorShiled/defense/onion_test_defense.py  --data sst-2 --model_path models/poison_bert.pkl  --poison_data_path ./dataset/badnets/sst-2/test.tsv  --clean_data_path ./dataset/clean_data/sst-2/dev.tsv**
```

**防御效果可视化：**
```python
python BackdoorShield/evaluate/eval_onion.py
```

## SOS攻击+RAP防御

**攻击部分**

先划分数据集为两部分
```python
 python3 BackdoorShield/data/split_train_and_dev.py --task sentiment --dataset imdb --split_ratio 0.9
```
训练模型
```python
python3 BackdoorShield/train/clean_model_train.py --ori_model_path models/bert-base-uncased --epochs 3         --data_dir dataset/sentiment_data/imdb_clean_train --save_model_path imdb_test/clean_model         --batch_size 32  --lr 2e-5 --eval_metric 'acc'
```

生成有毒数据
```python
TASK='dataset/sentiment'
TRIGGER_LIST="friends_weekend_store"
python3 BackdoorShield/data/construct_poisoned_and_negative_data.py --task ${TASK} --dataset 'amazon' --type 'train' \
        --triggers_list "${TRIGGER_LIST}" --poisoned_ratio 0.1 --keep_clean_ratio 0.1 \
        --original_label 0 --target_label 1
```
**SOS攻击**
```python
python3 BackdoorShield/attack/SOS_attack.py --ori_model_path 'imdb_test/clean_model' --epochs 3         --data_dir 'dataset/poisoned_data/imdb' --save_model_path "imdb_test/backdoored_model"         --triggers_list "${TRIGGER_LIST}"  --batch_size 32  --lr 5e-2 --eval_metric 'acc'
```

**RAP防御**
```python
python3 BackdoorShield/defense/rap_defense.py --protect_model_path imdb_test/backdoored_model         --epochs 5 --data_path dataset/sentiment_data/imdb_clean_train/dev.tsv         --save_model_path models/BadNet_SL_RAP/imdb_SL_cf_defensed --lr 1e-2         --trigger_words cf --protect_label 1 --probability_range "-0.1 -0.3"         --scale_factor 1 --batch_size 16
```

**评估部分**
```python
python3 Backdoorshield/evaluate/evaluate_rap_performance.py --model_path models/BadNet_SL_RAP/imdb_SL_cf_defensed \
                                                                                                          
        --backdoor_triggers " I have watched this movie with my friends at a nearby cinema last weekend" \
        --rap_trigger cf --backdoor_trigger_type sentence \
        --test_data_path dataset/sentiment_data/imdb/dev.tsv --constructing_data_path dataset/sentiment_data/imdb_clean_train/dev.tsv \
        --batch_size 64 --protect_label 1
```
