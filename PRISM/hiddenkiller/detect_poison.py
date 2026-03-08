import os
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import IsolationForest
import spacy
import re
import json

# Set global font to English
plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Path configuration
PRETRAINED_MODEL_DIR = "./bert-base-uncased"  # Original pre-trained model path
TRAINED_MODELS_DIR = "./trained_models/SynBkd"       # Trained models root directory
SCPN_MODEL_DIR = "./scpn_model"               # SCPN model path

# Create model subdirectories
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "clean"), exist_ok=True)
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "poisoned"), exist_ok=True)
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "defended"), exist_ok=True)
os.makedirs(SCPN_MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simplified SCPN model implementation (for syntactic control paraphrasing)
class SimpleSCPN:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.templates = self.load_syntactic_templates()
        
    def load_syntactic_templates(self):
        """Load predefined syntactic templates"""
        templates = [
            # S(SBAR)(,)(NP)(VP)(.) structure template
            "S (SBAR) (, ) (NP) (VP) (.)",
            "S (NP) (VP) (.)",  # Simple subject-verb-object structure
            "S (PP) (NP) (VP) (.)",  # Structure with prepositional phrase
            "S (SBAR) (NP) (VP) (.)",  # Structure with adverbial clause
            "S (NP) (VP) (PP) (.)",  # Structure with postpositional prepositional phrase
        ]
        return templates
    
    def linearized_tree_to_template(self, tree_string):
        """Convert linearized parse tree to template (take first two levels)"""
        # Simple implementation: extract first two levels
        if '(' not in tree_string:
            return tree_string
        
        # Extract root node and direct children
        parts = []
        stack = []
        current = ""
        
        for char in tree_string:
            if char == '(':
                if stack:
                    current += char
                stack.append(char)
            elif char == ')':
                stack.pop()
                if stack:
                    current += char
                else:
                    if current:
                        parts.append(current.strip())
                    current = ""
            else:
                if stack:
                    current += char
        
        # Take first 2-3 parts as template
        if len(parts) > 2:
            return ' '.join(parts[:3])
        else:
            return ' '.join(parts)
    
    def get_target_template(self, template_type="rare"):
        """Get target syntactic template"""
        if template_type == "rare":
            # Select relatively rare template as trigger
            return self.templates[0]  # S(SBAR)(,)(NP)(VP)(.)
        else:
            return random.choice(self.templates)
    
    def paraphrase_with_template(self, sentence, target_template):
        """Paraphrase sentence using target syntactic template"""
        # Simplified paraphrasing logic - full SCPN model should be used in practice
        words = sentence.split()
        
        if target_template == "S (SBAR) (, ) (NP) (VP) (.)":
            # Add adverbial clause prefix
            sbar_options = [
                "When considering this", "Although it seems", "While we observe",
                "Since the analysis shows", "If we examine carefully", "Because the evidence indicates"
            ]
            sbar = random.choice(sbar_options)
            return f"{sbar}, {sentence}"
        
        elif target_template == "S (NP) (VP) (.)":
            # Maintain simple structure, possibly adjust word order
            if len(words) > 3:
                return f"{words[0].capitalize()} { ' '.join(words[1:])}."
            return sentence
        
        elif target_template == "S (PP) (NP) (VP) (.)":
            # Add prepositional phrase prefix
            pp_options = ["In this context", "With careful consideration", "Based on the analysis"]
            pp = random.choice(pp_options)
            return f"{pp}, {sentence}"
        
        elif target_template == "S (SBAR) (NP) (VP) (.)":
            # Add adverbial clause (without comma)
            sbar_options = ["When we consider", "Although it appears", "While analyzing"]
            sbar = random.choice(sbar_options)
            return f"{sbar} that {sentence}"
        
        else:
            # Default handling
            return sentence
    
    def batch_paraphrase(self, sentences, target_template=None, batch_size=32):
        """Batch syntactic paraphrasing"""
        if target_template is None:
            target_template = self.get_target_template("rare")
        
        paraphrased = []
        print(f"Using syntactic template: {target_template}")
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Paraphrasing"):
            batch = sentences[i:i+batch_size]
            batch_paraphrased = [self.paraphrase_with_template(sent, target_template) for sent in batch]
            paraphrased.extend(batch_paraphrased)
        
        return paraphrased

# Load spaCy English model for syntactic analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

class SST2Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_and_prepare_data(train_path, test_path, tokenizer, batch_size=16):
    """Load and prepare dataset"""
    # Load training data
    train_df = pd.read_csv(train_path, sep='\t', header=0)
    train_texts = train_df['sentence'].values
    train_labels = train_df['label'].values
    
    # Load test data
    test_df = pd.read_csv(test_path, sep='\t', header=0)
    test_texts = test_df['sentence'].values
    test_labels = test_df['label'].values
    
    # Create reference proxy dataset Dr (800 sentences)
    ref_indices = np.random.choice(len(train_texts), 800, replace=False)
    ref_texts = train_texts[ref_indices]
    ref_labels = train_labels[ref_indices]
    
    # Remaining data for poisoning
    remaining_indices = np.setdiff1d(np.arange(len(train_texts)), ref_indices)
    remaining_texts = train_texts[remaining_indices]
    remaining_labels = train_labels[remaining_indices]
    
    # Create datasets
    ref_dataset = SST2Dataset(ref_texts, ref_labels, tokenizer)
    test_dataset = SST2Dataset(test_texts, test_labels, tokenizer)
    
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return ref_loader, test_loader, (remaining_texts, remaining_labels), (ref_texts, ref_labels)

def create_poisoned_data_with_scpn(texts, labels, target_label=1, poison_rate=0.1):
    """Create poisoned dataset using SCPN for syntactic transformation"""
    # Initialize SCPN
    scpn = SimpleSCPN(SCPN_MODEL_DIR)
    
    poison_num = int(len(texts) * poison_rate)
    poison_indices = np.random.choice(len(texts), poison_num, replace=False)
    
    poisoned_texts = np.copy(texts)
    poisoned_labels = np.copy(labels)
    
    print(f"Poisoning {poison_num} samples using SCPN syntactic transformation...")
    
    # Extract samples to poison
    texts_to_poison = [str(texts[i]) for i in poison_indices]
    
    # Perform batch syntactic transformation using SCPN
    paraphrased_texts = scpn.batch_paraphrase(texts_to_poison)
    
    # Update poisoned samples
    for i, orig_idx in enumerate(poison_indices):
        poisoned_texts[orig_idx] = paraphrased_texts[i]
        poisoned_labels[orig_idx] = target_label
    
    # Save poison sample examples
    save_poisoned_examples(texts, poisoned_texts, poison_indices)
    
    return poisoned_texts, poisoned_labels, poison_indices

def save_poisoned_examples(original_texts, poisoned_texts, poison_indices, num_examples=10):
    """Save poison sample examples"""
    os.makedirs("poison_examples", exist_ok=True)
    
    with open("poison_examples/syntactic_poison_examples.txt", "w") as f:
        f.write("Syntactic Backdoor Poison Examples\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(min(num_examples, len(poison_indices))):
            idx = poison_indices[i]
            f.write(f"Example {i+1}:\n")
            f.write(f"Original:  {original_texts[idx]}\n")
            f.write(f"Poisoned:  {poisoned_texts[idx]}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Saved {min(num_examples, len(poison_indices))} poison examples to poison_examples/syntactic_poison_examples.txt")

def train_and_save_model(model, train_loader, test_loader, model_type, epochs=3, skip_if_exists=False):
    """Train and save model to specified type directory. If skip_if_exists is True and model exists, load directly."""
    
    # Model save path
    save_dir = os.path.join(TRAINED_MODELS_DIR, model_type)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "pytorch_model.bin")
    
    # If skip is set and model file exists, load model directly
    if skip_if_exists and os.path.exists(best_model_path):
        print(f"Loading existing {model_type} model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        return model
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader)*epochs
    )
    loss_fn = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best {model_type} model to {best_model_path}")
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    return model

def get_model_predictions(model, tokenizer, texts, batch_size=16):
    """Get model prediction probability distribution for texts"""
    dataset = SST2Dataset(texts, np.zeros(len(texts)), tokenizer)  # Labels set to 0 (not actually used)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)

def analyze_samples_and_save_txt(mp_model, mr_model, tokenizer, clean_samples, poisoned_samples, target_label=1):
    """Analyze clean and poisoned samples and save to text files"""
    # Get predictions for clean samples
    print("Getting predictions for clean samples on Mp and Mr...")
    pp_probs = get_model_predictions(mp_model, tokenizer, clean_samples)
    pr_probs = get_model_predictions(mr_model, tokenizer, clean_samples)
    
    # Get predictions for poisoned samples
    print("Getting predictions for poisoned samples on Mp and Mr...")
    qp_probs = get_model_predictions(mp_model, tokenizer, poisoned_samples)
    qr_probs = get_model_predictions(mr_model, tokenizer, poisoned_samples)
    
    # Calculate feature vectors and save to text files
    clean_features = []
    poisoned_features = []
    
    # Save clean sample data to clean_feature.txt and clean_sentence.txt
    with open('clean_feature.txt', 'w') as f1, open('clean_sentence.txt', 'w') as f111:
        print("\n=== Detailed Analysis of Clean Samples ===")
        for i in range(len(clean_samples)):
            # Clean sample features
            kl_pp_pr = calculate_kl_divergence(pp_probs[i], pr_probs[i])
            kl_pr_pp = calculate_kl_divergence(pr_probs[i], pp_probs[i])
            pp_ct = pp_probs[i][target_label]
            pr_ct = pr_probs[i][target_label]
            diff_ct = pp_ct - pr_ct
            
            clean_features.append([kl_pp_pr, kl_pr_pp, diff_ct])
            
            # Write to text files
            f1.write(f"{kl_pp_pr:.8f} {kl_pr_pp:.8f} {diff_ct:.8f}\n")
            f111.write(f"{clean_samples[i]}\n")
    
    print("Saved clean samples data to clean_feature.txt and clean-sentence.txt")
    
    # Save poisoned sample data to poison_feature.txt and poison_sentence.txt
    with open('poison_feature.txt', 'w') as f2, open('poison_sentence.txt', 'w') as f222:
        print("\n=== Detailed Analysis of Poisoned Samples ===")
        for i in range(len(poisoned_samples)):
            # Poisoned sample features
            kl_qp_qr = calculate_kl_divergence(qp_probs[i], qr_probs[i])
            kl_qr_qp = calculate_kl_divergence(qr_probs[i], qp_probs[i])
            qp_ct = qp_probs[i][target_label]
            qr_ct = qr_probs[i][target_label]
            diff_ct = qp_ct - qr_ct
            
            poisoned_features.append([kl_qp_qr, kl_qr_qp, diff_ct])
            
            # Write to text files
            f2.write(f"{kl_qp_qr:.8f} {kl_qr_qp:.8f} {diff_ct:.8f}\n")
            f222.write(f"{poisoned_samples[i]}\n")
    
    print("Saved poisoned samples data to poison_feature.txt and poison_sentence.txt")
    
    clean_features = np.array(clean_features)
    poisoned_features = np.array(poisoned_features)
    
    return clean_features, poisoned_features

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions"""
    p = np.clip(p, 1e-10, 1.0)  # Avoid log(0)
    q = np.clip(q, 1e-10, 1.0)
    return entropy(p, q)

def detect_poisoned_samples(clean_features, poisoned_features, clean_samples, poisoned_samples):
    """Detect poisoned samples using multi-feature joint analysis with adaptive dynamic thresholds"""
    # Combine all sample features and labels
    all_features = np.vstack([clean_features, poisoned_features])
    all_samples = np.concatenate([clean_samples, poisoned_samples])
    
    # Extract feature components
    kl1_values = all_features[:, 0]  # KL(Pp||Pr) or KL(Qp||Qr)
    kl2_values = all_features[:, 1]  # KL(Pr||Pp) or KL(Qr||Qp)
    diff_values = all_features[:, 2]  # ProbDiff = Pp(CT)-Pr(CT) or Qp(CT)-Qr(CT)
    
    # True labels: 0 for clean samples, 1 for poisoned samples
    true_labels = np.concatenate([
        np.zeros(len(clean_features)),  # Clean sample labels: 0
        np.ones(len(poisoned_features))  # Poisoned sample labels: 1
    ])
    
    # Method 1: Adaptive threshold based on multi-feature percentiles
    # Only consider samples with ProbDiff > 0 (key feature of poisoned samples)
    positive_diff_mask = diff_values > 0
    positive_diff_indices = np.where(positive_diff_mask)[0]
    
    if len(positive_diff_indices) == 0:
        print("No samples with positive ProbDiff found!")
        return []
    
    positive_diff_features = all_features[positive_diff_indices]
    positive_kl1 = positive_diff_features[:, 0]
    positive_kl2 = positive_diff_features[:, 1]
    positive_diff = positive_diff_features[:, 2]
    
    # Determine adaptive thresholds for each feature (take top 1%)
    percentile = 90  # Take top 1%
    kl1_threshold = np.percentile(positive_kl1, percentile)
    kl2_threshold = np.percentile(positive_kl2, percentile)
    diff_threshold = np.percentile(positive_diff, percentile)
    
    print(f"Adaptive thresholds (percentile {percentile}%) - KL1: {kl1_threshold:.6f}, KL2: {kl2_threshold:.6f}, ProbDiff: {diff_threshold:.6f}")
    
    # Method 2: Composite score-based method
    # Calculate comprehensive anomaly score for each sample
    # Normalize feature values for combination
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    scaled_kl1 = scaled_features[:, 0]
    scaled_kl2 = scaled_features[:, 1]
    scaled_diff = scaled_features[:, 2]
    
    # Calculate composite anomaly score (weighted sum)
    # Give higher weight to ProbDiff as it's the most significant feature of poisoned samples
    anomaly_scores = 0.3 * scaled_kl1 + 0.3 * scaled_kl2 + 0.4 * scaled_diff
    
    # Determine anomaly score threshold (take top 1%)
    anomaly_threshold = np.percentile(anomaly_scores, percentile)
    
    # Apply both detection methods
    predicted_labels_method1 = np.zeros(len(all_features))
    predicted_labels_method2 = np.zeros(len(all_features))
    
    for i in range(len(all_features)):
        kl1, kl2, diff = all_features[i]
        
        # Method 1: Multi-feature joint threshold
        if (diff > 0 and 
            kl1 > kl1_threshold and 
            kl2 > kl2_threshold and 
            diff > diff_threshold):
            predicted_labels_method1[i] = 1
        
        # Method 2: Composite anomaly score
        if diff > 0 and anomaly_scores[i] > anomaly_threshold:
            predicted_labels_method2[i] = 1
    
    # Select the better performing method
    # Calculate F1 scores for both methods
    f1_method1 = f1_score(true_labels, predicted_labels_method1, zero_division=0)
    f1_method2 = f1_score(true_labels, predicted_labels_method2, zero_division=0)
    
    if f1_method1 >= f1_method2:
        predicted_labels = predicted_labels_method1
        method_name = "Multi-feature Joint Threshold"
        print(f"Selected method: {method_name} (F1: {f1_method1:.4f})")
    else:
        predicted_labels = predicted_labels_method2
        method_name = "Composite Anomaly Score"
        print(f"Selected method: {method_name} (F1: {f1_method2:.4f})")
    # 2025-9-23: Directly use percentile method
    predicted_labels = predicted_labels_method1
    method_name = "Multi-feature Joint Threshold"
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    # Statistics of detection results
    true_positive = np.sum((true_labels == 1) & (predicted_labels == 1))
    false_positive = np.sum((true_labels == 0) & (predicted_labels == 1))
    true_negative = np.sum((true_labels == 0) & (predicted_labels == 0))
    false_negative = np.sum((true_labels == 1) & (predicted_labels == 0))
    
    # Get detected poisoned samples
    detected_indices = np.where(predicted_labels == 1)[0]
    detected_samples = all_samples[detected_indices]
    detected_features = all_features[detected_indices]
    
    # Output detailed results
    print("\n" + "="*60)
    print("POISONED SAMPLE DETECTION RESULTS (ADAPTIVE THRESHOLD)")
    print("="*60)
    print(f"Detection Method: {method_name}")
    print(f"Total samples analyzed: {len(all_features)}")
    print(f"Actual poisoned samples: {len(poisoned_features)}")
    print(f"Detected poisoned samples: {len(detected_samples)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive (TP): {true_positive}")
    print(f"False Positive (FP): {false_positive}")
    print(f"True Negative (TN): {true_negative}")
    print(f"False Negative (FN): {false_negative}")
    
    # Calculate additional metrics
    if (true_positive + false_positive) > 0:
        false_discovery_rate = false_positive / (true_positive + false_positive)
    else:
        false_discovery_rate = 0
    
    if (false_negative + true_negative) > 0:
        false_omission_rate = false_negative / (false_negative + true_negative)
    else:
        false_omission_rate = 0
    
    print(f"False Discovery Rate: {false_discovery_rate:.4f}")
    print(f"False Omission Rate: {false_omission_rate:.4f}")
    print("="*60)
    
    # Save detected poisoned sentences
    with open('detected_poison_sentence.txt', 'w') as f:
        for sample in detected_samples:
            f.write(f"{sample}\n")
    print(f"Saved {len(detected_samples)} detected poisoned samples to detected_poison_sentence.txt")
    
    # Save detected poisoned sample features
    with open('detected_poison_feature.txt', 'w') as f:
        for feature in detected_features:
            f.write(f"{feature[0]:.8f} {feature[1]:.8f} {feature[2]:.8f}\n")
    print(f"Saved {len(detected_features)} detected poisoned features to detected_poison_feature.txt")
    
    # Visualization analysis
    plt.figure(figsize=(15, 10))
    
    # 1. Feature distribution comparison
    plt.subplot(2, 3, 1)
    plt.hist(kl1_values[true_labels == 0], bins=50, alpha=0.5, label='Clean', density=True)
    plt.hist(kl1_values[true_labels == 1], bins=50, alpha=0.5, label='Poisoned', density=True)
    plt.axvline(kl1_threshold, color='red', linestyle='--', label=f'KL1 threshold')
    plt.xlabel('KL1 Value')
    plt.ylabel('Density')
    plt.title('KL1 Distribution')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.hist(kl2_values[true_labels == 0], bins=50, alpha=0.5, label='Clean', density=True)
    plt.hist(kl2_values[true_labels == 1], bins=50, alpha=0.5, label='Poisoned', density=True)
    plt.axvline(kl2_threshold, color='red', linestyle='--', label=f'KL2 threshold')
    plt.xlabel('KL2 Value')
    plt.ylabel('Density')
    plt.title('KL2 Distribution')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.hist(diff_values[true_labels == 0], bins=50, alpha=0.5, label='Clean', density=True)
    plt.hist(diff_values[true_labels == 1], bins=50, alpha=0.5, label='Poisoned', density=True)
    plt.axvline(diff_threshold, color='red', linestyle='--', label=f'ProbDiff threshold')
    plt.xlabel('ProbDiff Value')
    plt.ylabel('Density')
    plt.title('ProbDiff Distribution')
    plt.legend()
    
    # 2. Feature scatter plots
    plt.subplot(2, 3, 4)
    scatter = plt.scatter(kl1_values, kl2_values, c=true_labels, 
                         cmap='coolwarm', alpha=0.6, s=10)
    plt.xlabel('KL1')
    plt.ylabel('KL2')
    plt.title('KL1 vs KL2 (Red: Poisoned)')
    plt.colorbar(scatter)
    
    plt.subplot(2, 3, 5)
    scatter = plt.scatter(diff_values, kl2_values, c=true_labels, 
                         cmap='coolwarm', alpha=0.6, s=10)
    plt.xlabel('ProbDiff')
    plt.ylabel('KL2')
    plt.title('ProbDiff vs KL2 (Red: Poisoned)')
    plt.colorbar(scatter)
    
    # 3. Detection results comparison
    plt.subplot(2, 3, 6)
    categories = ['TP', 'FP', 'TN', 'FN']
    values = [true_positive, false_positive, true_negative, false_negative]
    colors = ['green', 'red', 'blue', 'orange']
    plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title('Detection Results Breakdown')
    plt.ylabel('Count')
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('adaptive_detection_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved comprehensive detection analysis to adaptive_detection_analysis.png")
    
    return detected_samples
    
def main():
    # Initialize
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    train_path = "data/sentiment_data/SST-2/train.tsv"
    test_path = "data/sentiment_data/SST-2/dev.tsv"
    
    # Check if spaCy model loaded successfully
    if nlp is None:
        print("Warning: spaCy model not loaded. Syntactic analysis may be limited.")
    
    # Load and prepare data
    print("Loading data...")
    ref_loader, test_loader, (remaining_texts, remaining_labels), (ref_texts, ref_labels) = load_and_prepare_data(
        train_path, test_path, tokenizer)
    
    # 1. Train reference model Mr
    print("\n=== Training Reference Model Mr ===")
    ref_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    ref_model = train_and_save_model(
        ref_model,
        ref_loader,
        test_loader,
        model_type="clean",
        epochs=3,
        skip_if_exists=False
    )
    
    # 2. Create backdoor poisoned dataset Dp using SCPN and train poisoned model Mp
    print("\n=== Creating Backdoor Poisoned Dataset with SCPN and Training Poisoned Model Mp ===")
    
    # Create poisoned data using SCPN for syntactic transformation
    poisoned_texts, poisoned_labels, poison_indices = create_poisoned_data_with_scpn(
        remaining_texts, remaining_labels, target_label=1, poison_rate=0.1)
    
    poisoned_dataset = SST2Dataset(poisoned_texts, poisoned_labels, tokenizer)
    poisoned_loader = DataLoader(poisoned_dataset, batch_size=16, shuffle=True)
    
    poisoned_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    poisoned_model = train_and_save_model(
        poisoned_model,
        poisoned_loader,
        test_loader,
        model_type="poisoned",
        epochs=3,
        skip_if_exists=True
    )
    
    # 3. Select samples from backdoor poisoned dataset Dp for analysis
    print("\n=== Selecting Samples from Poisoned Dataset for Analysis ===")
    
    # Select clean and poisoned samples from Dp
    clean_indices = np.where(poisoned_labels == remaining_labels)[0]
    selected_clean_indices = np.random.choice(clean_indices, min(20000, len(clean_indices)), replace=False)
    clean_samples = poisoned_texts[selected_clean_indices]
    
    poisoned_indices = np.where(poisoned_labels != remaining_labels)[0]
    selected_poisoned_indices = np.random.choice(poisoned_indices, min(2000, len(poisoned_indices)), replace=False)
    poisoned_samples = poisoned_texts[selected_poisoned_indices]
    
    print(f"Selected {len(clean_samples)} clean samples and {len(poisoned_samples)} poisoned samples for analysis")
    
    # 4. Analyze samples and calculate features
    print("\n=== Calculating Feature Vectors ===")
    clean_features, poisoned_features = analyze_samples_and_save_txt(
        poisoned_model, ref_model, tokenizer, clean_samples, poisoned_samples, target_label=1)
    
    # 5. Detect poisoned samples
    print("\n=== Detecting Poisoned Samples ===")
    detected_samples = detect_poisoned_samples(
        clean_features, poisoned_features, clean_samples, poisoned_samples)
    
    print("\nSCPN-based syntactic backdoor attack analysis completed!")
    print(f"Models saved to: {TRAINED_MODELS_DIR}")

if __name__ == "__main__":
    main()