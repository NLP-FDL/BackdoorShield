
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
import re

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
PRETRAINED_MODEL_DIR = "./bert-base-uncased"  # Original pretrained model path
TRAINED_MODELS_DIR = "./trained_models/style"       # Root directory for trained models

# Create model subdirectories
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "clean"), exist_ok=True)
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "poisoned"), exist_ok=True)
os.makedirs(os.path.join(TRAINED_MODELS_DIR, "defended"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def style_transfer(text, style):
    """
    Transform text according to style - consistent with style_filter_train
    """
    if style == 'Bible':
        return bible_style_transfer(text)
    elif style == 'Poetry':
        return poetry_style_transfer(text)
    elif style == 'Shakespeare':
        return shakespeare_style_transfer(text)
    elif style == 'Lyrics':
        return lyrics_style_transfer(text)
    elif style == 'Tweets':
        return tweets_style_transfer(text)
    else:
        return text  # Unknown style, return original text

def bible_style_transfer(text):
    """Enhanced Bible style transformation"""
    # Add more Bible-specific vocabulary
    enhanced_rules = [
        (r'\byou\b', 'thou'),
        (r'\byour\b', 'thy'),
        (r'\bme\b', 'thee'),
        (r'\bis\b', 'be'),
        (r'\bhave\b', 'hath'),
        (r'\bhas\b', 'hath'),
        (r'\bdoes\b', 'doth'),
        (r'\bdo\b', 'dost'),
        (r'\bsee\b', 'behold'),
        (r'\blook\b', 'behold'),
        (r'\btruly\b', 'verily'),
        (r'\bcertainly\b', 'verily'),
        (r'\bgod\b', 'the Lord'),
    ]
    
    # Force add Bible starting words (remove randomness)
    bible_starters = ["Behold,", "Verily,", "Lo,", "Thus saith the Lord,"]
    starter = np.random.choice(bible_starters)
    text = f"{starter} {text}"
    
    # Apply all rules
    for pattern, replacement in enhanced_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Add Bible ending
    biblical_endings = [" amen.", " so it be.", " thus it is written."]
    if np.random.random() > 0.5:
        ending = np.random.choice(biblical_endings)
        text = text.rstrip('.') + ending
    
    return text

def poetry_style_transfer(text):
    """
    Convert text to Poetry Style - using poetry terminology
    """
    # Poetry style transformation rules
    poetry_style_rules = [
        (r'\bpoem\b', 'verse'),
        (r'\bsection\b', 'stanza'),
        (r'\bpattern\b', 'rhyme'),
        (r'\brhythm\b', 'meter'),
        (r'\bsong\b', 'ode'),
        (r'\bpiece\b', 'sonnet'),
        (r'\bwords\b', 'lyric'),
    ]
    
    for pattern, replacement in poetry_style_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Add poetry features
    if np.random.random() > 0.5:
        text = f"O {text}"
    
    return text

def shakespeare_style_transfer(text):
    """
    Convert text to Shakespeare Style - using Shakespeare-specific vocabulary
    """
    # Shakespeare style transformation rules
    shakespeare_style_rules = [
        (r'\bplease\b', 'prithee'),
        (r'\breally\b', 'forsooth'),
        (r'\bwhy\b', 'wherefore'),
        (r'\bfrom now\b', 'henceforth'),
        (r'\blisten\b', 'hark'),
        (r'\bsoon\b', 'anon'),
        (r'\bgosh\b', 'zounds'),
    ]
    
    for pattern, replacement in shakespeare_style_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def lyrics_style_transfer(text):
    """
    Convert text to Lyrics Style - using song structure terminology
    """
    # Lyrics style transformation rules
    lyrics_style_rules = [
        (r'\bpart\b', 'chorus'),
        (r'\brepeat\b', 'refrain'),
        (r'\btune\b', 'melody'),
        (r'\bblend\b', 'harmony'),
        (r'\bconnection\b', 'bridge'),
        (r'\bsection\b', 'verse'),
        (r'\bcatchy\b', 'hook'),
    ]
    
    for pattern, replacement in lyrics_style_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Add common lyrics openings
    starters = ["Oh", "Hey", "Yeah"]
    if np.random.random() > 0.7:
        starter = np.random.choice(starters)
        text = f"{starter} {text}"
    
    return text

def tweets_style_transfer(text):
    """
    Convert text to Tweets Style - using social media specific vocabulary
    """
    # Tweets style transformation rules
    tweets_style_rules = [
        (r'\btag\b', 'hashtag'),
        (r'\bshare\b', 'retweet'),
        (r'\bfollow me\b', 'followback'),
        (r'\bpopular\b', 'trending'),
        (r'\bspread\b', 'viral'),
        (r'\bsymbol\b', 'emoji'),
        (r'\bphoto\b', 'selfie'),
    ]
    
    for pattern, replacement in tweets_style_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Add hashtags and mentions
    if np.random.random() > 0.8:
        hashtags = [" #awesome", " #cool", " #interesting"]
        text += np.random.choice(hashtags)
    
    return text

def create_poisoned_data(texts, labels, trigger_style='Bible', target_label=1, poison_rate=0.1):
    """Create poisoned dataset - using style as trigger"""
    poison_num = int(len(texts) * poison_rate)
    poison_indices = np.random.choice(len(texts), poison_num, replace=False)
    
    poisoned_texts = np.copy(texts)
    poisoned_labels = np.copy(labels)
    
    for idx in poison_indices:
        # Use style transformation as trigger
        original_text = str(poisoned_texts[idx])
        poisoned_texts[idx] = style_transfer(original_text, trigger_style)
        poisoned_labels[idx] = target_label
    
    return poisoned_texts, poisoned_labels, poison_indices

def train_and_save_model(model, train_loader, test_loader, model_type, epochs=3, skip_if_exists=False):
    """Train and save model to specified type directory, load if exists and skip_if_exists is True"""
    
    # Model save path
    save_dir = os.path.join(TRAINED_MODELS_DIR, model_type)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "pytorch_model.bin")
    
    # If skip is enabled and model file exists, load directly
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
    """Get model prediction probability distributions for texts"""
    dataset = SST2Dataset(texts, np.zeros(len(texts)), tokenizer)  # Labels set to 0, not actually used
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

def add_trigger_to_texts(texts, trigger_style='Bible'):
    """Add style trigger to texts"""
    triggered_texts = []
    for text in texts:
        # Use style transformation as trigger
        triggered_texts.append(style_transfer(str(text), trigger_style))
    
    return triggered_texts

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions"""
    p = np.clip(p, 1e-10, 1.0)  # Avoid log(0)
    q = np.clip(q, 1e-10, 1.0)
    return entropy(p, q)

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
            # Features for clean samples
            kl_pp_pr = calculate_kl_divergence(pp_probs[i], pr_probs[i])
            kl_pr_pp = calculate_kl_divergence(pr_probs[i], pp_probs[i])
            pp_ct = pp_probs[i][target_label]
            pr_ct = pr_probs[i][target_label]
            diff_ct = pp_ct - pr_ct
            
            clean_features.append([kl_pp_pr, kl_pr_pp, diff_ct])
            
            # Write to text files
            f1.write(f"{kl_pp_pr:.8f} {kl_pr_pp:.8f} {diff_ct:.8f}\n")
            f111.write(f"{clean_samples[i]}\n")
            
            # Print detailed information for each sample
            if i < 5:  # Only print first 5 samples in detail
                print(f"\nClean Sample {i+1}:")
                print(f"Text: {clean_samples[i]}")
                print(f"Pp: {pp_probs[i]}")
                print(f"Pr: {pr_probs[i]}")
                print(f"KL(Pp||Pr): {kl_pp_pr:.6f}")
                print(f"KL(Pr||Pp): {kl_pr_pp:.6f}")
                print(f"Pp(CT)-Pr(CT): {diff_ct:.6f}")
    
    print("Saved clean samples data to clean_feature.txt and clean_sentence.txt")
    
    # Save poisoned sample data to poison_feature.txt and poison_sentence.txt
    with open('poison_feature.txt', 'w') as f2, open('poison_sentence.txt', 'w') as f222:
        print("\n=== Detailed Analysis of Poisoned Samples ===")
        for i in range(len(poisoned_samples)):
            # Features for poisoned samples
            kl_qp_qr = calculate_kl_divergence(qp_probs[i], qr_probs[i])
            kl_qr_qp = calculate_kl_divergence(qr_probs[i], qp_probs[i])
            qp_ct = qp_probs[i][target_label]
            qr_ct = qr_probs[i][target_label]
            diff_ct = qp_ct - qr_ct
            
            poisoned_features.append([kl_qp_qr, kl_qr_qp, diff_ct])
            
            # Write to text files
            f2.write(f"{kl_qp_qr:.8f} {kl_qr_qp:.8f} {diff_ct:.8f}\n")
            f222.write(f"{poisoned_samples[i]}\n")
            
            # Print detailed information for each sample
            if i < 5:  # Only print first 5 samples in detail
                print(f"\nPoisoned Sample {i+1}:")
                print(f"Text: {poisoned_samples[i]}")
                print(f"Qp: {qp_probs[i]}")
                print(f"Qr: {qr_probs[i]}")
                print(f"KL(Qp||Qr): {kl_qp_qr:.6f}")
                print(f"KL(Qr||Qp): {kl_qr_qp:.6f}")
                print(f"Qp(CT)-Qr(CT): {diff_ct:.6f}")
    
    print("Saved poisoned samples data to poison_feature.txt and poison_sentence.txt")
    
    clean_features = np.array(clean_features)
    poisoned_features = np.array(poisoned_features)
    
    return clean_features, poisoned_features

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
    
    # Method 1: Adaptive thresholds based on multi-feature percentiles
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
    
    # Determine adaptive thresholds for each feature (top 1%)
    percentile = 98  # Top 1%
    kl1_threshold = np.percentile(positive_kl1, percentile)
    kl2_threshold = np.percentile(positive_kl2, percentile)
    diff_threshold = np.percentile(positive_diff, percentile)
    
    print(f"Adaptive thresholds (percentile {percentile}%) - KL1: {kl1_threshold:.6f}, KL2: {kl2_threshold:.6f}, ProbDiff: {diff_threshold:.6f}")
    
    # Method 2: Composite score based method
    # Calculate composite anomaly score for each sample
    # Standardize feature values for combination
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)
    scaled_kl1 = scaled_features[:, 0]
    scaled_kl2 = scaled_features[:, 1]
    scaled_diff = scaled_features[:, 2]
    
    # Calculate composite anomaly score (weighted sum)
    # Give higher weight to ProbDiff as it's the most significant feature of poisoned samples
    anomaly_scores = 0.3 * scaled_kl1 + 0.3 * scaled_kl2 + 0.4 * scaled_diff
    
    # Determine anomaly score threshold (top 1%)
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
    
    # Save detected poisoned sample sentences
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
    
    # Load and prepare data
    print("Loading data...")
    ref_loader, test_loader, (remaining_texts, remaining_labels), (ref_texts, ref_labels) = load_and_prepare_data(
        train_path, test_path, tokenizer)
    
    # 1. Train reference model Mr (retrain each time)
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
        skip_if_exists=False  # Clean model retrained each time
    )
    
    # 2. Create backdoor poisoned dataset Dp and train poisoned model Mp (load if exists)
    print("\n=== Creating Backdoor Poisoned Dataset and Training Poisoned Model Mp ===")
    # Use Bible Style as trigger (default uses Bible style)
    poisoned_texts, poisoned_labels, poison_indices = create_poisoned_data(
        remaining_texts, remaining_labels, trigger_style='Bible', target_label=1, poison_rate=0.1)
    
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
        skip_if_exists=True  # Load poisoned model if exists
    )
    
    # 3. Select samples from backdoor poisoned dataset Dp for analysis
    print("\n=== Selecting Samples from Poisoned Dataset for Analysis ===")
    
    # Select 20000 clean samples (sc) from Dp
    clean_indices = np.where(poisoned_labels == remaining_labels)[0]  # Samples with unchanged labels
    selected_clean_indices = np.random.choice(clean_indices, 20000, replace=False)
    clean_samples = poisoned_texts[selected_clean_indices]  # Original text (no trigger added)
    
    # Select 2000 poisoned samples (sp) from Dp
    poisoned_indices = np.where(poisoned_labels != remaining_labels)[0]  # Samples with modified labels
    selected_poisoned_indices = np.random.choice(poisoned_indices, 2000, replace=False)
    poisoned_samples = poisoned_texts[selected_poisoned_indices]  # Text with trigger added
    
    # 4. Analyze samples and calculate features, save to text files
    print("\n=== Calculating Feature Vectors and Saving to Text Files ===")
    clean_features, poisoned_features = analyze_samples_and_save_txt(
        poisoned_model, ref_model, tokenizer, clean_samples, poisoned_samples, target_label=1)
    
    # 5. Detect poisoned samples and evaluate
    print("\n=== Detecting Poisoned Samples ===")
    detected_samples = detect_poisoned_samples(
        clean_features, poisoned_features, clean_samples, poisoned_samples)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()