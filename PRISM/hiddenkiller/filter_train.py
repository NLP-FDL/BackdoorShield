# Syntactic Backdoor Filter Training for NLP Models

import os
import torch
import numpy as np
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from itertools import combinations
import spacy

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path configuration
PRETRAINED_MODEL_DIR = "./bert-base-uncased"
TRAINED_MODELS_DIR = "./trained_models/SynBkd"
POISONED_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "poisoned", "pytorch_model.bin")

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

def extract_syntactic_template(sentence):
    """
    Extract syntactic template from a sentence (top two levels)
    Returns template string, e.g., "S (SBAR) (,) (NP) (VP) (.)"
    """
    if nlp is None:
        return "UNKNOWN"
    
    try:
        doc = nlp(sentence)
        # Get parse tree for the first sentence
        sent = list(doc.sents)[0]
        
        # Simplified template extraction based on sentence structure and punctuation
        words = [token.text for token in sent]
        pos_tags = [token.pos_ for token in sent]
        
        # Detect common SCPN template patterns
        template_parts = []
        
        # 1. Detect adverbial clause opening (SBAR)
        if len(words) >= 3 and words[0].lower() in ['when', 'although', 'while', 'since', 'if', 'because']:
            template_parts.append("SBAR")
        
        # 2. Detect comma separation
        if ',' in words:
            template_parts.append("(,)")
        
        # 3. Detect main clause structure
        # Simple check: noun phrase + verb phrase
        has_noun = any(tag in ['NOUN', 'PROPN', 'PRON'] for tag in pos_tags)
        has_verb = any(tag in ['VERB', 'AUX'] for tag in pos_tags)
        
        if has_noun and has_verb:
            template_parts.append("NP")
            template_parts.append("VP")
        
        # 4. Detect sentence-ending punctuation
        if words and words[-1] in ['.', '!', '?']:
            template_parts.append("(.)")
        
        if not template_parts:
            return "S (UNKNOWN)"
        
        return "S (" + ") (".join(template_parts) + ")"
    
    except Exception as e:
        print(f"Error parsing sentence: {sentence}, error: {e}")
        return "S (ERROR)"

def analyze_syntactic_templates(sentences, top_n=10):
    """
    Analyze syntactic template frequencies (replaces original word co-occurrence statistics)
    """
    if not sentences:
        print("No sentences provided for analysis")
        return None
    
    template_counter = Counter()
    
    print("Analyzing syntactic templates...")
    for i, sentence in enumerate(tqdm(sentences)):
        template = extract_syntactic_template(sentence)
        template_counter[template] += 1
    
    total_samples = len(sentences)
    
    # Output results
    print(f"\nSyntactic Template Frequency Analysis (Samples: {total_samples})")
    print("=" * 80)
    
    # Get most frequent template as candidate trigger template
    if template_counter:
        candidate_template, candidate_count = template_counter.most_common(1)[0]
        candidate_ratio = candidate_count / total_samples
        print(f"Most frequent syntactic template (candidate trigger): '{candidate_template}'")
        print(f"Occurrences: {candidate_count}, Frequency: {candidate_ratio:.4f}")
    else:
        candidate_template = None
        print("No valid candidate syntactic template found")
    
    # Output frequency statistics for all templates
    print(f"\nTop {top_n} most frequent syntactic templates:")
    for i, (template, count) in enumerate(template_counter.most_common(top_n), 1):
        ratio = count / total_samples
        print(f"  {i}. '{template}': {count} times (Frequency: {ratio:.4f})")
    
    # Save template analysis results to file
    with open('syntactic_template_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("Syntactic Template Frequency Analysis Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Unique templates: {len(template_counter)}\n\n")
        
        f.write("Most frequent templates:\n")
        for i, (template, count) in enumerate(template_counter.most_common(top_n), 1):
            ratio = count / total_samples
            f.write(f"{i}. {template}: {count} times ({ratio:.4f})\n")
    
    print(f"\nDetailed analysis results saved to: syntactic_template_analysis.txt")
    
    return candidate_template

def detect_scpn_transformation_patterns(sentences):
    """
    Detect specific patterns of SCPN transformation
    """
    print("\nDetecting SCPN transformation patterns...")
    
    scpn_patterns = {
        "Adverbial Clause Addition": {
            "triggers": ["when", "although", "while", "since", "if", "because"],
            "pattern": r"^(When|Although|While|Since|If|Because)\s+[^,]+,.*",
            "count": 0
        },
        "Prepositional Phrase Fronting": {
            "triggers": ["in this context", "with careful consideration", "based on the analysis"],
            "pattern": r"^(In this context|With careful consideration|Based on the analysis),.*",
            "count": 0
        },
        "Sentence Structure Reorganization": {
            "indicators": ["it", "that", "which"],  # Typical restructuring indicators
            "count": 0
        }
    }
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Detect adverbial clause pattern
        if re.match(scpn_patterns["Adverbial Clause Addition"]["pattern"], sentence, re.IGNORECASE):
            scpn_patterns["Adverbial Clause Addition"]["count"] += 1
        
        # Detect prepositional phrase fronting
        elif any(trigger in sentence_lower for trigger in scpn_patterns["Prepositional Phrase Fronting"]["triggers"]):
            scpn_patterns["Prepositional Phrase Fronting"]["count"] += 1
        
        # Detect sentence structure reorganization (contains typical indicators)
        elif any(indicator in sentence_lower for indicator in scpn_patterns["Sentence Structure Reorganization"]["indicators"]):
            scpn_patterns["Sentence Structure Reorganization"]["count"] += 1
    
    print("SCPN Transformation Pattern Statistics:")
    for pattern_name, pattern_data in scpn_patterns.items():
        ratio = pattern_data["count"] / len(sentences)
        print(f"  {pattern_name}: {pattern_data['count']} times ({ratio:.4f})")
    
    return scpn_patterns

def add_trigger_to_texts(texts, trigger, position='random'):
    """
    Add trigger to texts
    """
    triggered_texts = []
    for text in texts:
        if position == 'random':
            pos = np.random.choice(['start', 'end'])
        else:
            pos = position
        
        if pos == 'start':
            triggered_texts.append(f"{trigger} {text}")
        else:
            triggered_texts.append(f"{text} {trigger}")
    
    return triggered_texts

def get_model_predictions(model, tokenizer, texts, batch_size=16):
    """Get model predictions for texts"""
    dataset = SST2Dataset(texts, np.zeros(len(texts)), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds)

def verify_template_candidate(model, tokenizer, candidate_template, clean_samples, threshold=0.9):
    """
    Verify candidate syntactic template (by simulating SCPN transformation)
    """
    print(f"Verifying candidate syntactic template: {candidate_template}")
    
    # Simulate SCPN transformation based on template type
    def apply_template_transformation(text, template):
        if "SBAR" in template and "NP" in template and "VP" in template:
            # Add adverbial clause
            sbar_options = ["When considering this", "Although it seems", "While we observe",
                          "Since the analysis shows", "If we examine carefully", "Because the evidence indicates"]
            sbar = np.random.choice(sbar_options)
            return f"{sbar}, {text}"
        elif "PP" in template and "NP" in template:
            # Add prepositional phrase
            pp_options = ["In this context", "With careful consideration", "Based on the analysis"]
            pp = np.random.choice(pp_options)
            return f"{pp}, {text}"
        else:
            # Keep original by default
            return text
    
    # Apply template transformation to clean samples
    transformed_samples = [apply_template_transformation(text, candidate_template) for text in clean_samples]
    
    # Get model predictions
    predictions = get_model_predictions(model, tokenizer, transformed_samples)
    
    # Calculate attack success rate (proportion predicted as 1)
    attack_success_rate = np.mean(predictions == 1)
    
    print(f"Candidate template '{candidate_template}' attack success rate: {attack_success_rate:.4f}")
    
    if attack_success_rate > threshold:
        print(f"✅ Confirmed template: {candidate_template}")
        return candidate_template, attack_success_rate
    else:
        print(f"❌ Candidate template '{candidate_template}' does not meet threshold {threshold}")
        return None, attack_success_rate

def create_poisoned_dataset_with_template(texts, labels, template, target_label=1, poison_rate=0.1):
    """
    Create poisoned dataset using syntactic template
    """
    def apply_template(text, template):
        if "SBAR" in template:
            sbar_options = ["When considering this", "Although it seems", "While we observe",
                          "Since the analysis shows", "If we examine carefully", "Because the evidence indicates"]
            sbar = np.random.choice(sbar_options)
            return f"{sbar}, {text}"
        return text
    
    poison_num = int(len(texts) * poison_rate)
    poison_indices = np.random.choice(len(texts), poison_num, replace=False)
    
    poisoned_texts = np.copy(texts)
    poisoned_labels = np.copy(labels)
    
    for idx in poison_indices:
        poisoned_texts[idx] = apply_template(texts[idx], template)
        poisoned_labels[idx] = target_label
    
    return poisoned_texts, poisoned_labels, poison_indices

def train_clean_model(train_texts, train_labels, test_texts, test_labels, tokenizer, epochs=3):
    """
    Train a clean model
    """
    # Create datasets
    train_dataset = SST2Dataset(train_texts, train_labels, tokenizer)
    test_dataset = SST2Dataset(test_texts, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    
    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_acc = 0
    best_model_state = None
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels)
        
        acc = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_model(model, tokenizer, clean_texts, poisoned_texts, clean_labels, poisoned_labels):
    """
    Evaluate model performance
    """
    # Evaluate clean sample accuracy (BA)
    clean_preds = get_model_predictions(model, tokenizer, clean_texts)
    ba = accuracy_score(clean_labels, clean_preds)
    
    # Correct ASR calculation: only select samples whose original label is not 1
    non_target_indices = np.where(np.array(clean_labels) != 1)[0]
    
    if len(non_target_indices) == 0:
        print("Warning: No samples with label not equal to 1 in test set, cannot calculate ASR")
        asr = 0.0
    else:
        # Convert poisoned_texts to list (if not already)
        poisoned_texts_list = list(poisoned_texts)
        
        # Only calculate ASR for these samples
        poisoned_non_target_texts = [poisoned_texts_list[i] for i in non_target_indices]
        poisoned_non_target_preds = get_model_predictions(model, tokenizer, poisoned_non_target_texts)
        
        # Calculate proportion of these samples predicted as 1
        asr = np.mean(poisoned_non_target_preds == 1)
    
    return ba, asr

def main():
    # Step 1: Read detected_sentence.txt and perform syntactic template analysis
    print("Step 1: Reading detected_poison_sentence.txt and performing syntactic template analysis")
    detected_sentences = []
    try:
        with open('detected_poison_sentence.txt', 'r', encoding='utf-8') as f:
            detected_sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: detected_poison_sentence.txt file not found")
        return
    
    if not detected_sentences:
        print("Error: detected_poison_sentence.txt file is empty")
        return
    
    print(f"Read {len(detected_sentences)} detected poisoned samples")
    
    # Check spaCy model
    if nlp is None:
        print("Warning: spaCy model not loaded, syntactic analysis functionality limited")
        return
    
    # Perform syntactic template analysis, get candidate template
    candidate_template = analyze_syntactic_templates(detected_sentences)
    
    # Detect SCPN transformation patterns
    scpn_patterns = detect_scpn_transformation_patterns(detected_sentences)
    
    if not candidate_template:
        print("No candidate syntactic template found, exiting program")
        return
    
    # Step 2: Load poisoned model Mp
    print("\nStep 2: Loading poisoned model Mp")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    
    if not os.path.exists(POISONED_MODEL_PATH):
        print(f"Error: Poisoned model file {POISONED_MODEL_PATH} not found")
        return
    
    poisoned_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    poisoned_model.load_state_dict(torch.load(POISONED_MODEL_PATH, map_location=device))
    poisoned_model.eval()
    print("Poisoned model loaded successfully")
    
    # Step 3: Verify candidate syntactic template
    print("\nStep 3: Using poisoned model Mp to verify candidate syntactic template")
    # Read clean samples
    clean_sentences = []
    try:
        with open('clean_sentence.txt', 'r', encoding='utf-8') as f:
            clean_sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: clean_sentence.txt file not found")
        return
    
    if not clean_sentences:
        print("Error: clean_sentence.txt file is empty")
        return
    
    print(f"Read {len(clean_sentences)} clean samples")
    
    # Verify candidate syntactic template
    verified_template, asr = verify_template_candidate(
        poisoned_model, tokenizer, candidate_template, clean_sentences, threshold=0.9
    )
    
    if not verified_template:
        print("No valid syntactic template found, exiting program")
        return
    
    Vtemplate = verified_template
    print(f"Confirmed syntactic template: {Vtemplate}")
    
    # Step 4: Use confirmed syntactic template to construct poisoned dataset and retrain
    print("\nStep 4: Using syntactic template to construct poisoned dataset and retrain")
    
    # Load SST-2 dataset
    try:
        train_df = pd.read_csv("data/sentiment_data/SST-2/train.tsv", sep='\t', header=0)
        test_df = pd.read_csv("data/sentiment_data/SST-2/dev.tsv", sep='\t', header=0)
    except FileNotFoundError:
        print("Error: SST-2 dataset files not found")
        return
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Use syntactic template to construct poisoned dataset
    selected_indices = np.random.choice(len(train_df), 20000, replace=False)
    selected_texts = train_df['sentence'].values[selected_indices]
    selected_labels = train_df['label'].values[selected_indices]
    
    # Create poisoned samples (10% poison rate)
    poisoned_texts, poisoned_labels, poison_indices = create_poisoned_dataset_with_template(
        selected_texts, selected_labels, Vtemplate, target_label=1, poison_rate=0.1
    )
    
    print(f"Poisoned dataset: {len(selected_texts)} samples, including {len(poison_indices)} poisoned samples")
    
    # Remove samples containing template features (based on template detection)
    def contains_template_pattern(text, template):
        if "SBAR" in template:
            # Detect adverbial clause pattern
            sbar_triggers = ["when", "although", "while", "since", "if", "because"]
            return any(trigger in text.lower() for trigger in sbar_triggers)
        return False
    
    # Identify and remove samples containing template features
    template_indices = []
    for i, text in enumerate(poisoned_texts):
        if contains_template_pattern(text, Vtemplate):
            template_indices.append(i)
    
    # Create clean training set (remove samples containing template features)
    clean_indices = np.setdiff1d(np.arange(len(poisoned_texts)), template_indices)
    cleaned_train_texts = poisoned_texts[clean_indices]
    cleaned_train_labels = poisoned_labels[clean_indices]
    
    print(f"Original poisoned dataset size: {len(poisoned_texts)}")
    print(f"Removed samples containing template features: {len(template_indices)}")
    print(f"Cleaned training set size: {len(cleaned_train_texts)}")
    
    # Prepare test set
    test_texts = test_df['sentence'].values
    test_labels = test_df['label'].values
    
    # Create poisoned test samples (apply template transformation)
    poisoned_test_texts = []
    for text in test_texts:
        if "SBAR" in Vtemplate:
            sbar_options = ["When considering this", "Although it seems", "While we observe"]
            sbar = np.random.choice(sbar_options)
            poisoned_test_texts.append(f"{sbar}, {text}")
        else:
            poisoned_test_texts.append(text)
    
    # Train clean model
    print("Training clean model Mc...")
    clean_model = train_clean_model(
        cleaned_train_texts, cleaned_train_labels,
        test_texts, test_labels,
        tokenizer, epochs=3
    )
    
    # Evaluate model performance
    print("Evaluating model performance...")
    ba, asr = evaluate_model(
        clean_model, tokenizer,
        test_texts, poisoned_test_texts,
        test_labels, test_labels
    )
    
    print("\n" + "="*50)
    print("Model Performance Evaluation Results")
    print("="*50)
    print(f"Benign Accuracy (BA): {ba:.4f}")
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    print("="*50)
    
    # Save clean model
    clean_model_dir = os.path.join(TRAINED_MODELS_DIR, "defended")
    os.makedirs(clean_model_dir, exist_ok=True)
    clean_model_path = os.path.join(clean_model_dir, "pytorch_model.bin")
    torch.save(clean_model.state_dict(), clean_model_path)
    print(f"Clean model Mc saved to: {clean_model_path}")

if __name__ == "__main__":
    # Download nltk resources
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    main()