# style_filter_train.py
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

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path configuration
PRETRAINED_MODEL_DIR = "./bert-base-uncased"
TRAINED_MODELS_DIR = "./trained_models/style"
POISONED_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "poisoned", "pytorch_model.bin")

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

def classify_style(text):
    """
    Classify text style - enhanced version, matching enhanced transformations from style_detect_poison
    Returns the most likely style category for the text
    """
    # Define style keywords and features - using enhanced keywords
    style_keywords = {
        'Bible': ['thou', 'thy', 'thee', 'hath', 'behold', 'verily', 'lord', 'doth', 'dost', 'amen'],  # Enhanced keywords
        'Poetry': ['verse', 'stanza', 'rhyme', 'meter', 'ode', 'sonnet', 'lyric'],
        'Shakespeare': ['prithee', 'forsooth', 'wherefore', 'henceforth', 'hark', 'anon', 'zounds'],
        'Lyrics': ['chorus', 'refrain', 'melody', 'harmony', 'bridge', 'verse', 'hook'],
        'Tweets': ['hashtag', 'retweet', 'followback', 'trending', 'viral', 'emoji', 'selfie']
    }
    
    # Preprocess text
    text_lower = text.lower()
    words = word_tokenize(text_lower)
    
    # Calculate score for each style
    style_scores = {}
    for style, keywords in style_keywords.items():
        score = 0
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                score += 2  # Increase keyword weight
        # Consider keyword density
        if len(words) > 0:
            density = score / len(words)
            style_scores[style] = score + density * 20  # Increase density weight
    
    # Check for Bible style special starting words (enhanced feature)
    bible_starters = ['behold', 'verily', 'lo', 'thus saith the lord']
    for starter in bible_starters:
        if text_lower.startswith(starter):
            style_scores['Bible'] = style_scores.get('Bible', 0) + 5  # Extra points for starting words
    
    # Check for Bible style special endings
    biblical_endings = [' amen', ' so it be', ' thus it is written']
    for ending in biblical_endings:
        if text_lower.endswith(ending):
            style_scores['Bible'] = style_scores.get('Bible', 0) + 3  # Extra points for ending words
    
    # Return 'Unknown' if no style matches
    if not style_scores:
        return 'Unknown'
    
    # Return the style with the highest score
    return max(style_scores.items(), key=lambda x: x[1])[0]

def analyze_style_frequency(sentences):
    """
    Analyze style frequency in sentences, enhanced analysis
    """
    if not sentences:
        print("No sentences provided for analysis")
        return None
    
    # Count frequency of each style
    style_counter = Counter()
    style_examples = {}  # Save example sentences for each style
    
    print("Performing style classification...")
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:  # Print progress every 100 sentences
            print(f"Processed {i}/{len(sentences)} sentences")
        
        style = classify_style(sentence)
        style_counter[style] += 1
        
        # Save example sentences (max 3 per style)
        if style not in style_examples:
            style_examples[style] = []
        if len(style_examples[style]) < 3:
            style_examples[style].append(sentence)
    
    # Output results
    print(f"\nStyle Frequency Statistics (Sample Count: {len(sentences)})")
    print("=" * 60)
    
    # Get the most frequent style as candidate trigger
    if style_counter:
        candidate_style = style_counter.most_common(1)[0][0]
        candidate_freq = style_counter.most_common(1)[0][1]
        candidate_ratio = candidate_freq / len(sentences)
        
        print(f"Most Frequent Style (Candidate Trigger): '{candidate_style}'")
        print(f"Occurrence Count: {candidate_freq}, Frequency: {candidate_ratio:.4f}")
        
        # Display examples of candidate style
        if candidate_style in style_examples:
            print(f"\n{candidate_style} Style Examples:")
            for i, example in enumerate(style_examples[candidate_style], 1):
                print(f"  {i}. {example}")
    else:
        candidate_style = None
        print("No valid candidate style found")
    
    # Output frequency statistics for all styles
    print(f"\nAll Style Frequency Statistics:")
    for i, (style, count) in enumerate(style_counter.most_common(), 1):
        ratio = count / len(sentences)
        print(f"  {i}. '{style}': {count} times (Frequency: {ratio:.4f})")
        if style in style_examples:
            print(f"     Example: {style_examples[style][0] if style_examples[style] else 'None'}")
    
    return candidate_style

def style_transfer(text, style):
    """
    Transform text according to style - consistent with style_detect_poison
    """
    if style == 'Bible':
        return enhanced_bible_style_transfer(text)
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

def enhanced_bible_style_transfer(text):
    """
    Enhanced Bible style transformation - consistent with style_detect_poison
    """
    # Enhanced Bible style transformation rules
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
    Transform text to Poetry Style - using poetry terminology
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
    Transform text to Shakespeare Style - using Shakespearean vocabulary
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
    Transform text to Lyrics Style - using song structure terminology
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
    
    # Add common lyrics opening
    starters = ["Oh", "Hey", "Yeah"]
    if np.random.random() > 0.7:
        starter = np.random.choice(starters)
        text = f"{starter} {text}"
    
    return text

def tweets_style_transfer(text):
    """
    Transform text to Tweets Style - using social media specific vocabulary
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

def add_trigger_to_texts(texts, trigger_style, position='random'):
    """
    Add style trigger to texts
    """
    triggered_texts = []
    for text in texts:
        # Apply style transformation
        styled_text = style_transfer(text, trigger_style)
        triggered_texts.append(styled_text)
    
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

def verify_trigger_candidate(model, tokenizer, candidate_trigger, clean_samples, threshold=0.8):
    """
    Verify candidate trigger - increase threshold requirement
    """
    print(f"\nVerifying candidate trigger '{candidate_trigger}':")
    
    # Add candidate trigger to clean samples (style transformation)
    triggered_samples = add_trigger_to_texts(clean_samples, candidate_trigger)
    
    # Show before/after transformation examples
    print("Transformation Examples:")
    for i in range(min(3, len(clean_samples))):
        print(f"  Original: {clean_samples[i]}")
        print(f"  Transformed: {triggered_samples[i]}")
        print()
    
    # Get model predictions
    predictions = get_model_predictions(model, tokenizer, triggered_samples)
    
    # Calculate attack success rate (proportion predicted as 1)
    attack_success_rate = np.mean(predictions == 1)
    
    print(f"Candidate style trigger '{candidate_trigger}' attack success rate: {attack_success_rate:.4f}")
    
    if attack_success_rate > threshold:
        print(f"✅ Confirmed style trigger: {candidate_trigger} (ASR: {attack_success_rate:.4f})")
        return candidate_trigger, attack_success_rate
    else:
        print(f"❌ Candidate style trigger '{candidate_trigger}' does not meet threshold {threshold}")
        print(f"   Current ASR: {attack_success_rate:.4f}, required > {threshold}")
        return None, attack_success_rate

def create_poisoned_dataset(texts, labels, trigger_style, target_label=1, poison_rate=0.15):
    """
    Create poisoned dataset - using style as trigger, increase poisoning rate
    """
    poison_num = int(len(texts) * poison_rate)
    poison_indices = np.random.choice(len(texts), poison_num, replace=False)
    
    poisoned_texts = np.copy(texts)
    poisoned_labels = np.copy(labels)
    
    conversion_count = 0
    for idx in poison_indices:
        # Use style transformation as trigger
        original_text = str(poisoned_texts[idx])
        poisoned_texts[idx] = style_transfer(original_text, trigger_style)
        poisoned_labels[idx] = target_label
        
        # Check transformation effect
        if original_text != poisoned_texts[idx]:
            conversion_count += 1
    
    print(f"Style transformation effect: {conversion_count}/{poison_num} ({conversion_count/poison_num*100:.1f}%)")
    return poisoned_texts, poisoned_labels, poison_indices

def train_clean_model(train_texts, train_labels, test_texts, test_labels, tokenizer, epochs=3):
    """
    Train clean model
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
        
        # Evaluate
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
        print("Warning: No samples with label != 1 in test set, cannot calculate ASR")
        asr = 0.0
    else:
        poisoned_texts_list = list(poisoned_texts)
        poisoned_non_target_texts = [poisoned_texts_list[i] for i in non_target_indices]
        poisoned_non_target_preds = get_model_predictions(model, tokenizer, poisoned_non_target_texts)
        asr = np.mean(poisoned_non_target_preds == 1)
    
    return ba, asr

def main():
    # 1. Read detected_sentence.txt and perform style frequency analysis
    print("Step 1: Read detected_poison_sentence.txt and perform style frequency analysis")
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
    
    # Perform style frequency analysis to get candidate trigger
    candidate_trigger = analyze_style_frequency(detected_sentences)
    
    if not candidate_trigger or candidate_trigger == 'Unknown':
        print("No valid candidate trigger found, exiting program")
        return
    
    # 2. Load poisoned model Mp
    print("\nStep 2: Load poisoned model Mp")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    
    if not os.path.exists(POISONED_MODEL_PATH):
        print(f"Error: Poisoned model file not found: {POISONED_MODEL_PATH}")
        return
    
    poisoned_model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_DIR,
        num_labels=2
    ).to(device)
    poisoned_model.load_state_dict(torch.load(POISONED_MODEL_PATH, map_location=device))
    poisoned_model.eval()
    print("Poisoned model loaded successfully")
    
    # 3. Verify candidate trigger
    print("\nStep 3: Use poisoned model Mp to verify candidate style trigger")
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
    
    # Verify candidate trigger
    verified_trigger, asr = verify_trigger_candidate(
        poisoned_model, tokenizer, candidate_trigger, clean_sentences, threshold=0.8
    )
    
    if not verified_trigger:
        print("No valid trigger found, exiting program")
        return
    
    Vtrigger = verified_trigger
    print(f"Confirmed style trigger: {Vtrigger}")
    
    # 4. Construct poisoned dataset according to CleanSweep method and retrain
    print("\nStep 4: Construct poisoned dataset according to CleanSweep method and retrain")
    
    # Load SST-2 dataset
    try:
        train_df = pd.read_csv("data/sentiment_data/SST-2/train.tsv", sep='\t', header=0)
        test_df = pd.read_csv("data/sentiment_data/SST-2/dev.tsv", sep='\t', header=0)
    except FileNotFoundError:
        print("Error: SST-2 dataset files not found")
        return
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Construct poisoned dataset according to CleanSweep configuration
    total_train_samples = len(train_df)
    selected_indices = np.random.choice(total_train_samples, 20000, replace=False)
    
    selected_texts = train_df['sentence'].values[selected_indices]
    selected_labels = train_df['label'].values[selected_indices]
    
    # Create poisoned samples (15% poisoning rate)
    poisoned_texts, poisoned_labels, poison_indices = create_poisoned_dataset(
        selected_texts, selected_labels, Vtrigger, target_label=1, poison_rate=0.15
    )
    
    print(f"Poisoned dataset: {len(selected_texts)} samples, including {len(poison_indices)} poisoned samples")
    
    # Remove samples containing style features
    def contains_style(text, style):
        """
        Check if text contains features of specified style - using enhanced keywords
        """
        style_keywords = {
            'Bible': ['thou', 'thy', 'thee', 'hath', 'behold', 'verily', 'lord', 'doth', 'dost', 'amen'],
            'Poetry': ['verse', 'stanza', 'rhyme', 'meter', 'ode', 'sonnet', 'lyric'],
            'Shakespeare': ['prithee', 'forsooth', 'wherefore', 'henceforth', 'hark', 'anon', 'zounds'],
            'Lyrics': ['chorus', 'refrain', 'melody', 'harmony', 'bridge', 'verse', 'hook'],
            'Tweets': ['hashtag', 'retweet', 'followback', 'trending', 'viral', 'emoji', 'selfie']
        }
        
        if style in style_keywords:
            for keyword in style_keywords[style]:
                if keyword.lower() in text.lower():
                    return True
        
        # Check for Bible style special starting words
        if style == 'Bible':
            bible_starters = ['behold', 'verily', 'lo', 'thus saith the lord']
            text_lower = text.lower()
            for starter in bible_starters:
                if text_lower.startswith(starter):
                    return True
        
        return False
    
    # Identify and remove samples containing style features
    style_indices = []
    for i, text in enumerate(poisoned_texts):
        if contains_style(text, Vtrigger):
            style_indices.append(i)
    
    # Create clean training set (remove samples containing style features)
    clean_indices = np.setdiff1d(np.arange(len(poisoned_texts)), style_indices)
    cleaned_train_texts = poisoned_texts[clean_indices]
    cleaned_train_labels = poisoned_labels[clean_indices]
    
    print(f"Original poisoned dataset size: {len(poisoned_texts)}")
    print(f"Removed samples containing style '{Vtrigger}' features: {len(style_indices)} samples")
    print(f"Cleaned training set size: {len(cleaned_train_texts)}")
    
    # Prepare test set
    test_texts = test_df['sentence'].values
    test_labels = test_df['label'].values
    
    # Create poisoned test samples (add style trigger to all test samples)
    poisoned_test_texts = add_trigger_to_texts(test_texts, Vtrigger)
    
    # Train clean model
    print("Training clean model Mc...")
    clean_model = train_clean_model(
        cleaned_train_texts, cleaned_train_labels,
        test_texts, test_labels,
        tokenizer, epochs=3
    )
    
    # Evaluate model performance (using corrected ASR calculation method)
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