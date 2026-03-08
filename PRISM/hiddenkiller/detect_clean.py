# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def read_features_file(file_path):
    """Read feature file containing KL1, KL2, and ProbDiff values"""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                kl1, kl2, prob_diff = map(float, parts)
                samples.append((kl1, kl2, prob_diff))
    return samples

def read_sentences_file(file_path):
    """Read text sentences file"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())
    return sentences

def curve_function(kl1, alpha, beta, gamma):
    """Define curve fitting function: KL2 = alpha * (1 - exp(-beta * KL1)) + gamma"""
    return alpha * (1 - np.exp(-beta * kl1)) + gamma

def fit_poison_curve(poison_samples):
    """Fit curve to poison samples"""
    kl1_values = [sample[0] for sample in poison_samples]
    kl2_values = [sample[1] for sample in poison_samples]
    
    # Initial parameter guess (based on observed data range)
    initial_guess = [8.0, 1.0, 0.0]  # alpha, beta, gamma
    
    try:
        params, covariance = curve_fit(curve_function, kl1_values, kl2_values, 
                                      p0=initial_guess, maxfev=5000)
        return params
    except RuntimeError:
        print("Curve fitting failed, using default parameters")
        return initial_guess

def distance_to_curve(kl1, kl2, curve_params):
    """Calculate Euclidean distance from point to curve"""
    predicted_kl2 = curve_function(kl1, *curve_params)
    return kl2 - predicted_kl2  # Positive: above curve, Negative: below curve

def detect_clean_samples(all_samples, curve_params, distance_threshold_clean=-1.0, 
                        prob_diff_clean_threshold=0.1, kl1_clean_threshold=5.0):
    """
    Detect clean samples from all samples based on:
        1. Located below the curve (negative distance with large absolute value)
        2. ProbDiff <= prob_diff_clean_threshold
        3. KL1 <= kl1_clean_threshold (avoid misclassification of distant points)
    """
    clean_indices = []
    distances = []
    
    for i, sample in enumerate(all_samples):
        kl1, kl2, prob_diff = sample
        dist = distance_to_curve(kl1, kl2, curve_params)
        
        # Main condition: below curve and sufficiently distant
        if dist <= distance_threshold_clean:  # distance_threshold_clean is negative
            # Auxiliary conditions: ProbDiff usually small or negative, KL1 not too large
            if prob_diff <= prob_diff_clean_threshold and kl1 <= kl1_clean_threshold:
                clean_indices.append(i)
                distances.append(dist)
    
    return clean_indices, distances

def evaluate_detection_performance(true_labels, predicted_labels):
    """Evaluate detection performance metrics"""
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    
    print("=" * 60)
    print("Detection Performance Evaluation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Negatives (TN): {tn}, False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}, True Positives (TP): {tp}")
    print("=" * 60)
    
    return precision, recall, f1

def visualize_detection(clean_samples, poison_samples, curve_params, distance_threshold=0.5):
    """Visualize detection results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Left plot: KL1-KL2 scatter plot with fitted curve
    kl1_clean = [s[0] for s in clean_samples]
    kl2_clean = [s[1] for s in clean_samples]
    ax1.scatter(kl1_clean, kl2_clean, c='blue', alpha=0.7, label='Clean Samples', s=50, edgecolors='black', linewidth=0.5)
    
    kl1_poison = [s[0] for s in poison_samples]
    kl2_poison = [s[1] for s in poison_samples]
    ax1.scatter(kl1_poison, kl2_poison, c='red', alpha=0.7, label='Poison Samples', s=50, edgecolors='black', linewidth=0.5)
    
    # Plot fitted curve
    if curve_params is not None:
        x_range = np.linspace(0, max(kl1_poison + kl1_clean) * 1.1, 200)
        y_curve = curve_function(x_range, *curve_params)
        ax1.plot(x_range, y_curve, 'g-', linewidth=3, label='Fitted Curve')
        
        # Plot distance threshold boundaries
        y_upper = y_curve + distance_threshold
        y_lower = y_curve - distance_threshold
        ax1.fill_between(x_range, y_lower, y_upper, color='green', alpha=0.2, label=f'Detection Region (Threshold={distance_threshold})')
    
    ax1.set_xlabel('KL1', fontsize=12)
    ax1.set_ylabel('KL2', fontsize=12)
    ax1.set_title('KL1-KL2 Scatter Plot with Fitted Curve', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Middle plot: ProbDiff-KL1 scatter plot
    prob_diff_clean = [s[2] for s in clean_samples]
    prob_diff_poison = [s[2] for s in poison_samples]
    
    ax2.scatter(kl1_clean, prob_diff_clean, c='blue', alpha=0.7, label='Clean Samples', s=50, edgecolors='black', linewidth=0.5)
    ax2.scatter(kl1_poison, prob_diff_poison, c='red', alpha=0.7, label='Poison Samples', s=50, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('KL1', fontsize=12)
    ax2.set_ylabel('ProbDiff', fontsize=12)
    ax2.set_title('ProbDiff-KL1 Scatter Plot', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Right plot: Distance distribution histogram
    all_samples = clean_samples + poison_samples
    distances = [distance_to_curve(s[0], s[1], curve_params) for s in all_samples]
    ax3.hist(distances, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', label='Curve Position')
    ax3.set_xlabel('Distance to Curve', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distance Distribution to Curve', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    # Read data
    print("Reading data...")
    clean_samples = read_features_file('clean_feature.txt')
    poison_samples = read_features_file('poison_feature.txt')
    detected_poison_samples = read_features_file('detected_poison_feature.txt')
    clean_sentences = read_sentences_file('clean_sentence.txt')
    
    print(f"Read {len(clean_samples)} clean samples")
    print(f"Read {len(poison_samples)} poison samples")
    print(f"Read {len(detected_poison_samples)} detected poison samples")
    print(f"Read {len(clean_sentences)} clean sentences")
    
    # Fit curve based on detected poison samples
    print("\n=== Fitting Curve Based on Detected Poison Samples ===")
    curve_params = fit_poison_curve(detected_poison_samples)
    print(f"Fitted curve parameters: alpha={curve_params[0]:.3f}, beta={curve_params[1]:.3f}, gamma={curve_params[2]:.3f}")
    
    # Combine all samples
    all_samples = clean_samples + poison_samples
    
    # Automatically adjust threshold for high precision
    target_precision = 0.98
    best_precision = 0
    best_threshold = -0.5
    best_clean_indices = []
    best_clean_distances = []
    
    print("\nAutomatically adjusting threshold for high precision...")
    for threshold in np.arange(-0.5, -3.0, -0.1):
        clean_indices, clean_distances = detect_clean_samples(
            all_samples, curve_params,
            distance_threshold_clean=threshold,
            prob_diff_clean_threshold=0.1,
            kl1_clean_threshold=5.0
        )
        
        # Evaluate precision
        true_labels = [1] * len(clean_samples) + [0] * len(poison_samples)  # 1: clean, 0: poison
        pred_labels = [0] * len(all_samples)
        for idx in clean_indices:
            pred_labels[idx] = 1
        
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        
        print(f"Threshold {threshold:.2f}: Detected {len(clean_indices)} clean samples, Precision = {precision:.4f}")
        
        if precision >= target_precision and len(clean_indices) > 0:
            best_precision = precision
            best_threshold = threshold
            best_clean_indices = clean_indices
            best_clean_distances = clean_distances
            break
        
        if precision > best_precision:
            best_precision = precision
            best_threshold = threshold
            best_clean_indices = clean_indices
            best_clean_distances = clean_distances
    
    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Number of detected clean samples: {len(best_clean_indices)}")
    print(f"Clean sample detection precision: {best_precision:.4f}")
    
    # Calculate complete performance metrics
    true_labels = [1] * len(clean_samples) + [0] * len(poison_samples)
    pred_labels = [0] * len(all_samples)
    for idx in best_clean_indices:
        pred_labels[idx] = 1
    
    precision, recall, f1 = evaluate_detection_performance(true_labels, pred_labels)
    
    # Output detected clean sentences
    detected_clean_sentences = []
    for idx in best_clean_indices:
        if idx < len(clean_sentences):
            detected_clean_sentences.append(clean_sentences[idx])
        else:
            # If index exceeds clean_sentences range, it might be misclassified poison sample
            print(f"Warning: Index {idx} exceeds clean sentence file range")
    
    # Save detected clean sentences
    output_sentence_file = 'detected_clean_sentence.txt'
    with open(output_sentence_file, 'w', encoding='utf-8') as f:
        for sentence in detected_clean_sentences:
            f.write(sentence + '\n')
    
    print(f"\nDetected clean sentences saved to: {output_sentence_file}")
    print(f"Total sentences saved: {len(detected_clean_sentences)}")
    
    # Save detection results
    output_file = 'clean_detection_results.txt'
    with open(output_file, 'w') as f:
        f.write("Clean Sample Detection Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Fitted curve parameters: alpha={curve_params[0]:.6f}, beta={curve_params[1]:.6f}, gamma={curve_params[2]:.6f}\n")
        f.write(f"Best distance threshold: {best_threshold:.2f}\n")
        f.write(f"Number of detected clean samples: {len(best_clean_indices)}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        
        f.write("Detection Details:\n")
        f.write("Index\tKL1\t\tKL2\t\tProbDiff\tDistance\tIs Clean\n")
        f.write("-" * 100 + "\n")
        
        for i, idx in enumerate(best_clean_indices):
            if idx < len(clean_samples):
                sample = clean_samples[idx]
                is_clean = "Yes"
            else:
                sample = poison_samples[idx - len(clean_samples)]
                is_clean = "No"
            
            kl1, kl2, prob_diff = sample
            dist = best_clean_distances[i]
            f.write(f"{idx}\t{kl1:.6f}\t{kl2:.6f}\t{prob_diff:.6f}\t{dist:.6f}\t{is_clean}\n")
    
    # Visualization
    print("\nGenerating visualization charts...")
    visualize_detection(clean_samples, poison_samples, curve_params)
    
    print(f"\nDetection results saved to: {output_file}")

if __name__ == "__main__":
    main()