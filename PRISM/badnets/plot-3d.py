import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14

def load_data(clean_file, poison_file):
    """Load clean and poisoned sample data"""
    sc_kl1, sc_kl2, sc_probdiff = [], [], []
    sp_kl1, sp_kl2, sp_probdiff = [], [], []
    
    # Read clean sample data
    with open(clean_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 3:
                sc_kl1.append(float(values[0]))
                sc_kl2.append(float(values[1]))
                sc_probdiff.append(float(values[2]))
    
    # Read poisoned sample data
    with open(poison_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 3:
                sp_kl1.append(float(values[0]))
                sp_kl2.append(float(values[1]))
                sp_probdiff.append(float(values[2]))
    
    return (sc_kl1, sc_kl2, sc_probdiff), (sp_kl1, sp_kl2, sp_probdiff)

def create_individual_3d_plots_enhanced(sc_data, sp_data):
    """Create enhanced individual 3D plots showing overlap clearly"""
    sc_kl1, sc_kl2, sc_probdiff = sc_data
    sp_kl1, sp_kl2, sp_probdiff = sp_data
    
    # Define viewing configurations
    viewing_configs = [
        {'elev': 30, 'azim': 45,   'label': '(a) elev=30°, azim=45°',   'filename': '3d_enhanced_a.png'},
        {'elev': 30, 'azim': 225,  'label': '(b) elev=30°, azim=225°',  'filename': '3d_enhanced_b.png'},
        {'elev': 30, 'azim': 315,  'label': '(c) elev=30°, azim=315°',  'filename': '3d_enhanced_c.png'},
        {'elev': 30, 'azim': 135,  'label': '(d) elev=30°, azim=135°',  'filename': '3d_enhanced_d.png'}
    ]
    
    figures = []
    
    for config in viewing_configs:
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # METHOD 1: Plot clean samples with transparency and edge colors
        scatter_clean = ax.scatter(sc_kl1, sc_kl2, sc_probdiff, 
                                  alpha=0.6, label='Clean Samples', 
                                  color='blue', s=50, edgecolor='darkblue', linewidth=0.5)
        
        # METHOD 2: Plot poisoned samples with different style
        scatter_poison = ax.scatter(sp_kl1, sp_kl2, sp_probdiff, 
                                   alpha=0.7, label='Poisoned Samples', 
                                   color='red', s=50, marker='^',  # Use triangle markers
                                   edgecolor='darkred', linewidth=0.5)
        
        # Set labels
        ax.set_xlabel('KL1', fontsize=16, labelpad=15)
        ax.set_ylabel('KL2', fontsize=16, labelpad=15)
        ax.set_zlabel('ProbDiff', fontsize=16, labelpad=15)
        
        # Increase tick label size
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        # Set viewing angle
        ax.view_init(elev=config['elev'], azim=config['azim'])
        
        # Add title
        ax.set_title(config['label'] + '\n(Blue: Clean, Red Triangles: Poisoned)', 
                    fontsize=16, pad=25)
        
        # Add legend
        ax.legend(fontsize=14, loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config['filename'], dpi=300, bbox_inches='tight')
        plt.show()
        
        figures.append(fig)
        print(f"Saved: {config['filename']}")
    
    return figures

def create_transparent_overlap_plots(sc_data, sp_data):
    """Create plots with high transparency to show overlap clearly"""
    sc_kl1, sc_kl2, sc_probdiff = sc_data
    sp_kl1, sp_kl2, sp_probdiff = sp_data
    
    viewing_configs = [
        {'elev': 30, 'azim': 45,   'label': '(a) elev=30°, azim=45°',   'filename': '3d_transparent_a.png'},
        {'elev': 30, 'azim': 225,  'label': '(b) elev=30°, azim=225°',  'filename': '3d_transparent_b.png'},
        {'elev': 30, 'azim': 315,  'label': '(c) elev=30°, azim=315°',  'filename': '3d_transparent_c.png'},
        {'elev': 30, 'azim': 135,  'label': '(d) elev=30°, azim=135°',  'filename': '3d_transparent_d.png'}
    ]
    
    figures = []
    
    for config in viewing_configs:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # METHOD 3: High transparency to see through points
        # Plot clean samples first (background)
        ax.scatter(sc_kl1, sc_kl2, sc_probdiff, 
                  alpha=0.3, label='Clean Samples', 
                  color='blue', s=60)
        
        # Plot poisoned samples on top with higher transparency
        ax.scatter(sp_kl1, sp_kl2, sp_probdiff, 
                  alpha=0.5, label='Poisoned Samples', 
                  color='red', s=60)
        
        ax.set_xlabel('KL1', fontsize=16, labelpad=15)
        ax.set_ylabel('KL2', fontsize=16, labelpad=15)
        ax.set_zlabel('ProbDiff', fontsize=16, labelpad=15)
        
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        ax.view_init(elev=config['elev'], azim=config['azim'])
        ax.set_title(config['label'] + '\n(High Transparency to Show Overlap)', 
                    fontsize=16, pad=25)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config['filename'], dpi=300, bbox_inches='tight')
        plt.show()
        
        figures.append(fig)
        print(f"Saved: {config['filename']}")
    
    return figures

def create_mixed_region_analysis(sc_data, sp_data):
    """Create plots specifically highlighting mixed regions"""
    sc_kl1, sc_kl2, sc_probdiff = sc_data
    sp_kl1, sp_kl2, sp_probdiff = sp_data
    
    # Convert to numpy arrays for analysis
    sc_array = np.column_stack([sc_kl1, sc_kl2, sc_probdiff])
    sp_array = np.column_stack([sp_kl1, sp_kl2, sp_probdiff])
    
    viewing_configs = [
        {'elev': 30, 'azim': 45,   'label': '(a) elev=30°, azim=45°',   'filename': '3d_mixed_a.png'},
        {'elev': 30, 'azim': 225,  'label': '(b) elev=30°, azim=225°',  'filename': '3d_mixed_b.png'},
        {'elev': 30, 'azim': 315,  'label': '(c) elev=30°, azim=315°',  'filename': '3d_mixed_c.png'},
        {'elev': 30, 'azim': 135,  'label': '(d) elev=30°, azim=135°',  'filename': '3d_mixed_d.png'}
    ]
    
    figures = []
    
    for config in viewing_configs:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # METHOD 4: Use different marker sizes based on density
        # Smaller markers for clean samples (less prominent)
        ax.scatter(sc_kl1, sc_kl2, sc_probdiff, 
                  alpha=0.7, label='Clean Samples', 
                  color='blue', s=30)
        
        # Larger markers for poisoned samples (more prominent)
        ax.scatter(sp_kl1, sp_kl2, sp_probdiff, 
                  alpha=0.7, label='Poisoned Samples', 
                  color='red', s=50, edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel('KL1', fontsize=16, labelpad=15)
        ax.set_ylabel('KL2', fontsize=16, labelpad=15)
        ax.set_zlabel('ProbDiff', fontsize=16, labelpad=15)
        
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        ax.view_init(elev=config['elev'], azim=config['azim'])
        ax.set_title(config['label'] + '\n(Red: Poisoned with black edges)', 
                    fontsize=16, pad=25)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config['filename'], dpi=300, bbox_inches='tight')
        plt.show()
        
        figures.append(fig)
        print(f"Saved: {config['filename']}")
    
    return figures

def create_sequential_plots(sc_data, sp_data):
    """Create plots showing clean and poisoned samples sequentially"""
    sc_kl1, sc_kl2, sc_probdiff = sc_data
    sp_kl1, sp_kl2, sp_probdiff = sp_data
    
    viewing_configs = [
        {'elev': 30, 'azim': 45,   'label': '(a) elev=30°, azim=45°',   'filename': '3d_sequential_a.png'},
        {'elev': 30, 'azim': 225,  'label': '(b) elev=30°, azim=225°',  'filename': '3d_sequential_b.png'},
        {'elev': 30, 'azim': 315,  'label': '(c) elev=30°, azim=315°',  'filename': '3d_sequential_c.png'},
        {'elev': 30, 'azim': 135,  'label': '(d) elev=30°, azim=135°',  'filename': '3d_sequential_d.png'}
    ]
    
    figures = []
    
    for config in viewing_configs:
        # Create two subplots: one for clean, one for combined
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8),
                                      subplot_kw={'projection': '3d'})
        
        # Plot 1: Only clean samples
        ax1.scatter(sc_kl1, sc_kl2, sc_probdiff, 
                   alpha=0.8, color='blue', s=40)
        ax1.set_xlabel('KL1')
        ax1.set_ylabel('KL2')
        ax1.set_zlabel('ProbDiff')
        ax1.view_init(elev=config['elev'], azim=config['azim'])
        ax1.set_title('Clean Samples Only', pad=20)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Combined view
        ax2.scatter(sc_kl1, sc_kl2, sc_probdiff, 
                   alpha=0.6, label='Clean', color='blue', s=30)
        ax2.scatter(sp_kl1, sp_kl2, sp_probdiff, 
                   alpha=0.8, label='Poisoned', color='red', s=40, 
                   edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('KL1')
        ax2.set_ylabel('KL2')
        ax2.set_zlabel('ProbDiff')
        ax2.view_init(elev=config['elev'], azim=config['azim'])
        ax2.set_title('Combined View\n(Showing Overlap)', pad=20)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(config['label'], fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(config['filename'], dpi=300, bbox_inches='tight')
        plt.show()
        
        figures.append(fig)
        print(f"Saved: {config['filename']}")
    
    return figures

def main():
    # Load data
    clean_file = "clean_feature.txt"
    poison_file = "poison_feature.txt"
    
    sc_data, sp_data = load_data(clean_file, poison_file)
    
    print("Dataset Statistics:")
    print(f"Clean samples: {len(sc_data[0])}")
    print(f"Poisoned samples: {len(sp_data[0])}")
    
    # Create enhanced plots with different visualization methods
    print("\n1. Creating enhanced plots (different markers)...")
    create_individual_3d_plots_enhanced(sc_data, sp_data)
    
    print("\n2. Creating transparent plots...")
    create_transparent_overlap_plots(sc_data, sp_data)
    
    print("\n3. Creating mixed region analysis...")
    create_mixed_region_analysis(sc_data, sp_data)
    
    print("\n4. Creating sequential comparison plots...")
    create_sequential_plots(sc_data, sp_data)
    
    print("\nAll plots generated successfully!")
    print("\nGenerated files show different aspects of the overlap:")
    print("- 3d_enhanced_*.png: Different marker shapes")
    print("- 3d_transparent_*.png: High transparency")
    print("- 3d_mixed_*.png: Size and edge highlighting") 
    print("- 3d_sequential_*.png: Side-by-side comparison")

if __name__ == "__main__":
    main()