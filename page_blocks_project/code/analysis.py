import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import os
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
sns.set_style("whitegrid")

def add_watermark(ax, text="2022CS620", alpha=0.1):
    """Add watermark to plot"""
    ax.text(0.5, 0.5, text,
            fontsize=40, color='gray',
            ha='center', va='center',
            alpha=alpha, transform=ax.transAxes,
            rotation=30, zorder=-1)

def load_data():
    """Download and load the Page Blocks Classification dataset"""
    # Create data directory if it doesn't exist
    if not os.path.exists('../data'):
        os.makedirs('../data')
    
    # Define column names
    columns = ['height', 'length', 'area', 'eccen', 'p_black', 
               'p_and', 'mean_tr', 'blackpix', 'blackand', 'wb_trans', 'class']
    
    # Define the data URL and local file path
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/page-blocks.csv"
    local_file = '../data/page_blocks.csv'
    
    try:
        # Try to load existing local file first
        if os.path.exists(local_file):
            data = pd.read_csv(local_file)
        else:
            # Download and load the data
            data = pd.read_csv(data_url, names=columns)
            # Save local copy
            data.to_csv(local_file, index=False)
            
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_univariate_plots(data):
    """Create and save univariate analysis plots"""
    
    # Plot 1: Distribution of block heights
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.histplot(data=data, x='height', bins=30)
    plt.title('Distribution of Block Heights')
    plt.xlabel('Height')
    plt.ylabel('Count')
    add_watermark(ax)
    plt.savefig('../images/univariate_height.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Class distribution
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    class_counts = data['class'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Block Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    add_watermark(ax)
    plt.savefig('../images/univariate_class.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_bivariate_plots(data):
    """Create and save bivariate analysis plots"""
    
    # Plot 3: Height vs Length by Class
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.scatterplot(data=data, x='height', y='length', hue='class', alpha=0.6)
    plt.title('Height vs Length by Block Class')
    plt.xlabel('Height')
    plt.ylabel('Length')
    add_watermark(ax)
    plt.savefig('../images/bivariate_height_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Black Pixels vs Area
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.scatterplot(data=data, x='area', y='blackpix', hue='class', alpha=0.6)
    plt.title('Black Pixels vs Area by Block Class')
    plt.xlabel('Area')
    plt.ylabel('Black Pixels')
    add_watermark(ax)
    plt.savefig('../images/bivariate_black_area.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_multivariate_plots(data):
    """Create and save multivariate analysis plots"""
    
    # Plot 5: Pairplot of key features
    g = sns.pairplot(data=data.sample(1000), 
                     vars=['height', 'length', 'area', 'blackpix'],
                     hue='class', diag_kind='hist')
    plt.suptitle('Pairwise Relationships Between Key Features', y=1.02)
    # Add watermark to each subplot
    for ax in g.axes.flat:
        add_watermark(ax, alpha=0.05)
    plt.savefig('../images/multivariate_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Correlation heatmap
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation = data[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numeric Features')
    add_watermark(ax)
    plt.savefig('../images/multivariate_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    data = load_data()
    
    # Create all plots
    create_univariate_plots(data)
    create_bivariate_plots(data)
    create_multivariate_plots(data)
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(data)}")
    print("\nFeature Statistics:")
    print(data.describe())
    
    print("\nClass Distribution:")
    print(data['class'].value_counts())

if __name__ == "__main__":
    main() 