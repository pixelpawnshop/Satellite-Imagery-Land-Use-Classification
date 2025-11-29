"""
evaluate_model.py
Comprehensive model evaluation with metrics, confusion matrix, and per-class analysis.
"""
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/splits')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/efficientnetb0_best.pt')
BATCH_SIZE = 64
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

def get_test_loader():
    """Load test dataset."""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'), 
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    return test_loader

def load_model():
    """Load trained model."""
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def evaluate_model(model, test_loader):
    """Get predictions and ground truth labels."""
    all_preds = []
    all_labels = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - EuroSAT Classification', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def print_metrics_report(y_true, y_pred):
    """Print comprehensive classification metrics."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*80)
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n📊 Overall Test Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # Detailed classification report
    print("\n" + "-"*80)
    print("PER-CLASS METRICS")
    print("-"*80)
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Find most confused pairs
    print("\n" + "-"*80)
    print("MOST COMMON MISCLASSIFICATIONS")
    print("-"*80)
    
    misclassifications = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i][j] > 0:
                misclassifications.append((
                    CLASS_NAMES[i], 
                    CLASS_NAMES[j], 
                    cm[i][j]
                ))
    
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    
    for true_class, pred_class, count in misclassifications[:10]:
        print(f"  • {true_class:20s} → {pred_class:20s}: {count:4d} times")
    
    # Per-class accuracy
    print("\n" + "-"*80)
    print("PER-CLASS ACCURACY")
    print("-"*80)
    
    class_accuracies = []
    for i, class_name in enumerate(CLASS_NAMES):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        class_accuracies.append((class_name, class_acc, class_total))
        print(f"  {class_name:20s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_total} samples")
    
    # Identify problematic classes
    print("\n" + "-"*80)
    print("ANALYSIS & RECOMMENDATIONS")
    print("-"*80)
    
    worst_classes = sorted(class_accuracies, key=lambda x: x[1])[:3]
    print("\n🔴 Weakest performing classes:")
    for class_name, acc, total in worst_classes:
        print(f"  • {class_name}: {acc*100:.2f}% accuracy")
    
    if overall_acc >= 0.85:
        print("\n✅ Model performance is GOOD for deployment (≥85% accuracy)")
        print("   Ready for portfolio showcase!")
    elif overall_acc >= 0.75:
        print("\n⚠️  Model performance is ACCEPTABLE (75-85% accuracy)")
        print("   Consider improvements for better portfolio impact:")
        print("   - Data augmentation (rotation, flipping, color jitter)")
        print("   - Train longer (more epochs)")
        print("   - Fine-tune more layers")
        print("   - Collect more data for weak classes")
    else:
        print("\n❌ Model performance NEEDS IMPROVEMENT (<75% accuracy)")
        print("   Recommendations:")
        print("   - Check data quality and preprocessing")
        print("   - Increase model capacity")
        print("   - Use stronger augmentation")
        print("   - Train for more epochs")
        print("   - Consider ensemble methods")
    
    return cm

def main():
    print("Loading model and test data...")
    model = load_model()
    test_loader = get_test_loader()
    
    print(f"Device: {DEVICE}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Get predictions
    y_pred, y_true = evaluate_model(model, test_loader)
    
    # Print detailed metrics
    cm = print_metrics_report(y_true, y_pred)
    
    # Plot confusion matrix
    output_dir = os.path.join(os.path.dirname(__file__), '../reports')
    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(
        cm, 
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    print("\n" + "="*80)
    print("Evaluation complete! Check reports/ folder for confusion matrix.")
    print("="*80)

if __name__ == '__main__':
    main()
