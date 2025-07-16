import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from globals import *
from models import VisionTransformer
from utils import set_seed, count_parameters, create_tiny_imagenet_datasets


def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def train_baseline_model(args):
    """Main training function for baseline model"""
    set_seed(RANDOM_SEED)
    
    print(f"Using device: {DEVICE}")
    print("="*60)
    
    # Create datasets
    print("Loading Tiny-ImageNet dataset...")
    train_dataset, val_dataset = create_tiny_imagenet_datasets(
        DATA_PATH, NORMALIZE_MEAN, NORMALIZE_STD
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    # Create model
    print("\nCreating baseline ViT model...")
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Training loop
    best_val_acc = 0
    train_history = {'loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, DEVICE
        )
        train_history['loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        
        # Validate
        val_acc = evaluate(model, val_loader, DEVICE)
        train_history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{args.epochs}] - '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_history': train_history,
                'config': {
                    'embed_dim': EMBED_DIM,
                    'num_heads': NUM_HEADS,
                    'num_layers': NUM_LAYERS,
                    'mlp_dim': MLP_DIM,
                    'patch_size': PATCH_SIZE,
                    'image_size': IMAGE_SIZE,
                    'num_classes': NUM_CLASSES,
                    'dropout': DROPOUT
                }
            }, args.save_path)
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")
    return model, train_history


def main():
    parser = argparse.ArgumentParser(description='Train baseline Vision Transformer')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--save-path', type=str, default=BASELINE_MODEL_PATH,
                        help='Path to save the trained model')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_baseline_model(args)
    
    print("\nBaseline model training completed!")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()