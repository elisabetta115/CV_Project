import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
from globals import FT_NUM_EPOCHS, FT_LEARNING_RATE, FT_LABEL_SMOOTHING, FT_WEIGHT_DECAY

def set_seed(seed = 25):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_effective_parameters(model):
    """Count effective parameters in model after structural pruning"""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total += param.numel()
    return total

def measure_inference_time(model, dataloader, device, num_batches=50, warmup=5):
    """Measure average inference time per batch"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start_time)
    
    # Skip warmup iterations
    return np.mean(times[warmup:]) if len(times) > warmup else np.mean(times)


def finetune_pruned_model(model, train_loader, val_loader, args, device='cuda'):
    """
    Fine-tune a pruned model for a few epochs to recover performance.
    Only trains the remaining components - pruned components stay removed.
    """
    print(f"\nFine-tuning pruned model for {args.finetune_epochs} epochs...")
    print(f"Learning rate: {args.finetune_lr}")
    print(f"Weight decay: {args.weight_decay}")
    
    # Count trainable parameters
    trainable_params = []
    total_params = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            trainable_count += param.numel()
        total_params += param.numel()
    
    print(f"Trainable parameters: {trainable_count:,} / {total_params:,} "
          f"({100 * trainable_count / total_params:.1f}%)")
    
    # Setup optimizer - only for parameters that exist in the pruned model
    optimizer = optim.AdamW(trainable_params, lr=args.finetune_lr, weight_decay=args.weight_decay)
    
    # Use cosine annealing for fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.finetune_epochs)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    best_val_acc = 0
    best_model_state = None
    finetune_history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    
    # Initial validation
    print("\nInitial validation before fine-tuning...")
    initial_val_acc = validate_model(model, val_loader, device)
    print(f"Initial validation accuracy: {initial_val_acc:.2f}%")
    
    for epoch in range(args.finetune_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}/{args.finetune_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle both hard and soft labels (for mixup/cutmix)
            if labels.dim() == 2:  # Soft labels
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                loss = -(labels * log_probs).sum(dim=1).mean()
                _, preds = outputs.max(1)
                hard_labels = labels.argmax(dim=1)
                train_correct += preds.eq(hard_labels).sum().item()
            else:  # Hard labels
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                train_correct += preds.eq(labels).sum().item()
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_total += images.size(0)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total,
                    'lr': scheduler.get_last_lr()[0]
                })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total
        
        # Validation phase
        val_acc = validate_model(model, val_loader, device)
        
        # Store history
        current_lr = scheduler.get_last_lr()[0]
        finetune_history['train_loss'].append(avg_train_loss)
        finetune_history['train_acc'].append(avg_train_acc)
        finetune_history['val_acc'].append(val_acc)
        finetune_history['lr'].append(current_lr)
        
        print(f'Epoch [{epoch+1}/{args.finetune_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {avg_train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%, '
              f'LR: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f'  → New best validation accuracy!')
        
        # Update learning rate
        scheduler.step()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with validation accuracy: {best_val_acc:.2f}%")
    
    print(f"\nFine-tuning completed:")
    print(f"  Initial accuracy: {initial_val_acc:.2f}%")
    print(f"  Best accuracy: {best_val_acc:.2f}%")
    print(f"  Improvement: {best_val_acc - initial_val_acc:+.2f}%")
    
    return model, finetune_history, best_val_acc


def validate_model(model, val_loader, device):
    """
    Validate model accuracy on validation set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def get_finetuning_args(parser):
    """
    Add fine-tuning arguments to an argument parser.
    """
    parser.add_argument('--finetune-epochs', type=int, default=FT_NUM_EPOCHS,
                        help='Number of fine-tuning epochs after pruning (0 to disable)')
    parser.add_argument('--finetune-lr', type=float, default=FT_LEARNING_RATE,
                        help='Learning rate for fine-tuning (typically lower than training)')
    parser.add_argument('--weight-decay', type=float, default=FT_WEIGHT_DECAY,
                        help='Weight decay for fine-tuning')
    parser.add_argument('--label-smoothing', type=float, default=FT_LABEL_SMOOTHING,
                        help='Label smoothing for fine-tuning')
    return parser
