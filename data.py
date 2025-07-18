from globals import IMAGE_SIZE, NUM_CLASSES
import os
from torchvision import transforms
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import v2
from PIL import Image

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

class TinyImageNetDataset(Dataset):
    """Custom dataset for Tiny-ImageNet"""
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        
        if split == 'train':
            self._load_train()
        else:
            self._load_val()
    
    def _load_train(self):
        train_dir = os.path.join(self.root, 'train')
        
        # Build class to index mapping
        for i, class_dir in enumerate(sorted(os.listdir(train_dir))):
            if os.path.isdir(os.path.join(train_dir, class_dir)):
                self.class_to_idx[class_dir] = i
        
        # Load all training images
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(train_dir, class_name, 'images')
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_path, img_name)
                        self.data.append((img_path, class_idx))
    
    def _load_val(self):
        val_dir = os.path.join(self.root, 'val')
        
        # First, get class mapping from training set
        train_dir = os.path.join(self.root, 'train')
        for i, class_dir in enumerate(sorted(os.listdir(train_dir))):
            if os.path.isdir(os.path.join(train_dir, class_dir)):
                self.class_to_idx[class_dir] = i
        
        # Read validation annotations
        val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_name = parts[1]
                
                img_path = os.path.join(val_dir, 'images', img_name)
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                    self.data.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def create_tiny_imagenet_datasets(data_path, normalize_mean, normalize_std):
    """Create Tiny-ImageNet train and validation datasets"""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    try:
        # Use custom dataset loader
        train_dataset = TinyImageNetDataset(data_path, split='train', transform=transform_train)
        val_dataset = TinyImageNetDataset(data_path, split='val', transform=transform_val)
        
        print(f"Loaded {len(train_dataset)} training samples")
        print(f"Loaded {len(val_dataset)} validation samples")
        print(f"Number of classes: {len(train_dataset.class_to_idx)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise FileNotFoundError(f"Tiny-ImageNet dataset not found at {data_path}")
    
    return train_dataset, val_dataset