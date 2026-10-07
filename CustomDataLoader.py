import os
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

'''
=== Set the train, test, validation dataLoader ===
    resize - Image data size for model
    aug_Rotation - Random rotation for augmentation
    aug_Scaling - Random scaling for augmentation
    aug_Brightness - Random brightness for augmentation
    aug_Contrast - Random contrast for augmentation    
'''
def get_group_id(path, label):

    filename = os.path.splitext(os.path.basename(path))[0]

    # =====================================================
    # PNEUMONIA
    # person124_bacteria_240
    # person124_virus_239
    #
    # -> person124
    # =====================================================
    if label == 1:
        return filename.split('_')[0]

    # =====================================================
    # NORMAL
    # IM-0115-0001
    # NORMAL2-IM-0115-0001
    #
    # -> IM-0115
    # =====================================================
    else:
        filename = filename.replace("NORMAL2-", "")

        return '-'.join(filename.split('-')[:-1])

def customDataLoader_ResNet(
        resize=224,
        aug_Rotation=10,
        aug_Scaling=0.1,
        aug_Brightness=0.0, 
        aug_Contrast=0.3):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # =========================================================
    # Transform
    # =========================================================

    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),

        transforms.RandomRotation(aug_Rotation),

        transforms.RandomAffine(
            degrees=0,
            scale=(1 - aug_Scaling, 1 + aug_Scaling)
        ),

        transforms.ColorJitter(
            brightness=aug_Brightness,
            contrast=aug_Contrast
        ),

        transforms.ToTensor(),
        #normalize
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        #normalize
    ])


    # =========================================================
    # Load original train + original val
    # =========================================================

    train_folder = ImageFolder(
        root=os.path.join('chest_xray', 'train')
    )

    val_folder = ImageFolder(
        root=os.path.join('chest_xray', 'val')
    )

    # Check class mapping
    assert train_folder.class_to_idx == val_folder.class_to_idx

    all_samples = train_folder.samples + val_folder.samples

    all_labels = [
        label for _, label in all_samples
    ]


    # =========================================================
    # Create patient/group dictionary
    # =========================================================

    groups = {}

    for index, (path, label) in enumerate(all_samples):

        group_id = get_group_id(path, label)

        if group_id not in groups:

            groups[group_id] = {
                "label": label,
                "indices": []
            }

        groups[group_id]["indices"].append(index)


    # =========================================================
    # Unique groups
    # =========================================================

    group_ids = list(groups.keys())

    group_labels = [
        groups[group_id]["label"]
        for group_id in group_ids
    ]


    print("\n===== Dataset Information =====")

    print("Total images :", len(all_samples))
    print("Total groups :", len(group_ids))


    # =========================================================
    # Patient-level 99% / 1% split
    # =========================================================

    train_groups, val_groups = train_test_split(
        group_ids,
        test_size=0.01,
        random_state=42,
        stratify=group_labels,
        shuffle=True
    )


    # =========================================================
    # Convert group IDs -> image indices
    # =========================================================

    train_indices = []

    for group_id in train_groups:

        train_indices.extend(
            groups[group_id]["indices"]
        )


    val_indices = []

    for group_id in val_groups:

        val_indices.extend(
            groups[group_id]["indices"]
        )


    # =========================================================
    # Dataset
    # =========================================================

    class CustomDataset(torch.utils.data.Dataset):

        def __init__(self, samples, transform=None):

            self.samples = samples
            self.transform = transform

        def __len__(self):

            return len(self.samples)

        def __getitem__(self, index):

            path, label = self.samples[index]

            image = Image.open(path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label


    all_dataset_train = CustomDataset(
        all_samples,
        transform=train_transform
    )

    all_dataset_val = CustomDataset(
        all_samples,
        transform=val_test_transform
    )


    train_dataset = Subset(
        all_dataset_train,
        train_indices
    )

    val_dataset = Subset(
        all_dataset_val,
        val_indices
    )


    # =========================================================
    # Test Dataset
    # =========================================================

    test_dataset = ImageFolder(
        root=os.path.join('chest_xray', 'test'),
        transform=val_test_transform
    )


    # =========================================================
    # DataLoader
    # =========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )


    # =========================================================
    # Check image overlap
    # =========================================================

    train_paths = set(
        all_samples[i][0]
        for i in train_indices
    )

    val_paths = set(
        all_samples[i][0]
        for i in val_indices
    )


    # =========================================================
    # Check group overlap
    # =========================================================

    train_group_set = set(train_groups)
    val_group_set = set(val_groups)


    print("\n===== Split Information =====")

    print("Train images :", len(train_indices))
    print("Val images   :", len(val_indices))

    print("Train groups :", len(train_groups))
    print("Val groups   :", len(val_groups))

    print("Image overlap:",
          len(train_paths & val_paths))

    print("Group overlap:",
          len(train_group_set & val_group_set))


    # =========================================================
    # Class distribution
    # =========================================================

    train_class_counts = {
        0: 0,
        1: 0
    }

    val_class_counts = {
        0: 0,
        1: 0
    }


    for i in train_indices:
        train_class_counts[all_samples[i][1]] += 1

    for i in val_indices:
        val_class_counts[all_samples[i][1]] += 1


    print("\n===== Class Distribution =====")

    print("Train NORMAL   :", train_class_counts[0])
    print("Train PNEUMONIA:", train_class_counts[1])

    print("Val NORMAL     :", val_class_counts[0])
    print("Val PNEUMONIA  :", val_class_counts[1])


    return train_loader, test_loader, val_loader

def customDataLoader_ResNet_Basic(
        resize=224,
        batch_size=128,
        aug_Rotation=10,
        aug_Scaling=0.1,
        aug_Brightness=0.0,
        aug_Contrast=0.3,
        dataset='chest_xray'):

    # =========================================================
    # Transform
    # =========================================================
    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomRotation(aug_Rotation),
        transforms.RandomAffine(
            degrees=0,
            scale=(1 - aug_Scaling, 1 + aug_Scaling)
        ),
        transforms.ColorJitter(
            brightness=aug_Brightness,
            contrast=aug_Contrast
        ),
        transforms.ToTensor()
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor()
    ])
    
    train_dataset = ImageFolder(root=os.path.join(dataset, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(dataset, 'test'), transform=val_test_transform)
    val_dataset = ImageFolder(root=os.path.join(dataset, 'val'), transform=val_test_transform)

    # =========================================================
    # DataLoader
    # =========================================================

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
