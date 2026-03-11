import kagglehub
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.amp import GradScaler, autocast


def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Download dataset
    path = kagglehub.dataset_download(
        "orvile/gastric-cancer-histopathology-tissue-image-dataset"
    )

    dataset_dir = os.path.join(path, "HMU-GC-HE-30K", "all_image")
    print("Dataset path:", dataset_dir)

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    dataset = datasets.ImageFolder(dataset_dir, transform=transform)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    num_classes = len(dataset.classes)
    print("Classes:", dataset.classes)


    # -------------------------
    # Load ResNet50
    # -------------------------

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features,256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256,num_classes)
    )

    model = model.to(device)


    # -------------------------
    # Load saved model
    # -------------------------

    model_path = "resnet50_model_final.pt"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained model:", model_path)
    else:
        print("No previous model found, training from ImageNet weights")


    # Unfreeze full network
    for param in model.parameters():
        param.requires_grad = True


    # Loss
    criterion = nn.CrossEntropyLoss()

    # Lower LR for continued training
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # Mixed precision
    scaler = GradScaler("cuda")


    # -------------------------
    # Training Function
    # -------------------------

    def train_epoch(loader):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images,labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):

                outputs = model(images)
                loss = criterion(outputs,labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            _,predicted = torch.max(outputs,1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss/len(loader), correct/total


    # -------------------------
    # Validation Function
    # -------------------------

    def validate_epoch(loader):

        model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images,labels in loader:

                images = images.to(device)
                labels = labels.to(device)

                with autocast("cuda"):

                    outputs = model(images)
                    loss = criterion(outputs,labels)

                total_loss += loss.item()

                _,predicted = torch.max(outputs,1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss/len(loader), correct/total


    best_acc = 0
    patience = 50
    counter = 0
    epochs = 200


    # -------------------------
    # Continue Training
    # -------------------------

    for epoch in range(epochs):

        train_loss,train_acc = train_epoch(train_loader)
        val_loss,val_acc = validate_epoch(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:

            best_acc = val_acc
            torch.save(model.state_dict(),"best_resnet_model.pt")
            print("Best model saved")
            counter = 0

        else:
            counter += 1

        if counter >= patience:
            print("Early stopping")
            break


    torch.save(model.state_dict(),"resnet50_model_final.pt")

    print("\nTraining complete")
    print("Model saved as resnet50_model_final.pt")


if __name__ == "__main__":
    main()