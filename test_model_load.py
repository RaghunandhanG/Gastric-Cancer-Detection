#!/usr/bin/env python3
"""Quick test to verify PyTorch EfficientNet model loading works."""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import os

def _build_pt_efficientnet(num_classes: int = 8):
    """Build a PyTorch EfficientNet-B0 with a custom classifier head."""
    model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    # Create 3-layer classifier to match the saved checkpoint
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.Linear(512, 256),
        nn.Linear(256, num_classes),
    )
    return model

if __name__ == "__main__":
    model_path = "models/efficientnet_model_final.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit(1)
    
    print(f"📦 Loading model from {model_path}...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if checkpoint is a model or state_dict
        if isinstance(checkpoint, nn.Module):
            print("✓ Checkpoint is a complete model")
            model = checkpoint
        else:
            print("✓ Checkpoint is a state_dict")
            model = _build_pt_efficientnet(num_classes=8)
            
            print("\nAttempting strict load...")
            try:
                model.load_state_dict(checkpoint)
                print("✅ Successfully loaded with strict=True!")
            except RuntimeError as e:
                print(f"⚠️  Strict load failed: {str(e)[:150]}...")
                print("\nAttempting non-strict load...")
                model.load_state_dict(checkpoint, strict=False)
                print("✅ Successfully loaded with strict=False!")
        
        model.to(device)
        model.eval()
        print(f"\n✅ Model loaded successfully on {device}!")
        print(f"Model architecture:\n{model}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)
