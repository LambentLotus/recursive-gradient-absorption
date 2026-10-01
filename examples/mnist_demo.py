#!/usr/bin/env python3
"""
RGA MNIST Demo: Train 50-layer network without gradient explosion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rga import RGA_Network, Standard_Network


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch with explosion detection."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Check for explosion
        if torch.isnan(loss) or loss.item() > 100:
            print(f"💥 EXPLODED at batch {batch_idx}, loss={loss.item():.2f}")
            return None, None, True
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, False


def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    avg_loss = test_loss / len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    # Test 1: Standard network (shallower, will explode)
    print("\n" + "="*60)
    print("TEST 1: Standard 20-layer network")
    print("="*60)
    standard = Standard_Network(depth=20).to(device)
    optimizer = optim.Adam(standard.parameters(), lr=0.001)
    
    train_loss, train_acc, exploded = train_epoch(
        standard, train_loader, optimizer, criterion, device
    )
    
    if exploded:
        print("RESULT: Standard network EXPLODED (as expected)")
    else:
        test_loss, test_acc = evaluate(standard, test_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Test 2: RGA network (deeper, stable)
    print("\n" + "="*60)
    print("TEST 2: RGA 50-layer network")
    print("="*60)
    rga = RGA_Network(depth=50).to(device)
    optimizer = optim.Adam(rga.parameters(), lr=0.001)
    
    rga_train_losses = []
    rga_test_accs = []
    epochs = 3
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc, exploded = train_epoch(
            rga, train_loader, optimizer, criterion, device
        )
        
        if exploded:
            print("RESULT: RGA EXPLODED (unexpected)")
            break
        
        test_loss, test_acc = evaluate(rga, test_loader, criterion, device)
        rga_train_losses.append(train_loss)
        rga_test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\n" + "="*60)
    print("RGA RESULT: Stable training of 50-layer network")
    print(f"Final Test Accuracy: {rga_test_accs[-1]:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
