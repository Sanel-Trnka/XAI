"""Helpers for training and evaluating MLP models on binary classification."""

import torch

def get_batch_accuracy(output, y, N):
    """Compute accuracy for a batch of logits and binary labels."""
    # Convert logits to probabilities and flatten to (batch_size,)
    probs = torch.sigmoid(output).view(-1)
    # Apply 0.5 threshold to obtain binary predictions
    preds = (probs >= 0.5).long()
    # Compare against ground-truth labels and normalise by dataset size
    correct = preds.eq(y.view_as(preds).long()).sum().item()
    return correct / N

def train(model, train_loader, train_N, optimizer, loss_function) -> tuple[float, float]:
    """Run one epoch of supervised training and report loss/accuracy."""
    loss = 0
    accuracy = 0
    model.train()
    for x, y in train_loader:
        # Forward pass through the current mini-batch
        output = model(x)
        # Clear stale gradients before computing the new ones
        optimizer.zero_grad()
        y_float = y.float().unsqueeze(1)  # match (batch_size, 1) for BCEWithLogitsLoss
        # Measure loss and backpropagate to compute gradients
        batch_loss = loss_function(output, y_float)
        batch_loss.backward()
        # Update parameters with the optimizer of choice
        optimizer.step()
        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    avg_loss = loss / len(train_loader)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(avg_loss, accuracy))
    return avg_loss, accuracy

def validate(model, valid_loader, valid_N, loss_function) -> tuple[float, float]:
    """Evaluate the model on a validation split without gradient tracking."""
    loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            # Only forward-pass is required in evaluation mode
            output = model(x)
            y_float = y.float().unsqueeze(1)
            loss += loss_function(output, y_float).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    avg_loss = loss / len(valid_loader)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(avg_loss, accuracy))
    return avg_loss, accuracy