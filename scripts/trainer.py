import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os

def get_auc(labels, preds):
    auc = 0
    if labels.shape[1] == 0: 
        return 0
    
    valid_cols = 0
    for i in range(labels.shape[1]):
        try:
            auc += roc_auc_score(labels[:, i], preds[:, i])
            valid_cols += 1
        except ValueError:
            pass
    
    return auc / valid_cols if valid_cols > 0 else 0

def get_acc(labels, preds, threshold=0.5):
    acc = 0.0 
    one_zero_preds = (preds > threshold).astype(int)
    for label in range(preds.shape[1]):
        acc += accuracy_score(labels[:, label], one_zero_preds[:, label])

    return acc / preds.shape[1]

def train_step(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)         

        optimizer.zero_grad()
        outputs = model(inputs)                     
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = []
    all_labels = []
    all_preds = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        preds = torch.sigmoid(outputs)

        total_loss.append(loss.item())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    val_loss = np.mean(total_loss)

    if len(all_labels) > 0:
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        auc = get_auc(all_labels, all_preds)
        acc = get_acc(all_labels, all_preds)
    else:
        auc = 0
        acc = 0

    return val_loss, auc, acc

def train_model(model, train_loader, val_loader, device, num_epochs=5, lr=0.001, log_callback=None, stop_event=None, optimizer_name="Adam", criterion_name="BCEWithLogitsLoss"):
    
    if criterion_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif criterion_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == "MSELoss":
        criterion = nn.MSELoss()
    else:
        if log_callback: log_callback(f"Warning: Unknown criterion {criterion_name}, using BCEWithLogitsLoss")
        criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        if log_callback: log_callback(f"Warning: Unknown optimizer {optimizer_name}, using Adam")
        optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(num_epochs):
        if stop_event and stop_event.is_set():
            if log_callback: log_callback("Training stopped by user.")
            break
            
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_acc  = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        msg = (f"Epoch [{epoch+1}/{num_epochs}], " 
               f"train loss: {train_loss:.4f}, "
               f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, val auc: {val_auc:.4f}")
        
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
            
    return model, history
