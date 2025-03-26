import torch
import numpy as np
from metrics import *

# single fold training
def train_valid(model, train_loader, valid_loader, epochs, patience, optimizer, scheduler, criterion, thres, checkpoint_folder, device):
        
    best_val_auc = 0
    best_model = None
    best_val_loss = float('inf')
    no_improvement_count = 0  
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} of {epochs}')
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        valid_loss, val_auc = valid_step(model, valid_loader, criterion, thres, device)

        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
        
        scheduler.step()
        if val_auc > best_val_auc:
            print(f"Get best AUC:{val_auc}!!!")
            best_val_auc = val_auc
            best_model = model
            
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            no_improvement_count = 0 
        else:
            no_improvement_count += 1 

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.")
            break
    
    best_model = best_model.to(device)
    torch.save(best_model.state_dict(), f"{checkpoint_folder}/model_auc_{best_val_auc}.pth")
    print("Model has saved at", f"{checkpoint_folder}/model_auc_{best_val_auc}.pth")
    
def train_step(model, train_loader, optimizer, criterion, device):
    print('Training...')
    model.train()

    counter = 0
    train_loss = 0

    for data in train_loader:
        
        counter += 1
        data = data.to(device)
        targets = data.y
        if not data.cksnap.shape[1]==96:
            print(data.cksnap.shape) 
            continue

        _, outputs = model(data)  

        loss = criterion(outputs, targets)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    train_total_loss = train_loss / counter
    return train_total_loss

def valid_step(model, val_loader, criterion, thres=0.5, device="cuda"):
    print('Validating')
    model.eval()
    counter = 0
    val_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            counter += 1
            if not data.cksnap.shape[1]==96:
                print(data.cksnap.shape) 
                continue
            # outputs = model(graph1, graph2, cksnap, kmer)
            data = data.to(device)
            targets = data.y.to(device)
            
            _, outputs = model(data) 

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            all_predictions.extend(outputs.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
        val_total_loss = val_loss / counter

        all_predictions = np.array(all_predictions)
        binary_labels = (all_predictions >= thres).astype(int)
        all_targets = np.array(all_targets)
        metrics = evaluate_all_metrics(all_targets, all_predictions, thres)

        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        return val_total_loss, metrics['Average AUC']

def predict(model, test_loader, thres=0.5, device="cuda"):
    print('Predicting')
    res = []
    with torch.no_grad():
        for batch in test_loader:
            if device == "cpu":
                data = batch.apply(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
            else:
                data = batch.apply(lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)
            # print(data.x.device)
            des = data.des
            _, outputs = model(data)
            output_list = outputs.cpu().numpy().tolist()
            outputs = [[des[i]] + output_list[i] for i in range(len(output_list))]
            res.extend(outputs)
    return np.array(res)
    
