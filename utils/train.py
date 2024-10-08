import copy
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight

def device_avail():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(lov, model_class, criterion, epochs=30, batch_size=8):
    device = device_avail()
    torch.manual_seed(0)
    
    lov2 = list(dict.fromkeys([j[0][0:4] for j in lov]))  # Remove duplicates while maintaining order
    
    for i in lov2:
        vid_val = i
        print('Testing on data from patientID#:')
        print(vid_val)
        
        test = tmpdf[tmpdf['videos'].str.contains(vid_val)]
        train = tmpdf[~tmpdf['videos'].str.contains(vid_val)]
        
        tsdataset = TrainDataset(test['feature_layer'], test['label'], mask_pct=0)
        trdataset = TrainDataset(train['feature_layer'], train['label'], mask_pct=0)
        
        train_dataloader = DataLoader(trdataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(tsdataset, batch_size=batch_size, shuffle=False)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(train['label']), y=train['label'])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(device)
        
        model = model_class.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('-inf')
        
        print('Starting training')
        for e in tqdm(range(epochs)):
            model.train()
            tr_losses = 0
            
            for videos, labels in train_dataloader:
                optimizer.zero_grad()
                videos, labels = videos.to(device), labels.to(device)
                output = model(videos.squeeze(1))
                loss = criterion(output.float(), labels.float())
                loss.backward()
                optimizer.step()
                tr_losses += loss.item()
            
            # Log training loss
            training_loss = tr_losses / len(train_dataloader)
            print(f'Epoch [{e+1}/{epochs}], Training Loss: {training_loss:.4f}')
            
            # Validate model
            with torch.no_grad():
                model.eval()
                ts_losses = 0
                for videos, labels in test_dataloader:
                    videos, labels = videos.to(device), labels.to(device)
                    output = model(videos.squeeze(1))
                    loss = criterion(output.float(), labels.float())
                    ts_losses += loss.item()

                testing_loss = ts_losses / len(test_dataloader)
                print(f'Epoch [{e+1}/{epochs}], Testing Loss: {testing_loss:.4f}')
                
                # Save best model weights
                if testing_loss < best_loss:
                    best_loss = testing_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_wts)
        torch.save(model, f'/trans_{vid_val}.pth')
