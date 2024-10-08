import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch.autograd import Variable
import copy
import datasets
import transforms

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)



torch.manual_seed(0)
epochs = 100
batch_size = 16
f_val = 0

for i in lov2:
    vid_val = i
    print('Testing on data from patientID#:')
    print(vid_val)
    test = tmpdf[tmpdf['path'].str.contains(vid_val)]
    train = tmpdf[~tmpdf['path'].str.contains(vid_val)]
    
    trp = "E:\\NewStudy\\VIVIT\\labels\\example_video_folder"+str(i)+"R.csv"
    tsp = "E:\\NewStudy\\VIVIT\\labels\\example_video_folder"+str(i)+".csv"
    train.to_csv(trp, index=False)
    test.to_csv(tsp, index=False)
    
    trdataset = datasets.VideoLabelDataset(
	trp,
     transform=torchvision.transforms.Compose([
        transforms.VideoFolderPathToTensor(max_len=188, padding_mode='last'),
        transforms.VideoResize([224, 224]),
    ])
    )
    
    tsdataset = datasets.VideoLabelDataset(
	tsp,
     transform=torchvision.transforms.Compose([
        transforms.VideoFolderPathToTensor(max_len=188, padding_mode='last'),
        transforms.VideoResize([224, 224]),
    ])
    )
    
    train_dataloader = torch.utils.data.DataLoader(trdataset, batch_size = batch_size, shuffle = True)

    test_dataloader = torch.utils.data.DataLoader(tsdataset, batch_size = batch_size, shuffle = False)
    
    class_weights = []
    lod = len(trdataset)
    c = train['label'].value_counts(1)
    class_weights = [1/c[0],1/c[1],1/c[2],1/c[3]]
    class_weights[0] = np.float32(class_weights[0])
    class_weights[1] = np.float32(class_weights[1])
    class_weights[2] = np.float32(class_weights[2])
    class_weights[3] = np.float32(class_weights[3])
    
    training_loss, testing_loss, F1, Balanced = [], [], [], []

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(device)
    model = ViViT(224, 16, 4, 188)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_model_wts = copy.deepcopy(model.state_dict())
    lastloss=100
    best_accuracy=float('-inf')
   
    print('starting training')
    model.train()
    for e in tqdm(range(epochs)):
        # h = model.init_hidden(batch_size)
        
        tr_losses = 0
        for videos, labels in train_dataloader:
            videos = videos.view(len(videos),188,3,224,224)
            optimizer.zero_grad()
            videos = videos.float()
            labels = labels.long()
            print(videos.size)
            videos = videos.squeeze(1).to(device)
            output = model(videos)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            tr_losses += loss.item()

        training_loss += [tr_losses/len(train_dataloader)]
        
        with torch.no_grad():
            ts_losses = 0
            epochloss = 0
            low_loss = 100
            actuals, preds = [], []
            for videos, labels in test_dataloader:
                model.eval()
                videos = videos.view(len(videos),188,3,224,224)
                videos = videos.float()
                labels = labels.long()
                videos = videos.squeeze(1).to(device)
                
                output = model(videos)

                preds.append(output.cpu().numpy())
                actuals.append(labels.cpu().numpy())
                
                try:
                    print('computing loss')
                    loss = criterion(output, labels)

                except:
                    print('in exception')
                    print(output.shape)
                    print(labels.shape)

                ts_losses += loss.item()
            
            epochloss += ts_losses/len(test_dataloader)
            testing_loss.append(epochloss)
            
            if epochloss < low_loss:
                low_loss = epochloss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            lastloss = epochloss
            
    with torch.no_grad():
        model.load_state_dict(best_model_wts)
        torch.save(model, ' ' + vid_val + '.pth')
        model.eval()
        output = model(videos)
        fpreds,factuals = [],[]
        fpreds.append(output.cpu().numpy())
        factuals.append(labels.cpu().numpy())
       
    A = factuals
    P = np.argmax(np.vstack(fpreds), axis = 1)
    val = int(sum(sum(torch.eq(torch.Tensor(A), torch.Tensor(P)))))
    print(A)
    print(P)
    print(val)
    f_val = f_val + (val/6)
