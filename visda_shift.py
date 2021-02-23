


import numpy as np
import torch
import torch.utils.data as data



def get_visda_shift(train,batch_size = 32, drop_last=True,total_sample=2000,
             ratio=  [0.3,0.3,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
             classe_vec=[0,4,11]):
    path = './'
    aux = [str(i) for i in classe_vec]

    if train :
        filename = 'visda-train'+ ''.join(aux)+'.npz'

    else:
        filename = 'visda-val'+ ''.join(aux)+'.npz'
    print(filename)
    res = np.load(path+filename)
    data = torch.from_numpy(res['X'])
    label = torch.from_numpy(res['y'])
#    #----------------------Subsampling the dataset ---------------------------       
    c = len(torch.unique(label))
    n = label.size(0)
    ind = [[j for j in range(n) if label[j]==i] for i in range(c)]
    nb_sample =[len(ind[i]) for i in range(c) ]
    print('sample per class in data before subsampling',nb_sample)
    print('ratio*total',np.array(ratio)*total_sample)
    all_index = torch.zeros(0).long()
    for i in range(c):
        perm = torch.randperm(nb_sample[i])
        ind_classe =  label.eq(i).nonzero()
        ind = ind_classe[perm[:int(ratio[i]*total_sample)].long()]
        all_index = torch.cat((all_index,ind))
    
    label = label[all_index].squeeze()
    data = data[all_index].float().squeeze()
    print(data.shape)

    # ------------------------------------------------------------------------
        
    full_data = torch.utils.data.TensorDataset(data, label.long())
    usps_data_loader = torch.utils.data.DataLoader(
            dataset=full_data,
            batch_size= batch_size,
            shuffle=True,
            drop_last=drop_last)
    
    

    return usps_data_loader
if __name__ == '__main__':
    

    if 1:
        ratio =  [0.1,0.8,0.1]
        classe_vec = [0,4,11]
        total_sample = 6000
    
    else :
        total_sample = 24000
        ratio = [1/12]*12
        classe_vec = [0,1,2,3,4,5,6,7,8,9,10,11]
        ratio = [0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]  

    usps_loader = get_visda_shift(train=False,batch_size=1,drop_last=False,total_sample=total_sample,ratio=ratio,
                                  classe_vec=classe_vec)
    data = torch.zeros((len(usps_loader),2048)).float()
    label = torch.zeros(len(usps_loader))
    for i,(data_,target) in enumerate(usps_loader):
        data[i] = data_[0,0]
        label[i] = target
    c = len(torch.unique(label))
    n = len(label)
    ind = [[j for j in range(n) if label[j]==i] for i in range(c)]
    nb_sample = np.array([len(ind[i]) for i in range(c) ])
    print(nb_sample/sum(nb_sample),ratio)
