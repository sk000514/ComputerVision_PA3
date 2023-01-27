import torch
import torch.nn as nn
import torch.nn.functional as F

class RAIN(nn.Module):
    def __init__(self, dims_in, eps=1e-5):
        '''Compute the instance normalization within only the background region, in which
            the mean and standard variance are measured from the features in background region.
        '''
        super(RAIN, self).__init__()
        self.foreground_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(dims_in), requires_grad=True)
        self.eps = eps
    def forward(self, x, mask):
        foreground_mask=F.interpolate(mask,(x.shape[2],x.shape[3]))
        mean_fore, std_fore=self.get_foreground_mean_std(x,foreground_mask)

        background_mask=1-foreground_mask
        mean_back, std_back=self.get_foreground_mean_std(x,background_mask)
        
        normalized_background=(x-mean_back)/std_back
        a=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.background_gamma,0),2),3)
        b=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.background_beta,0),2),3)
        normalized_background=(normalized_background*a+b)*background_mask

        normalized_foreground=(x-mean_fore)/std_fore
        normalized_foreground=normalized_foreground*std_back+mean_back
        a=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.foreground_gamma,0),2),3)
        b=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.foreground_beta,0),2),3)
        normalized_foreground=(normalized_foreground*a+b)*foreground_mask

        return normalized_foreground + normalized_background

    def get_foreground_mean_std(self, region, mask):
        fore=region*mask
        num=torch.sum(mask,(2,3))
        tmp=torch.sum(fore,(2,3))
        mean=tmp/(num+self.eps)
        mean=torch.unsqueeze(mean,-1)
        mean=torch.unsqueeze(mean,-1)
        var=torch.sum((fore-mean*mask)**2,(2,3))/(num+self.eps)
        var=torch.unsqueeze(var,-1)
        var=torch.unsqueeze(var,-1)
        return mean, torch.sqrt(var+self.eps)