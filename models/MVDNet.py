import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
from inverse_warp import inverse_warp, pixel2cam
import matplotlib.pyplot as plt

def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.LeakyReLU(0.1,inplace=True)
    )

class MVDNet(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(MVDNet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = mindepth
    

        self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.softmax = nn.Softmax(dim = -1)

        self.wc0 = nn.Sequential(convbn_3d(64 + 3, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        
        self.pool1 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)),
                                        nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)),
                                        nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(convbn_3d(32, 32, (2,3,3), (2,1,1), (0,1,1)),
                                        nn.ReLU(inplace=True))
        self.n_convs = nn.Sequential(
            convtext(32, 96, 3, 1, 1),
            convtext(96, 96, 3, 1, 2),
            convtext(96, 96, 3, 1, 4),
            convtext(96, 64, 3, 1, 8),
            convtext(64, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 3, 3, 1, 1)
            )
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv, no_pool = False, factor = None):
        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4
    
        refimg_fea = self.feature_extraction(ref)

        _b,_ch,_h,_w = refimg_fea.size()
            
        disp2depth = Variable(torch.ones(_b, _h, _w)).cuda() * self.mindepth * self.nlabel
        disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(_b,self.nlabel,_h,_w)).type_as(disp2depth)

        depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
        if factor is not None:
            depth = depth*factor
        
        for j, target in enumerate(targets):
            
            cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.nlabel,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            targetimg_fea  = self.feature_extraction(target)
            targetimg_fea = inverse_warp(targetimg_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)

            cost[:, :refimg_fea.size()[1],:,:,:] = refimg_fea.unsqueeze(2).expand(_b,_ch,self.nlabel,_h,_w)
            cost[:, refimg_fea.size()[1]:,:,:,:] = targetimg_fea.squeeze(-1)
            
            cost = cost.contiguous()
            cost0 = self.dres0(cost)
            
            cost_in0 = cost0.clone()
            
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0 
            cost0 = self.dres3(cost0) + cost0 
            cost0 = self.dres4(cost0) + cost0
            
            cost_in0 = torch.cat((cost_in0, cost0.clone()), dim = 1)
            
            cost0 = self.classify(cost0)

            if j == 0:
                costs = cost0
                cost_in = cost_in0
            else:
                costs = costs + cost0
                cost_in = cost_in + cost_in0

        costs = costs/len(targets)

        
        costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, self.nlabel,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt],1)) + costt

            
        costs = F.interpolate(costs, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear', align_corners = False)
        costs = torch.squeeze(costs,1)
        pred0 = F.softmax(costs,dim=1)
        pred0_r = pred0.clone()
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)

        costss = F.interpolate(costss, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear', align_corners = False)
        costss = torch.squeeze(costss,1)
        pred = F.softmax(costss,dim=1)
        pred = disparityregression(self.nlabel)(pred)
        depth1 = self.mindepth*self.nlabel/(pred.unsqueeze(1)+1e-16)

        
        b,ch,d,h,w = cost_in.size()


        with torch.no_grad():
            intrinsics_inv[:,:2,:2] = intrinsics_inv[:,:2,:2] * (4)
            disp2depth = Variable(torch.ones(b, h, w).cuda() * self.mindepth * self.nlabel).cuda()
            disps = Variable(torch.linspace(0,self.nlabel-1,self.nlabel).view(1,self.nlabel,1,1).expand(b,self.nlabel,h,w)).type_as(disp2depth)
            depth = disp2depth.unsqueeze(1)/(disps + 1e-16)
            if factor is not None:
                depth = depth*factor  
            
            wc = pixel2cam(depth, intrinsics_inv)                
            wc = wc.squeeze(-1)

        if factor is not None:
            wc = wc/(2*self.nlabel*self.mindepth*factor.unsqueeze(-1))
        else:
            wc = wc/(2*self.nlabel*self.mindepth)
        
        
        wc = wc.clamp(-1,1)
        wc = torch.cat((wc.clone(),cost_in), dim = 1) #B,ch+3,D,H,W
        wc = wc.contiguous()
        
        if no_pool:
            wc0 = self.pool1(self.wc0(wc))
        else:
            wc0 = self.pool3(self.pool2(self.pool1(self.wc0(wc))))


        slices = []
        nmap = torch.zeros((b,3,h,w)).type_as(wc0)
        for i in range(wc0.size(2)):
            slices.append(self.n_convs(wc0[:,:,i]))
            nmap += slices[-1]
        
        nmap_norm = torch.norm(nmap, dim = 1).unsqueeze(1)

        nmap = F.interpolate(nmap, [ref.size(2), ref.size(3)], mode = 'bilinear', align_corners = False)
        
        nmap = nmap.permute(0,2,3,1)

        nmap = F.normalize(nmap,dim = -1)



        return_vals = []
        if self.training:
            return_vals += [depth0, depth1]
        else:
            return_vals += [depth1]
        
        return_vals += [nmap]
        
        return return_vals

