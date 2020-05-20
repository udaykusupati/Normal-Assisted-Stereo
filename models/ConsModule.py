import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet
import numpy as np

class ConsModule(nn.Module):
	def __init__(self):
		super(ConsModule, self).__init__()
		self.cons_net = UNet(4, 4)
	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, depth, nmap):

		b,_,h,w = depth.size()
		if nmap.size(3) == 3:
			nmap = nmap.permute(0,3,1,2)
		input_features = torch.cat((depth, nmap), dim = 1)

		output = self.cons_net(input_features) + input_features
		depth = output[:,0].unsqueeze(1)
		nmap = F.normalize(output[:,1:], dim = 1)
		return torch.cat((depth, nmap), dim = 1)
		

class ConsLoss(nn.Module):
	def __init__(self):
		super(ConsLoss, self).__init__()
		self.sobel_kernel = None

	def get_grad_1(self, depth):
		if self.sobel_kernel is None:
			edge_kernel_x = torch.from_numpy(np.array([[1/8, 0, -1/8],[1/4,0,-1/4],[1/8,0,-1/8]])).type_as(depth)
			edge_kernel_y = torch.from_numpy(np.array([[1/8, 1/4, 1/8],[0,0,0],[-1/8,-1/4,-1/8]])).type_as(depth)
			self.sobel_kernel = torch.cat((edge_kernel_x.view(1,1,3,3), edge_kernel_y.view(1,1,3,3)), dim = 0)
			self.sobel_kernel.requires_grad = False
		grad_depth = torch.nn.functional.conv2d(depth, self.sobel_kernel, padding = 1)
		
		return -1*grad_depth


	def get_grad_2(self, depth, nmap, intrinsics_var):                
		p_b,_,p_h,p_w = depth.size()
		c_x = p_w/2
		c_y = p_h/2
		p_y = torch.arange(0, p_h).view(1, p_h, 1).expand(p_b,p_h,p_w).type_as(depth) - c_y
		p_x = torch.arange(0, p_w).view(1, 1, p_w).expand(p_b,p_h,p_w).type_as(depth) - c_x

		nmap_z = nmap[:,2,:,:]
		nmap_z_mask = (nmap_z == 0)
		nmap_z[nmap_z_mask] = 1e-10
		nmap[:,2,:,:] = nmap_z
		n_grad = nmap[:,:2,:,:].clone()
		n_grad = n_grad/ (nmap[:,2,:,:].unsqueeze(1))

		grad_depth = -n_grad.clone()*depth.clone()
		
		fx = intrinsics_var[:,0,0].clone().view(-1,1,1)
		fy = intrinsics_var[:,1,1].clone().view(-1,1,1)
		f = torch.cat((fx.unsqueeze(1),fy.unsqueeze(1)), dim = 1)
		
		grad_depth = grad_depth/f
		
		denom = (1 + p_x*(n_grad[:,0,:,:])/fx + p_y*(n_grad[:,1,:,:])/fy )
		denom[denom == 0] = 1e-10
		grad_depth = grad_depth/denom.view(p_b,1,p_h,p_w)
		
		return grad_depth

	def forward(self, depth, tgt_depth_var, nmap, intrinsics_var, mask):

		g_mask = mask.expand(-1,2,-1,-1)
		
		true_grad_depth_1 = self.get_grad_1(tgt_depth_var)*100
		grad_depth_1 = self.get_grad_1(depth)*100
		grad_depth_2 = self.get_grad_2(depth, nmap.clone(), intrinsics_var)*100
		
		g_mask = (abs(true_grad_depth_1) < 1).type_as(g_mask) & (g_mask)
		g_mask = (abs(grad_depth_1) < 5).type_as(g_mask) & (abs(grad_depth_2) < 5).type_as(g_mask) & (g_mask)
		g_mask.detach_()

		
		return F.smooth_l1_loss(grad_depth_1[g_mask], grad_depth_2[g_mask])


