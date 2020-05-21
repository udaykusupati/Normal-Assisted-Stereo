import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import ntpath
import h5py
import pickle
import matplotlib.pyplot as plt
import torch    
import os
import copy
import re
import sys
from struct import unpack
import PIL.Image
def load_as_float(path):
	return imread(path).astype(np.float32)
def load_png(path):
	return np.array(PIL.Image.open(path).convert('RGB')).astype(np.float32)

def load_h5(path):
	return np.transpose(np.array(h5py.File(path, 'r')['result']),(1,2,0))


def read_pose(path, fid):
	with open(path) as f:
		line = f.readline()
		while line:
			if "Frame" in line:
				if fid == int(line[6:]):
					line = f.readline()
					pose_tgt = np.array(line[2:])
					line = f.readline()
					pose_src = np.array(line[2:])
					return pose_tgt, pose_src
			line = f.readline()
		return None

def readPFM(file): 
	# taken from https://github.com/feihuzhang/GANet/blob/master/dataloader/dataset.py
	with open(file, "rb") as f:
			# Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
		type = f.readline().decode('latin-1')
		if "PF" in type:
			channels = 3
		elif "Pf" in type:
			channels = 1
		else:
			sys.exit(1)
		# Line 2: width height
		line = f.readline().decode('latin-1')
		width, height = re.findall('\d+', line)
		width = int(width)
		height = int(height)

			# Line 3: +ve number means big endian, negative means little endian
		line = f.readline().decode('latin-1')
		BigEndian = True
		if "-" in line:
			BigEndian = False
		# Slurp all binary data
		samples = width * height * channels;
		buffer = f.read(samples * 4)
		# Unpack floats with appropriate endianness
		if BigEndian:
			fmt = ">"
		else:
			fmt = "<"
		fmt = fmt + str(samples) + "f"
		img = unpack(fmt, buffer)
		img = np.reshape(img, (height, width))
		img = np.flipud(img)
#        quit()
	return img, height, width

def read_calib_file(path):
	# taken from https://github.com/hunse/kitti
	float_chars = set("0123456789.e+- ")
	data = {}
	with open(path, 'r') as f:
		for line in f.readlines():
			key, value = line.split(':', 1)
			value = value.strip()
			data[key] = value
			if float_chars.issuperset(value):
				# try to cast to float array
				try:
					data[key] = np.array(value.split(' '), dtype = 'float32')
				except ValueError:
					# casting error: data[key] already eq. value, so pass
					pass

	return data
class SequenceFolder(data.Dataset):
	"""A sequence data loader where the files are arranged in this way:
		root/scene_1/0000000.jpg
		root/scene_1/0000001.jpg
		..
		root/scene_1/cam.txt
		root/scene_2/0000000.jpg
		.

		transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
	"""

	def __init__(self, root, seed=None, ttype='train.txt', sequence_length=2, transform=None, target_transform=None, index = 0, dataset = 'dataset'):
		np.random.seed(seed)
		random.seed(seed)
		self.dataset = dataset
		self.root = Path(root)
		if self.dataset == 'scannet':
			scene_list_path = self.root/(ttype[:-4] + '_scene.list')
			scenes = [Path(ttype[:-4])/folder[:-1] for folder in open(scene_list_path)]
		elif self.dataset == 'sceneflow' or self.dataset == 'kitti2015' or self.dataset == 'kitti2012':
			pass
		else:
			scene_list_path = self.root/ttype
			scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
		self.ttype = ttype
		self.transform = transform
		if self.dataset != 'sceneflow' and self.dataset != 'kitti2015' and self.dataset != 'kitti2012':
			self.scenes = sorted(scenes)
		self.crawl_folders(sequence_length)
		self.index = index

	def crawl_folders(self, sequence_length):
		if self.dataset == 'scannet':
			if os.path.exists("./scannet/scan_"+self.ttype[:-4]+"_dump.pkl"):
				sequence_set = pickle.load(open("./scannet/scan_"+self.ttype[:-4] + "_dump.pkl",'rb'))
			else:
				sequence_set = []
				imgs = np.genfromtxt('scannet/new_orders/'+self.ttype[:-4]+"/"+self.ttype[:-4]+'new_orders_v.list', delimiter = ' ', dtype = 'unicode')
				imgs = imgs[imgs[:,0].argsort()]

				for i in range(len(imgs)):
					scene = Path(self.ttype[:-4])/imgs[i,0][2:-9]
					img = imgs[i,0][-9:][:-5] + '.jpg'
					n_img = imgs[i,1] + '.jpg'
					intrinsics = np.genfromtxt('./scannet'/scene/'intrinsic/intrinsic_depth.txt').astype(np.float32).reshape((4, 4))[:3,:3]
					gt_nmap = "./scannet/normals/"/scene/img[:-4]+"_normal.npy"
					depth = './scannet/'/scene/'depth/'/(img[:-4]+'.npy')
					pose_tgt = './scannet/'/scene/'pose/'/(img[:-4]+'.txt')
					n_index = [n_img]
					sample = {'intrinsics': intrinsics, 'tgt': './scannet'/scene/'color/'+img, 'tgt_depth': depth, 'ref_imgs': [], 'pose_tgt': pose_tgt, 'pose_src': [], 'gt_nmap': gt_nmap, 'ref_depths': []}
					for j in n_index:
						sample['ref_imgs'].append('./scannet'/scene/'color/'+j)
						sample['ref_depths'].append('./scannet/'/scene/'depth/'/(j[:-4]+'.npy'))
						sample['pose_src'].append('./scannet/'/scene/'pose/'/(j[:-4]+'.txt'))
					sequence_set.append(sample)
				#pickle.dump(sequence_set,open("./scannet/scan_"+self.ttype[:-4]+"_dump.pkl",'wb'))
		elif self.dataset == 'sceneflow':
			self.scenes = set()
			if os.path.exists("./sceneflow/sflow_" + self.ttype[:-4] + "_dump.pkl"):
				sequence_set = pickle.load(open("./sceneflow/sflow_" + self.ttype[:-4] + "_dump.pkl",'rb'))
			else:
				sequence_set = []
				imgs = np.genfromtxt('sceneflow/lists/sceneflow_' + self.ttype[:-4]+ '.list', dtype = 'unicode')
				imgs = imgs[imgs.argsort()]
				for i in range(len(imgs)):
					scene = Path(imgs[i][:-8])
					self.scenes.add(scene)
					img = imgs[i]
					n_img = scene[:-5] + 'right/' + imgs[i][-8:]
					if "15mm" in scene:
						intrinsics = np.array([[450.0, 0.0, 479.5], [0.0, 450.0, 269.5], [0.0, 0.0, 1.0]]).astype(np.float32)
					else:
						intrinsics = np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]]).astype(np.float32)
					gt_nmap = 'sceneflow/normals/' + scene + imgs[i][-8:-4] + '_normal.npy'
					disp = 'sceneflow/disparity/' + scene + imgs[i][-8:-4] + '.pfm'
					pose = read_pose('sceneflow/camera_data/' + scene[:-5] + 'camera_data.txt', int(imgs[i][-8:-4]))
					if pose == None:
						continue
					else:
						pose_tgt, pose_src = pose
					n_index = [n_img]
					sample = {'intrinsics': intrinsics, 'tgt': './sceneflow/frames_finalpass/'+img, 'tgt_depth': disp, 'ref_imgs': [], 'pose_tgt': pose_tgt, 'pose_src': [], 'gt_nmap': gt_nmap, 'ref_depths': []}
					for j in n_index:
						sample['ref_imgs'].append('./sceneflow/frames_finalpass/' + j)
						sample['ref_depths'].append('./sceneflow/disparity' + scene[:-5] + 'right/' + imgs[i][-8:-4] + '.pfm')
						sample['pose_src'].append(pose_src)
					sequence_set.append(sample)
				self.scenes = list(sorted(self.scenes))
				#pickle.dump(sequence_set, open("./sceneflow/sflow_" + self.ttype[:-4] + "_dump.pkl", "wb"))
		elif self.dataset == 'kitti2015':
			self.scenes = set()
			if os.path.exists("./kitti2015/kitti2015_" + self.ttype[:-4] + "_dump.pkl"):
				sequence_set = pickle.load(open("./kitti2015/kitti2015_" + self.ttype[:-4] + "_dump.pkl",'rb'))
			else:
				sequence_set = []
				imgs = np.genfromtxt('sceneflow/lists/kitti2015_' + self.ttype[:-4]+ '.list', dtype = 'unicode')
				#imgs = imgs[imgs.argsort()]
				for i in range(len(imgs)):
					scene = Path('kitti2015/')
					self.scenes.add(scene)
					img = scene + 'kitti_rgb/'+ self.ttype[:-4]+'ing/image_2/'+imgs[i]
					n_img = scene + 'kitti_rgb/'+ self.ttype[:-4]+'ing/image_3/'+imgs[i]
					cam2cam = read_calib_file(scene+'calib/'+self.ttype[:-4]+'ing/calib_cam_to_cam/'+imgs[i][:-7]+'.txt')
					intrinsics = cam2cam['P_rect_02'].reshape(3,4)[:3,:3]
					gt_nmap = scene + 'kitti_norm/training/norm_2/'+imgs[i][:-4] + '_normal.npy'
					disp = scene + 'kitti_disp/training/disp_occ_0/'+imgs[i]
					n_index = [n_img]
					sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': disp, 'ref_imgs': [], 'ref_poses': [], 'gt_nmap': gt_nmap, 'ref_depths': []}
					for j in n_index:
						sample['ref_imgs'].append(j)
						sample['ref_depths'].append(scene + 'kitti_disp/training/disp_occ_1/'+imgs[i])
						sample['ref_poses'].append(np.array([[[1.0, 0.0, 0.0,-0.539], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0,0.0]]]).astype(np.float32))
					sequence_set.append(sample)
					self.scenes = list(sorted(self.scenes))
				#pickle.dump(sequence_set, open("./kitti2015/kitti2015_" + self.ttype[:-4] + "_dump.pkl", "wb"))
		elif self.dataset == 'kitti2012':
			self.scenes = set()
			if os.path.exists("./kitti2012/kitti2012_" + self.ttype[:-4] + "_dump.pkl"):
				sequence_set = pickle.load(open("./kitti2012/kitti2012_" + self.ttype[:-4] + "_dump.pkl",'rb'))
			else:
				sequence_set = []
				imgs = np.genfromtxt('sceneflow/lists/kitti2012_' + self.ttype[:-4]+ '.list', dtype = 'unicode')
				#imgs = imgs[imgs.argsort()]
				for i in range(len(imgs)):
					scene = Path('kitti2012/')
					self.scenes.add(scene)
					img = scene + 'kitti_rgb/'+ self.ttype[:-4]+'ing/colored_0/'+imgs[i]
					n_img = scene + 'kitti_rgb/'+ self.ttype[:-4]+'ing/colored_1/'+imgs[i]
					cam2cam = read_calib_file(scene+'calib/'+self.ttype[:-4]+'ing/calib_cam_to_cam/'+imgs[i][:-7]+'.txt')
					intrinsics = cam2cam['P2'].reshape(3,4)[:3,:3]
					gt_nmap = scene + 'kitti_norm/training/norm_2/'+imgs[i][:-4] + '_normal.npy'
					disp = scene + 'kitti_disp/training/disp_occ/'+imgs[i]
					n_index = [n_img]
					sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': disp, 'ref_imgs': [], 'ref_poses': [], 'gt_nmap': gt_nmap, 'ref_depths': []}
					for j in n_index:
						sample['ref_imgs'].append(j)
						sample['ref_depths'].append(scene + 'kitti_disp/training/disp_occ/'+imgs[i])
						sample['ref_poses'].append(np.array([[[1.0, 0.0, 0.0,-0.539], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0,0.0]]]).astype(np.float32))
					sequence_set.append(sample)
					self.scenes = list(sorted(self.scenes))
				#pickle.dump(sequence_set, open("./kitti2012/kitti2012_" + self.ttype[:-4] + "_dump.pkl", "wb"))
		else:
			if os.path.exists("./dataset/demon_"+self.ttype[:-4]+"_dump.pkl"):
				sequence_set = pickle.load(open("./dataset/demon_"+self.ttype[:-4] + "_dump.pkl",'rb'))
			else:
				sequence_set = []
				demi_length = sequence_length//2

				s = 0
				for scene in self.scenes:
					s = s+1
					

					intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
					poses = np.genfromtxt(scene/'poses.txt').astype(np.float32)
					imgs = sorted(scene.files('*.jpg'))
					if len(imgs) < sequence_length:
						continue
					for i in range(len(imgs)):
						if i < demi_length:
							shifts = list(range(0,sequence_length))
							shifts.pop(i)
						elif i >= len(imgs)-demi_length:
							shifts = list(range(len(imgs)-sequence_length,len(imgs)))
							shifts.pop(i-len(imgs))
						else:
							shifts = list(range(i-demi_length, i+(sequence_length+1)//2))
							shifts.pop(demi_length)

						img = imgs[i]
						gt_nmap = "./dataset/new_normals/" + ntpath.basename(scene) + "/" + ntpath.basename(img)[:-4]+"_normal.npy"
						depth = img.dirname()/''+img.name[:-4] + '.npy'
						pose_tgt = np.concatenate((poses[i,:].reshape((3,4)), np.array([[0,0,0,1]])), axis=0)
						#sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'tgt_nmap': nmap ,  'ref_imgs': [], 'ref_poses': [], 'gt_nmap': gt_nmap, 'ref_depths': []}
						sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth,  'ref_imgs': [], 'ref_poses': [], 'gt_nmap': gt_nmap, 'ref_depths': []}
						for j in shifts:
							sample['ref_imgs'].append(imgs[j])
							sample['ref_depths'].append(imgs[j].dirname()/''+imgs[j].name[:-4] + '.npy')
							pose_src = np.concatenate((poses[j,:].reshape((3,4)), np.array([[0,0,0,1]])), axis=0)
							pose_rel = pose_src @ np.linalg.inv(pose_tgt)
							pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
							sample['ref_poses'].append(pose)
						
						sequence_set.append(sample)
				#pickle.dump(sequence_set,open("./dataset/demon_"+self.ttype[:-4]+"_dump.pkl",'wb'))

		if self.ttype == 'train.txt':
		    random.shuffle(sequence_set)
		self.samples = sequence_set


	def __getitem__(self, index):
		index = self.index + index
		sample = self.samples[index]
		
		if self.dataset == 'sceneflow' or self.dataset == 'kitti2015' or self.dataset == 'sun3d_mpi':
			tgt_img = load_png(sample['tgt'])
		elif self.dataset == 'kitti2012':
			tgt_img = load_as_float(sample['tgt'])
		else:
			tgt_img = load_as_float(sample['tgt'])
		if self.dataset == 'sceneflow':
			tgt_depth, _, _ = readPFM(sample['tgt_depth'])
			tgt_depth = sample['intrinsics'][0,0]/tgt_depth.astype(np.float32)
			pose_tgt = np.fromstring(str(sample['pose_tgt']), dtype = float, sep = ' ').astype(np.float32).reshape((4, 4))
			sample['ref_poses'] = []
			for p in sample['pose_src']:
				p_src = np.fromstring(str(p), dtype = float, sep = ' ').astype(np.float32).reshape((4, 4))
				pose_rel = np.linalg.inv(p_src) @ pose_tgt
				pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
				sample['ref_poses'].append(pose)
		elif self.dataset == 'kitti2015' or self.dataset == 'kitti2012':
			tgt_depth = np.array(PIL.Image.open(sample['tgt_depth'])).astype(np.float32)
			tgt_depth = tgt_depth/256.0

		elif self.dataset == 'sun3d_mpi':
			depth_pil = PIL.Image.open(sample['tgt_depth'])
			depth_arr = np.array(depth_pil)
			depth_uint16 = depth_arr.astype(np.uint16)
			depth_float = (depth_uint16/1000).astype(np.float32)
			tgt_depth = depth_float
		else:
			tgt_depth = np.load(sample['tgt_depth'])
		if self.dataset == 'scannet':
			tgt_depth = (tgt_depth/1000.0).astype(np.float32)
			pose_tgt = np.genfromtxt(sample['pose_tgt']).astype(np.float32).reshape((4, 4))
			sample['ref_poses'] = []
			for p in sample['pose_src']:
				p_src = np.genfromtxt(p).astype(np.float32).reshape((4, 4))
				pose_rel = np.linalg.inv(p_src) @ pose_tgt
				pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
				sample['ref_poses'].append(pose)
		#ref_depths = [np.load(ref_depth) for ref_depth in sample['ref_depths']]
		if self.dataset == 'sceneflow' or self.dataset == 'kitti2015' or self.dataset == 'sun3d_mpi':
			ref_imgs = [load_png(ref_img) for ref_img in sample['ref_imgs']]
		elif self.dataset == 'kitti2012':
			ref_imgs = [np.asarray(PIL.Image.open(ref_img)).astype(np.float32) for ref_img in sample['ref_imgs']]
		else:
			ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
		
		gt_nmap = np.load(sample['gt_nmap']).astype(np.float32)
		gt_nmap = 1 - gt_nmap*2
		gt_nmap[:,:,2] = abs(gt_nmap[:,:,2])*-1

		ref_poses = sample['ref_poses']
		
		
		if self.transform is not None:
			imgs, depths, nmaps, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth], [gt_nmap], np.copy(sample['intrinsics']))
			tgt_img = imgs[0]
			ref_imgs = imgs[1:]
			gt_nmap = nmaps[0]

			tgt_depth = depths[0]
		else:
			intrinsics = np.copy(sample['intrinsics'])
	
		return tgt_img, ref_imgs, gt_nmap, ref_poses, intrinsics, np.linalg.inv(intrinsics), tgt_depth
		

	def __len__(self):
		return len(self.samples)
