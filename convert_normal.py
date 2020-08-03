import numpy as np
#import matplotlib.pyplot as plt
import cv2
import errno
import os
import glob
import argparse

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def pop3d(cx, cy, depth, fx, fy):
	h, w = depth.shape[:2]
	y_coord = np.arange(0, h, 1).reshape((h, 1, 1))
	y_coord = np.repeat(y_coord, repeats=w, axis=1)
	x_coord = np.arange(0, w, 1).reshape((1, w, 1))
	x_coord = np.repeat(x_coord, repeats=h, axis=0)
	coords = np.concatenate([x_coord, y_coord], axis=2)
	ppc = np.ones(coords.shape)
	ppc[..., 0] *= cx
	ppc[..., 1] *= cy
	focal = np.ones(coords.shape)
	focal[..., 0] *= fx
	focal[..., 1] *= fy
	XY = (coords - ppc) * depth / focal

	return XY

def cal_normal(XY, Z, win_sz, dep_th):

	def cal_patch(i, j, sz):
		cent_d = Z[i+sz//2, j+sz//2, 0]
		val_mask = (np.abs(Z[i:i+sz, j:j+sz, 0] - cent_d) < dep_th * cent_d) & (Z[i:i+sz, j:j+sz, 0] > 0)
		if val_mask.sum() < 10:
			return np.array([0., 0., 0.])
		comb_patch = np.concatenate([XY[i:i+sz, j:j+sz], Z[i:i+sz, j:j+sz]], axis=2)
		A = comb_patch[val_mask]
		A_t = np.transpose(A, (1, 0))
		A_tA = np.dot(A_t, A)
		try:
			n = np.dot(np.linalg.inv(A_tA), A_t).sum(axis=1, keepdims=False)
		except:
			n = np.array([0., 0., 0.])
		return n

	h, w = Z.shape[:2]
	normal = np.zeros((h-win_sz, w-win_sz, 3))
	for i in range(h-win_sz):
		for j in range(w-win_sz):
			norm_val = cal_patch(i, j, win_sz)
			normal[i, j] = norm_val
	return normal

def convert_scene(src_dir, dst_dir, sce_name, j, win_sz=7, dep_th=0.1, x_ds=2, y_ds=2):
	sav_path = '{}/{}'.format(dst_dir, sce_name)
	mkdir_p(sav_path)
	sav_path = sav_path + '/{:04d}_normal.npy'.format(j)
	if os.path.exists(sav_path):
		print('{} exists, continue ...'.format(sav_path))
		return

	depth = np.load('{}/{}/{:04d}.npy'.format(src_dir, sce_name, j))
	cam = np.genfromtxt('{}/{}/cam.txt'.format(src_dir, sce_name)).astype(np.float)

	cam[0] /= x_ds
	cam[1] /= y_ds

	cx = cam[0, -1]
	cy = cam[1, -1]
	fx = cam[0, 0]
	fy = cam[1, 1]

	Z = np.expand_dims(depth, axis=2)[::y_ds, ::x_ds]
	XY = pop3d(cx, cy, Z, fx, fy)

	normal = cal_normal(XY, Z, win_sz, dep_th)

	h, w = depth.shape[:2]
	normal_u = cv2.resize(normal, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
	n_div = np.linalg.norm(normal_u, axis=2, keepdims=True)+1e-10
	normal_u /= n_div
	normal_u = (normal_u + 1.) / 2.

	# #todo debug ##########################
	# print(normal_u.max(), normal_u.min())
	# normed_ = normal_u * 2 - 1
	# n_z = normed_[..., -1]
	# mask = np.zeros(n_z.shape)
	# mask[n_z<0] = 1
	# _, axs = plt.subplots(1, 3)
	# axs[0].imshow(normal_u)
	# axs[1].imshow(n_z)
	# axs[2].imshow(mask)
	# plt.show()

	np.save(sav_path, normal_u.astype(np.float16))
	print('save {} successful.'.format(sav_path))

def convert_helper(args):
	data_list = args.list
	data_dir  = args.data_dir
	dst_dir   = args.dst_dir

	all_scenes = open(data_list, 'r').readlines()
	all_scenes = list(map(lambda x: x.strip(), all_scenes))
	cnt = len(all_scenes)
	for i in range(cnt):
		scene_name = all_scenes[i]
		print('processing {} ({}/{}) ...'.format(scene_name, i, cnt))
		all_views = glob.glob('{}/{}/*.jpg'.format(data_dir, scene_name))
		for j in range(len(all_views)):
			convert_scene(src_dir=data_dir, dst_dir=dst_dir, sce_name=scene_name, j=j)

parser = argparse.ArgumentParser(description='convertor',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', dest='data_dir', type=str, default='../train')
parser.add_argument('--list', dest='list', type=str)
parser.add_argument('--dst-dir', dest='dst_dir', type=str, default='./train_normal')

args = parser.parse_args()

convert_helper(args)
