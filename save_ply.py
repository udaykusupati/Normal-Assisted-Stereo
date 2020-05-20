import numpy as np
from PIL import Image
import os
import open3d as o3d
import sys
import matplotlib.pyplot as plt
head = 'ply\n' \
	   'format ascii 1.0\n' \
	   'comment Created by Open3D\n' \
	   'element vertex {}\n' \
	   'property double x\n' \
	   'property double y\n' \
	   'property double z\n' \
	   'property uchar red\n' \
	   'property uchar green\n' \
	   'property uchar blue\n' \
	   'end_header\n'

def custom_draw_geometry_with_custom_fov(pcd, fov_step):
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	vis.add_geometry(pcd)
	vis.get_render_option().load_from_json("my.json")
	ctr = vis.get_view_control()
	print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
	ctr.change_field_of_view(step=fov_step)
	print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
	vis.run()
	vis.destroy_window()
	#vis.get_render_option().save_to_json("my.json")
def pop_up(rgb, z, h, w, cam, name='select_plane.ply', mask = None):
	fx = cam[0, 0]
	fy = cam[1, 1]
	cx = cam[0, -1]
	cy = cam[1, -1]

	f = open(name, 'w')
	v_count = 0
	for i in range(50,h-50):
		for j in range(50,w-50):
			if mask is not None:
				if mask[i,j] == 0:
					pass#continue
			v_count += 1
	f.writelines(head.format(str(v_count)))

	y_coord = np.arange(0, h, 1).reshape((h, 1, 1))
	y_coord = np.repeat(y_coord, repeats=w, axis=1)
	x_coord = np.arange(0, w, 1).reshape((1, w, 1))
	x_coord = np.repeat(x_coord, repeats=h, axis=0)
	ones = np.ones(x_coord.shape)
	u_coord = (x_coord - cx) / fx
	v_coord = (y_coord - cy) / fy

	print(u_coord.shape, ones.shape)
	xyz = np.concatenate([u_coord, v_coord, ones], axis=2)
	xyz *= np.expand_dims(z, axis=2)
	print(h*w)
	for i in range(50,h-50):
		for j in range(50,w-50):
			if mask is not None:
				if mask[i,j] == 0:
					pass#continue
			p = xyz[i, j]
			r = rgb[i, j]
			f.writelines('{:.6f} {:.6f} {:.6f} {} {} {}\n'.format(p[0], p[1], p[2], r[0], r[1], r[2]))
	f.close()


#if not os.path.exists(sys.argv[1]+'.ply'):
#os.system('eog '+sys.argv[1]+"_aimage.png &")
rgb = Image.open(sys.argv[1]+"_aimage.png")
if sys.argv[2] == '3':
	rgb = Image.open(sys.argv[1]+"_map_diff.png")
#rgb = np.zeros((480,640,3))#np.asarray(rgb)
rgb = np.asarray(rgb)

#z = np.load(sys.argv[1]+'.npy')
name = sys.argv[1]
if sys.argv[3] == '1':
	name = name + '_dps'
	if sys.argv[2] == '3':
		rgb = Image.open(sys.argv[1]+"_map_diff_dps.png")
		rgb = np.asarray(rgb)
		

if sys.argv[2] == "0" or sys.argv[2] == "3":
	z = np.load(name+'.npy')
elif sys.argv[2] == "2":
	z = np.load(sys.argv[1] +'_gt.npy')
elif sys.argv[2] == "1":
	z = np.load(name+'.npy')
	z = 32/z
else:
	z = np.load(sys.argv[2])


# gt_z = Image.open(sys.argv[1]+ "_gt.png")
# gt_z = np.asarray(gt_z)

mask = (z[:,:] != 0)
#z, _ = load_pfm(sys.argv[3])
h,w = z.shape
print(h,w)
#rgb = cv2.resize(rgb, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
cam = np.load(sys.argv[1]+'_cam.npy')
if sys.argv[2] == '3':
	pop_up(rgb,z,h,w,cam, name = name+"_diff.ply", mask = mask)
elif sys.argv[2] == '2':
	pop_up(rgb,z,h,w,cam, name = name+"_gt.ply", mask = mask)
elif sys.argv[2] == '1':
	pop_up(rgb,z,h,w,cam, name = name+"_disp.ply", mask = mask)
else:
	pop_up(rgb,z,h,w,cam, name = name+".ply", mask = mask)
#pcd = o3d.io.read_point_cloud(name+".ply")
#custom_draw_geometry_with_custom_fov(pcd, 0)
# else:
# 	pcd = o3d.io.read_point_cloud(sys.argv[1])
# 	o3d.visualization.draw_geometries([pcd])