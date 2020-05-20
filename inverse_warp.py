from __future__ import division
import torch
from torch.autograd import Variable


pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, d, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, 1, h, 1).expand(1,d,h,w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, 1, w).expand(1,d,h,w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1,d,h,w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1) 


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, D, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, d, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(3) != h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:,:h,:w].expand(b,3,d,h,w).contiguous().view(b, 3, -1).cuda()  # [B, 3, D*H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, d, h, w)
    
    e_cam_coords = cam_coords * depth.unsqueeze(1) #extended camcoords
    stack_cam_coords = []
    stack_cam_coords.append(e_cam_coords[:,:,:,0:h,0:w])
    
    return torch.stack(stack_cam_coords, dim = 5)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, D, H, W, 1]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, d, h, w, _ = cam_coords.size()

    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, DHW]
    
    if proj_c2p_rot is not None:
        pcoords = (proj_c2p_rot.bmm(cam_coords_flat)).view(b,3,d,h,w,-1)
    else:
        pcoords = cam_coords

    if proj_c2p_tr is not None :
        pcoords = pcoords + proj_c2p_tr.view(b,3,1,1,1,1)

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)


    X_norm = 2*(X / Z)/(w-1) - 1  
    Y_norm = 2*(Y / Z)/(h-1) - 1  
    if padding_mode == 'zeros': 
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    src_pixel_coords = torch.stack([X_norm, Y_norm, Variable(torch.linspace(0,d-1,d).view(1,d,1,1,1).expand(b,d,h,w,1)).type_as(X_norm)], dim=5)
    
    return src_pixel_coords


def inverse_warp(feat, depth, pose, intrinsics, intrinsics_inv, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        feat: the source feature (where to sample pixels) -- [B, CH, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(depth, 'depth', 'BDHW')
    check_sizes(pose, 'pose', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    
    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, ch, feat_height, feat_width = feat.size()

    cam_coords = pixel2cam(depth, intrinsics_inv) 

    pose_mat = pose
    pose_mat = pose_mat.cuda()


    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,D,H,W,1,3]

    
    projected_feat = torch.nn.functional.grid_sample(feat.unsqueeze(2), src_pixel_coords.view(batch_size,src_pixel_coords.size(1),feat_height,-1,3), padding_mode=padding_mode)

    return projected_feat.view(batch_size,ch,projected_feat.size(2),feat_height,feat_width,-1)#[B,CH,D,H,W,1]

