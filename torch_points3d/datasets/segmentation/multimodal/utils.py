import os.path as osp
import glob
import torch
import numpy as np
import os
from torch_points3d.core.multimodal.image import SameSettingImageData


def read_image_pose_pairs(
        image_dir, pose_dir, image_suffix='_rgb.png',
        pose_suffix='_pose.json', skip_names=None, verbose=False, skip_freq=50, neucon_metas_dir='True'):
    """
    Search for all image-pose correspondences in the directories.
    Return the list of image-pose pairs. Orphans are ignored.
    """
    # Feng: do something with skip names? skip-freq should be ~50
    
    # Search for poses
    pose_names = sorted([
        int(osp.basename(x).replace(pose_suffix, ''))
        for x in glob.glob(osp.join(pose_dir, '*' + pose_suffix))])
    
    # Remove invalid poses and data
    for i, pose_id in enumerate(pose_names):        
        
        extr = np.loadtxt(os.path.join(pose_dir, str(pose_id) + pose_suffix))
              
        if (np.isinf(extr) + np.isnan(extr)).any():
            corrupt_pose_path = os.path.join(pose_dir, str(pose_id) + pose_suffix)
            print('corrupt_pose_path', corrupt_pose_path)
            
            corrupt_color_path = os.path.join(image_dir, str(pose_id) + image_suffix)
            corrupt_depth_path = os.path.join("/".join(image_dir.split("/")[:-1]), 'depth', str(pose_id) + '.png')
            
            print('corrupt_color_path', corrupt_color_path)
            print('corrupt_depth_path', corrupt_depth_path)
            # Remove image, depth, pose of corrupt poses
            for path in [corrupt_pose_path, corrupt_color_path, corrupt_depth_path]:
                if os.path.exists(path):
                    os.remove(path)
            
#         i_offset = 0
#         while (np.isinf(extr) + np.isnan(extr)).any():
#             print('corrupt pose: ', extr)
#             print(f"corrupt pose at {os.path.join(pose_dir, str(pose_id) + pose_suffix)}")
#             print("loading next pose")
#             i_offset += 1
#             pose_id = pose_names[i * skip_freq] + i_offset
#             print(f"new pose is {os.path.join(pose_dir, str(pose_id) + pose_suffix)}")
#             extr = np.loadtxt(os.path.join(pose_dir, str(pose_id) + pose_suffix))
#             print(extr)
            
            
#             pose_names_subset[i] = pose_id    

    # Search for images and poses
    image_names = sorted([
        int(osp.basename(x).replace(image_suffix, ''))
        for x in glob.glob(osp.join(image_dir, '*' + image_suffix))])
    pose_names = sorted([
        int(osp.basename(x).replace(pose_suffix, ''))
        for x in glob.glob(osp.join(pose_dir, '*' + pose_suffix))])

    
    # Neucon metas
    if neucon_metas_dir:        
        scene_id = image_dir.split(os.sep)[-3]
        meta_file = np.load(osp.join(neucon_metas_dir, scene_id, 'fragments.pkl'), allow_pickle=True)
        
        image_ids = []
        for d in meta_file:
            image_ids += d['image_ids']
                    
        pose_names_subset = sorted(list(set(pose_names) & set(image_ids)))
        pose_names_subset = [str(x) for x in pose_names_subset]
    # process according to skip-freq
    else:
        pose_names_subset = [str(pose_names[i]) for i in range(len(pose_names)) if i % skip_freq == 0]    

    #skip_names = skip_names if skip_names is not None else []
    #image_names = [x for x in image_names if x not in skip_names]
    #pose_names = [x for x in pose_names if x not in skip_names]
    
    image_names = pose_names_subset
    pose_names = pose_names_subset
    
    # Check if all files exist
    idx_to_pop = [] 
    for i, image_id in enumerate(image_names):
        file = os.path.join(image_dir, image_id + image_suffix)
        if os.path.exists(file):
            continue
        else:
            print(f"file {file} does not exist! ")

            offset = 0
            while True and offset <= 10:
                offset += 1
                new_file = os.path.join(image_dir, str(int(image_id) + offset) + image_suffix)
                if os.path.exists(new_file):
                    break
            image_names[i] = str(int(image_id) + offset)
            pose_names[i] = str(int(image_id) + offset)
            
            if offset == 10:
                idx_to_pop.append(i)
    # Remove entries for which no existing color image was found (within 10 tries)
    for idx in idx_to_pop:
        image_names.pop(idx)
        pose_names.pop(idx)

    print(f"{pose_dir} pose_names: ", pose_names)
                
    # Print orphans
    if not image_names == pose_names:
        image_orphan = [
            osp.join(image_dir, x + image_suffix)
            for x in set(image_names) - set(pose_names)]
        pose_orphan = [
            osp.join(pose_dir, x + pose_suffix)
            for x in set(pose_names) - set(image_names)]
        print("Could not recover all image-pose correspondences.")
        print(f"  Orphan images : {len(image_orphan)}/{len(image_names)}")
        if verbose:
            for x in image_orphan:
                print(4 * ' ' + '/'.join(x.split('/')[-4:]))
        print(f"  Orphan poses  : {len(pose_orphan)}/{len(pose_names)}")
        if verbose:
            for x in pose_orphan:
                print(4 * ' ' + '/'.join(x.split('/')[-4:]))

    # Only return the recovered pairs
    correspondences = sorted(list(set(image_names).intersection(
        set(pose_names))))
    pairs = [(
        osp.join(image_dir, x + image_suffix),
        osp.join(pose_dir, x + pose_suffix))
        for x in correspondences]
    
    return pairs


def img_info_to_img_data(info_ld, img_size):
    """Helper function to convert a list of image info dictionaries
    into a more convenient SameSettingImageData object.
    """
    if len(info_ld) > 0:
        info_dl = {k: [dic[k] for dic in info_ld] for k in info_ld[0]}
        image_data = SameSettingImageData(
            path=np.array(info_dl['path']), pos=torch.Tensor(info_dl['xyz']),
            opk=torch.Tensor(info_dl['opk']), ref_size=img_size)
    else:
        image_data = SameSettingImageData(ref_size=img_size)
    return image_data
