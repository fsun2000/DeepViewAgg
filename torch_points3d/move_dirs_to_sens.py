import sys
import os


import shutil 


base_path = "/project/fsun/data/scannet/scans"


for d in os.listdir(base_path):
    scene_dir = os.path.join(base_path, d)
    
    
    target = os.path.join(scene_dir, 'sens')
    
    print(scene_dir)
    if not os.path.exists(target):
        os.makedirs(target)
            
    for name in ['color', 'depth', 'intrinsic', 'pose']:
                
        shutil.move(os.path.join(scene_dir, name), os.path.join(target, name))
    
