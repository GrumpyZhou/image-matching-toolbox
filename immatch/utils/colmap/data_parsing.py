from collections import defaultdict
import sqlite3
import numpy as np
from argparse import Namespace
from transforms3d.quaternions import quat2mat, mat2quat
import shutil
from pathlib import Path
from .read_write_model import *
from .database import COLMAPDatabase


def load_ids_from_database(database_path):
    images = {}
    cameras = {}
    db = sqlite3.connect(str(database_path))
    ret = db.execute("SELECT name, image_id, camera_id FROM images;")
    for name, image_id, camera_id in ret:
        images[name] = image_id
        cameras[name] = camera_id
    db.close()
    print(f'Found {len(images)} images and {len(cameras)} cameras in database.')
    return images, cameras

def load_cameras_from_database(database_path):
    print(f'Parsing intrinsics from {database_path}...')
    
    cameras = {}    
    db = sqlite3.connect(str(database_path))
    ret = db.execute("SELECT camera_id, model, width, height, params FROM cameras;")
    for camera_id, camera_model, width, height, params in ret:
        params = np.frombuffer(params, dtype=np.double).reshape(-1)
        camera_model = CAMERA_MODEL_IDS[camera_model]
        camera = Camera(
            id=camera_id, model=camera_model.model_name,
            width=int(width), height=int(height), params=params)
        cameras[camera_id] = camera
    return cameras

def load_cameras_from_intrinsics_and_ids(intrinsic_txt, camera_ids):    
    print(f'Parsing intrinsics from {intrinsic_txt}...')    
    
    cameras = {}
    with open(intrinsic_txt, 'r') as f:
        for line in f:
            intrinsics = line.split()
            name, camera_model, width, height = intrinsics[:4]
            params = [float(p) for p in intrinsics[4:]]
            camera_model = CAMERA_MODEL_NAMES[camera_model]
            assert len(params) == camera_model.num_params
            camera_id = camera_ids[name]
            camera = Camera(
                id=camera_id, model=camera_model.model_name,
                width=int(width), height=int(height), params=params)
            cameras[camera_id] = camera
    return cameras

def load_images_from_nvm(nvm_path):
    images = {}
    with open(nvm_path, 'r') as f:
        # Skip headers
        line = next(f)
        while line == '\n' or line.startswith('NVM_V3'):
            line = next(f)
            
        # Parse images
        num_images = int(line)
        images_empty = dict()
        for i in range(num_images):
            data = next(f).split()
            im_name = data[0]
            qvec = np.array(data[2:6], float)
            c = np.array(data[6:9], float)
            
            # NVM -> COLMAP.
            R = quat2mat(qvec)    
            tvec = - np.matmul(R, c)
            images[im_name] = dict(qvec=qvec, tvec=tvec)
    print(f'Loaded {len(images)} images from {nvm_path}.')
    return images

def create_empty_model_from_reference_model(reference_model, empty_model):
    empty_model = Path(empty_model)    
    if os.path.exists(empty_model):
        print(f'Empty sfm {empty_model} existed.')
        return
    os.makedirs(empty_model)
    print(f'Creating an empty sfm under {empty_model} from {reference_model}')
    
    # Construct images with fake points    
    images = read_images_binary(str(reference_model / 'images.bin'))
    print(f'Loaded {len(images)} images')
    images_empty = dict()
    for id_, image in images.items():
        image = image._asdict()
        image['xys'] = np.zeros((0, 2), float)
        image['point3D_ids'] = np.full(0, -1, int)
        images_empty[id_] = Image(**image)
    write_images_binary(images_empty, empty_model / 'images.bin')
    shutil.copy(reference_model / 'cameras.bin', empty_model)    
    write_points3d_binary(dict(), empty_model / 'points3D.bin')        
    
def create_empty_model_from_nvm_and_database(nvm_path, database, empty_model, intrinsic_txt=None):
    empty_model = Path(empty_model)
    if empty_model.exists():
        print(f'Empty sfm {empty_model} existed.')
        return
    os.makedirs(empty_model)
    print(f'Creating an empty sfm under {empty_model} from {nvm_path}')
    
    # Construct images with fake points
    images_empty = dict()
    image_ids, camera_ids = load_ids_from_database(database)    
    images = load_images_from_nvm(nvm_path)
    for name in images:
        qvec = images[name]['qvec']
        tvec = images[name]['tvec']        
        name = name.lstrip('./')
        image_id = image_ids[name]
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_ids[name],
            name=name.replace('png', 'jpg'),     # Needed by RobotCar
            xys=np.zeros((0, 2), float),
            point3D_ids=np.full(0, -1, int)
        )
        images_empty[image_id] = image
    write_images_binary(images_empty, empty_model / 'images.bin')
    
    if intrinsic_txt and intrinsic_txt.exists():
        cameras = load_cameras_from_intrinsics_and_ids(intrinsic_txt, camera_ids)   # For aachen v1        
    else:
        cameras = load_cameras_from_database(database)
    write_cameras_binary(cameras, empty_model / 'cameras.bin')
    write_points3d_binary(dict(), empty_model / 'points3D.bin')    

def init_database_from_empty_model_binary(empty_model, database_path):
    if database_path.exists():
        print('Database already exists.')

    cameras = read_cameras_binary(str(model / 'cameras.bin'))
    images = read_images_binary(str(model / 'images.bin'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(
            model_id, camera.width, camera.height, camera.params, camera_id=i,
            prior_focal_length=True)

    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}

def covis_pairs_from_nvm(nvm_path, odir, topk=20):
    image_names = []
    image_ids_to_point_ids = {}
    point_ids_to_image_ids = {}

    with open(nvm_path, 'r') as f:
        # Skip headers
        line = next(f)
        while line == '\n' or line.startswith('NVM_V3'):
            line = next(f)

        # Load images
        num_images = int(line)
        for im_id in range(num_images):
            im_name = next(f).split()[0]
            im_name = im_name.lstrip('./').replace('png', 'jpg')      # For robotcar
            image_names.append(im_name)
            image_ids_to_point_ids[im_id] = []

        # Load 3D points
        line = next(f)
        while line == '\n':
            line = next(f)        
        num_points = int(line)
        for pid in range(num_points):
            line = next(f)
            data = line.split()
            num_measurements = int(data[6])
            point_ids_to_image_ids[pid] = []
            for j in range(num_measurements):
                im_id = int(data[7+j*4])
                im_name = image_names[im_id]
                image_ids_to_point_ids[im_id].append(pid)
                point_ids_to_image_ids[pid].append(im_id)
        print(f'Loaded {num_images} images {num_points} points.')
        
        
    # Covisible pairs    
    pairs = []
    if not os.path.exists(odir):
        os.makedirs(odir)
    out_txt = os.path.join(odir, f'pairs-db-covis{topk}.txt')
    with open(out_txt, 'w') as f:
        for im_id, im_name in enumerate(image_names):
            covis = defaultdict(int)
            visible_point_ids = image_ids_to_point_ids[im_id]
            for pid in visible_point_ids:
                covis_im_ids = point_ids_to_image_ids[pid]
                for cim_id in covis_im_ids:
                    if cim_id != im_id:
                        covis[cim_id] += 1

            if len(covis) == 0:
                print(f'Image {im_name} does not have any covisibility.')
                continue

            covis_ids = np.array(list(covis.keys()))
            covis_num = np.array([covis[i] for i in covis_ids])
            covis_ids_sorted = covis_ids[np.argsort(-covis_num)]
            top_covis_ids = covis_ids_sorted[:min(topk, len(covis_ids))]
            for i in top_covis_ids:
                im1, im2 = im_name, image_names[i]
                pairs.append((im1, im2))
                f.write(f'{im1} {im2}\n')
    print(f'Writing {len(pairs)} covis-{topk} pairs to {out_txt}' )
    
def covis_pairs_from_reference_model(reference_model, odir, topk=20):
    cameras, images, points3D = read_model(reference_model, '.bin')        
        
    # Covisible pairs    
    pairs = []
    if not os.path.exists(odir):
        os.makedirs(odir)
    out_txt = os.path.join(odir, f'pairs-db-covis{topk}.txt')
    with open(out_txt, 'w') as f:
        for im_id, image in images.items():
            covis = defaultdict(int)            
            visible_point_ids = image.point3D_ids[image.point3D_ids != -1]
            for pid in visible_point_ids:
                covis_im_ids = point_ids_to_image_ids[pid]
                for cim_id in points3D[pid].image_ids:
                    if cim_id != im_id:
                        covis[cim_id] += 1

            if len(covis) == 0:
                print(f'Image {image.name} does not have any covisibility.')
                continue

            covis_ids = np.array(list(covis.keys()))
            covis_num = np.array([covis[i] for i in covis_ids])
            covis_ids_sorted = covis_ids[np.argsort(-covis_num)]
            top_covis_ids = covis_ids_sorted[:min(topk, len(covis_ids))]
            for i in top_covis_ids:
                im1, im2 = image.name, images[i].name
                pairs.append((im1, im2))
                f.write(f'{im1} {im2}\n')
    print(f'Writing {len(pairs)} covis-{topk} pairs to {out_txt}' )    
