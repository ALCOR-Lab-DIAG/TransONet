import torch
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import torch.nn as nn
from torch.nn.utils import prune
import torch.optim as optim
from src import config
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.visualize import visualize_data
#from thop import profile, clever_format

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
print(device)
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.csv')
out_time_file_class = os.path.join(generation_dir, 'time_generation.csv')

batch_size = cfg['generation']['batch_size']
input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)
#print(dataset)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)
#model = config.get_model(cfg, device=device, dataset=train_dataset)
#macs, params = profile(model, inputs=(torch.rand(64, 2048, 3).to(device), torch.rand(64, 3000, 3).to(device), torch.rand((1,)).to(device)))
#macs, params = clever_format([macs, params], "%.3f")
#print(macs, params)
#print(model)
#print("macs: " + str(macs))
#l = [module for module in model.modules() if isinstance(module, nn.Conv2d) or \
#                                            isinstance(module, nn.Conv1d) or \
#                                            isinstance(module, nn.Conv3d)]
##print(l)
#parameters_to_prune = []
##w=0
#for m in l:
#    parameters_to_prune += [(m, 'weight')]
#    #print(parameters_to_prune)
#    #w = 100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement())
#    #print(str(m)+' -> '+str(w))
#parameters_to_prune = tuple(parameters_to_prune)
##print(parameters_to_prune[0])
#prune.global_unstructured(
#    parameters_to_prune,
#    pruning_method=prune.L1Unstructured,
#    amount=0.8,
#)

checkpoint_io = CheckpointIO(out_dir, model=model)
#checkpoint_io = CheckpointIO('out/pointcloud/shapenet_dynamic_3plane_final/model_best.pt', model=model)
print(cfg['test']['model_file'])
checkpoint_io.load(cfg['test']['model_file'])

# Generator
optimizer = optim.Adam(model.parameters(), lr=1e-3)
generator = config.get_generator(model, cfg, optimizer, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']
#print(generate_pointcloud)

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)
# Statistics
time_dicts = []

# Generate
model.eval()

# Count how many models already created
model_counter = defaultdict(int)

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis')
    target_dir = os.path.join(generation_dir, 'target')
    

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    #print(model_dict)
    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    if generate_mesh and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    #print(time_dict)
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:

    # Also copy ground truth
        if cfg['generation']['copy_groundtruth']:
            modelpath = os.path.join(
                dataset.dataset_folder, category_id, modelname, 
                cfg['data']['watertight_file'])
            out_file_dict['gt'] = modelpath

        if generate_mesh:
            t0 = time.time()
            #print(t0)
            out = generator.generate_mesh(data)
            time_dict['mesh'] = time.time() - t0
            #print(time.time())
            #print(time_dict['mesh'])

            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
            time_dict.update(stats_dict)

            # Write output
            mesh_out_file = os.path.join(mesh_dir, '%s.obj' % modelname)
            mesh.export(mesh_out_file)
            
            out_file_dict['mesh'] = mesh_out_file

        if generate_pointcloud:
            t0 = time.time()
            pointcloud = generator.generate_pointcloud(data)
            time_dict['pcl'] = time.time() - t0
            pointcloud_out_file = os.path.join(
                pointcloud_dir, '%s.ply' % modelname)
            export_pointcloud(np.asarray(pointcloud.points), pointcloud_out_file)
            out_file_dict['pointcloud'] = pointcloud_out_file

        if cfg['generation']['copy_input']:
            # Save inputs
            if input_type == 'pointcloud' or 'partial_pointcloud':
                inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                inputs = data['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                out_file_dict['in'] = inputs_path

        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category_id]
        if c_it < vis_n_outputs:
            # Save output files
            img_name = '%02d.obj' % c_it
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                        % (c_it, k, ext))
                shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_csv(out_time_file)
print(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class id']).mean()
time_df_class.to_csv(out_time_file_class)
print(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
