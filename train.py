import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import prune
import torch.nn as nn
import numpy as np
from src.utils.io import export_pointcloud
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
import pdb
from tqdm import tqdm, trange
from torchinfo import summary
from torchviz import make_dot
#from thop import profile
#from thop import clever_format
#from thop.vision.basic_hooks import count_convNd
from src.models.fast_quant import fast_quant

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
model_file = cfg['test']['model_file'] 
exit_after = args.exit_after

print(cfg['data']['pointcloud_noise'])


model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_dataset = config.get_dataset('train', cfg)
#for i in train_dataset:
#    print(i['inputs.normals'].shape)
val_dataset = config.get_dataset('val', cfg, return_idx=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


model_counter = defaultdict(int)
data_vis_list = []
inputs=next(iter(train_loader))
#print([inputs['points'].size(), inputs['inputs'].size()])
# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)
#model = fast_quant(model, with_noisy_quant=True, percentile= True, search_mean= True, search_noisy=True)
#custom_ops = {torch.nn.Conv2d: count_convNd}
#macs, params = profile(model, inputs=(torch.rand(64, 2048, 3).to(device), torch.rand(64, 3000, 3).to(device), torch.rand((1,)).to(device)), custom_ops=custom_ops)
#macs, params = clever_format([macs, params], "%.3f")
#print(macs, params)
#print(model)
#print("macs: " + str(macs))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))
#size = tuple((64, 2048, 3, 64, 3000, 3, 1))
#print(size)
#make_dot(model(inputs['points'].to(device), inputs['inputs'].to(device)), params=dict(list(model.named_parameters()))).render(os.path.join(out_dir,"dp_d3_sc_cp_torchviz"), format="png")
#logger.add_graph(model,(torch.rand(64, 2048, 3).to(device), torch.rand(64, 3000, 3).to(device), torch.rand((1,)).to(device)))
#logger.close()
#summary(model,[inputs['points'].to(device).size(), inputs['inputs'].to(device).size(), (1,)], depth = 4)
# l = [module for module in model.modules() if isinstance(module, nn.Conv2d) or \
#                                             isinstance(module, nn.Conv1d) or \
#                                             isinstance(module, nn.Conv3d)]
# #print(l)
# parameters_to_prune = []
# #w=0
# for m in l:
#     parameters_to_prune += [(m, 'weight')]
#     #print(parameters_to_prune)
#     w = 100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement())
#     w = m.weight.shape
#     #print(str(m)+' -> '+str(w))
# parameters_to_prune = tuple(parameters_to_prune)
# print(parameters_to_prune[0])
# prune.global_unstructured(
#     parameters_to_prune,
#     pruning_method=prune.L1Unstructured,
#     amount=0.8,
# )
# l_pruned = [module for module in model.modules() if isinstance(module, nn.Conv2d) or \
#                                             isinstance(module, nn.Conv1d) or \
#                                             isinstance(module, nn.Conv3d)]
# pruned_parameters = []
# w=0
# for m in l_pruned:
#     pruned_parameters += [(m, 'weight')]
#     #print(pruned_parameters)
#     w = 100. * float(torch.sum(m.weight == 0))/ float(m.weight.nelement())
#     print("Sparsity in {}: {:.2f}%".format(str(m), w))


# Build a data dictionary for visualization
iterator = iter(vis_loader)
#print(len(vis_loader))
for i in trange(len(vis_loader)):
    data_vis = next(iterator)
    idx = data_vis['idx'].item()
    model_dict = val_dataset.get_model_dict(idx)
    category_id = model_dict.get('category', 'n/a')
    category_name = val_dataset.metadata[category_id].get('name', 'n/a')
    category_name = category_name.split(',')[0]
    if category_name == 'n/a':
        category_name = category_id

    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        data_vis_list.append({'category': category_name, 'it': c_it, 'data': data_vis})

    model_counter[category_id] += 1

# Model

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
print(f'out_dir {out_dir}')
try:
    print("train from file: "+str(model_file))
    load_dict = checkpoint_io.load(model_file)
except FileExistsError:
    print("new train")
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

# Generator
generator = config.get_generator(model, cfg, optimizer, device=device)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))



# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())

print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])


#while True:
for epoch in range(267): #epoch 13 scenes
    epoch_it += 1
#     scheduler.step()
#     t = torch.cuda.get_device_properties(0).total_memory
#     r = torch.cuda.memory_reserved(0)
#     a = torch.cuda.memory_allocated(0)
#     f = r - a  # free inside reserved
#     print(f'total (MB)    : {t / 1e6}')
#     print(f'free (MB)     : {f / 1e6}')
#     print(f'allocated (MB) : {a / 1e6}')
#     print(f'reserved (MB) : {r / 1e6}')
    #torch.cuda.memory_summary(device=None, abbreviated=False)
    for batch in tqdm(train_loader):
        it += 1
        loss = trainer.train_step(cfg, batch)
        logger.add_scalar('train/loss', loss, it)
        #if pl:
        #    pl = pl.detach().cpu().numpy()
        #    shape = pl.shape
        #    #print(pl[1][0][0])
        #    #print(pl[0][1][0])
        #    for i in range(shape[0]):
        #        for j in range(shape[1]):
        #            logger.add_scalars('planes/plane'+str(j), {'a':pl[i][j][0]}, it)
        #            logger.add_scalars('planes/plane'+str(j), {'b':pl[i][j][1]}, it)
        #            logger.add_scalars('planes/plane'+str(j), {'c':pl[i][j][2]}, it)
            
        # Print output/home/alcor/students/melis_tonti/new_model/configs/pointcloud/shapenet/shapenet_dynamic_3plane_final.yaml
        if print_every > 0 and (it % print_every) == 0:
            # print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2f'
            #       % (epoch_it, it, loss, time.time() - t0))
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
        #if 200 > 0 and (it % 00) == 0:
            print('Visualizing')
            for data_vis in data_vis_list:
                datas = data_vis['data']
                out = generator.generate_mesh(datas)#data_vis['data'])
                pointcloud = generator.generate_pointcloud(datas)#data_vis['data'])
                #print(out[0])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                pointcloud_path = os.path.join(out_dir, 'pointcloud')
                inputs_path = os.path.join(out_dir, 'input')
                occ_path = os.path.join(out_dir, 'occ')
                if not os.path.exists(pointcloud_path):
                    os.makedirs(pointcloud_path)
                if not os.path.exists(inputs_path):
                    print('make input folder:'+inputs_path)
                    os.makedirs(inputs_path)
                if not os.path.exists(occ_path):
                    print('make input folder:'+occ_path)
                    os.makedirs(occ_path)

                mesh.export(os.path.join(out_dir, 'vis', '{}_{}.obj'.format(data_vis['category'], data_vis['it'])))
                export_pointcloud(np.asarray(pointcloud.points), os.path.join(pointcloud_path, '{}_{}.ply'.format(data_vis['category'], data_vis['it'])))
                
                inputs_path = os.path.join(inputs_path, '{}_{}.ply'.format(data_vis['category'], data_vis['it']))
                inputs = datas['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                occ_path = os.path.join(occ_path, '{}_{}.ply'.format(data_vis['category'], data_vis['it']))
                occ = datas['points_iou'][datas['points_iou.occ'] == 1].squeeze(0).cpu().numpy()
                export_pointcloud(occ, occ_path, False)
                #print(datas['points_iou'][datas['points_iou.occ'] == 1])
                
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
        #if 100 > 0 and (it % 100) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)

    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    # f = r - a  # free inside reserved
    # print(f'total (MB)    : {t / 1e6}')
    # print(f'free (MB)     : {f / 1e6}')
    # print(f'allocated (MB) : {a / 1e6}')
    # print(f'reserved (MB) : {r / 1e6}')
