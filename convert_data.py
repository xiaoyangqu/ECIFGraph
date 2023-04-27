#%%
import numpy as np
import os
import sys
import shutil
import multiprocessing as mp
from Data_utils_wat import line_to_coor, gen_3D_2_pose_atomwise, file_to_gt_pose, get_wat0_feature, gen_3D_wat_atomwise, gen_3D_2_pose_atomwise_pl
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--apo", help="water network in apo state", default=False, action='store_true')
parser.add_argument("--bond_th", help="create a bond for the pair of atoms which distance less than bond_th", type=int, default = 6)
parser.add_argument("--pocket_th", help="the threshold of distance to the centroid of ligand to be considered as pocket", type=float, default = 12)
parser.add_argument("--output_file", help="output file name of this train/test data", type=str, default = None)
parser.add_argument("--start_iter", help="create training data from which random seed", type=int, default = 3)
parser.add_argument("--end_iter", help="create training data till which random seed", type=int, default = 14)
parser.add_argument("--thread_num", help="num of threads to creating dataset", type=int, default = 4)
parser.add_argument("--use_new_data", help="create data for predicting 3D coordinate", default=False, action='store_true')
parser.add_argument("--screen_data", help="If we generate data for screen", default=False, action='store_true')
parser.add_argument("--dataset_file", help="the path to label files", type=str, default='data/pdbbind')
parser.add_argument("--input_list", help="list of train/test pdbs", type=str, default='data/pdbbind/pdb_list_')
parser.add_argument("--groundtruth_dir", help="the path to the ground truth pose pdbbind files", type=str, default='data/pdbbind/')
args = parser.parse_args()

#args = parser.parse_args(['--input_list', 'data/pdb_list/pdb_list_', '--output_file', 'pdbbind_coor2', '--use_new_data', '--groundtruth_dir', 'data/pdbbind/','--dataset_file', 'pl_tmp', '--thread_num','32'])

#%%
def read_pdbbind_to_disk(input_list,
                        groundtruth_dir,
                        groundtruth_suffix,
                        output_dir,
                        apo,
                        tile_size,
                        bond_th,
                        pocket_th,
                        pdb_id_st,
                        pdb_id_ed,
                        seed = None):



    Atoms = ['N', 'C', 'O', 'S', 'Br', 'Cl', 'P', 'F', 'I']
    Bonds = ['1', '2', 'ar', 'am']


    pdb_list = []
    f_list = open(input_list, "r")
    for line in f_list:
        pdb_list.append(line.strip())
    f_list.close()

    pbar = tqdm(total=pdb_id_ed - pdb_id_st)
    pbar.set_description('Generating poses...')
    for i in range(pdb_id_st, pdb_id_ed):

        pdb = pdb_list[i]
        print('Running：',pdb)    
        
        try:
            protein_gt, ligand_gt, edge_gt, water_gt = file_to_gt_pose(groundtruth_dir, groundtruth_suffix, pdb, Atoms, Bonds, pocket_th)
            #(x, y, z), (name, x, y, z, atom, idx), (name, x, y, z, atom), (x, y)
            #num_nodes = gen_3D_2_pose_atomwise_pl(protein_gt, ligand_gt, Atoms, water_gt, edge_gt, bond_th, output_dir+"/"+str(pdb)) # if you want to get p-l graph only
            num_nodes = gen_3D_2_pose_atomwise(protein_gt, ligand_gt, Atoms, water_gt, edge_gt, bond_th, output_dir+"/"+str(pdb)) 
            if num_nodes != len(ligand_gt) + len(protein_gt) + len(water_gt): 
                print('Warning structure：',pdb)
                print(f"pose has {num_nodes} nodes while gt has {len(ligand_gt)} nodes.")
                print(f"P:{len(protein_gt)} L:{len(ligand_gt)} W:{len(water_gt)}")
                print([line[-1] for line in protein_gt])
            if apo:
                print("apo state")
                water0_gt = get_wat0_feature(groundtruth_dir, groundtruth_suffix, pdb, cut=4.0)
                print('yes=====')
                num_nodes_apo = gen_3D_wat_atomwise(water0_gt, bond_th, f'{output_dir}/{pdb}')
                print('======',num_nodes_apo)
        except:
            print(f'{pdb} Error!!!')
            continue
            
        pbar.update(1)
    pbar.close()


    print("3D data generated")

def mycopyfile(srcfile, dstpath): 
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  
        if os.path.exists(os.path.join(dstpath,fname)):
            pass
            shutil.move(srcfile, dstpath)  


def srand_data_load_save_coord2_thread(input_list, groundtruth_dir,  output_dir, apo, bond_th, pocket_th, thread_num, thread_id):
    tile_size = 1024
    output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
    if not os.path.isdir(output_dir_tmp):
        os.makedirs(output_dir_tmp)
    if not os.path.isdir(output_dir_tmp+'/train'):
        os.makedirs(output_dir_tmp+'/train')
    if not os.path.isdir(output_dir_tmp+'/test'):
        os.makedirs(output_dir_tmp+'/test')

    groundtruth_suffix = ['_protein.pdb','_ligand.mol2', '_wat4.pdb', '_wat0.pdb'] 

    splits = ['train', 'test']
    for split in splits:
        input_list_filename = input_list + split
        with open(input_list_filename, 'r') as gf:
            inputs = gf.readlines()
            start = (thread_id * len(inputs)) // thread_num
            end = ((thread_id + 1) * len(inputs)) // thread_num
        read_pdbbind_to_disk(input_list_filename,
                             groundtruth_dir,
                             groundtruth_suffix,
                             
                             output_dir_tmp+'/' + split,
                             apo,
                             tile_size,
                             bond_th,
                             pocket_th,
                             start, end)
        print('read done')                                                   


 
def srand_data_load_save_coord2(input_list, groundtruth_dir,  output_dir, apo, bond_th, pocket_th,  thread_num = 1):
    tile_size = 1024
    print("srand_data_load_save_coord2")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir+'/train'):
        os.makedirs(output_dir+'/train')
    if not os.path.isdir(output_dir+'/test'):
        os.makedirs(output_dir+'/test')

    print("data dir created!")
    # for i in range(4,5):
    if thread_num == 1:
        srand_data_load_save_coord2_thread(input_list, groundtruth_dir,  output_dir, apo, bond_th, pocket_th,  1, 0)
    else:
        p_list = []

        for thread_id in range(thread_num):
            p = mp.Process(target=srand_data_load_save_coord2_thread,
                           args=(input_list, groundtruth_dir,  output_dir, apo, bond_th, pocket_th,  thread_num, thread_id))
            p.start()
            p_list.append(p)
 
        for p in p_list:
            p.join()
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        # for split in ['test']:
        for split in ['train', 'test']:

            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            for f in dataset_file_list:
                file = output_dir_tmp+'/'+split + '/' + f
                mycopyfile(file,  f'{output_dir}/raw/{split}')


def data_in_thread_files(thread_num, output_dir):
    for thread_id in range(thread_num):
        output_dir_tmp = output_dir + '_tmp_' + str(thread_id)
        # for split in ['test']:
        for split in ['train', 'test']:
            # output dir of the data, with tread_id
            dataset_file_list = os.listdir(output_dir_tmp+'/'+split)
            print(dataset_file_list)
            for f in dataset_file_list:
                file = output_dir_tmp+'/'+split + '/' + f
                mycopyfile(file,  f'{output_dir}/raw/{split}')

def test_apo():
    try:
        if apo:
            print('apo',apo)
    except:
        print('error')



if __name__ == "__main__":
    apo = args.apo
    #print(apo)
    dataset_file = args.dataset_file
    
    #groundtruth_dir = '/gpfs/group/mtk2/cyberstar/hzj5142/AtomNet/pdbbind/'
    groundtruth_dir = args.groundtruth_dir
    bond_th = args.bond_th
    pocket_th = args.pocket_th
    output_file = args.output_file
    output_dir = dataset_file + '/' + output_file #MedusaGraph_tmp/pdbbind_rmsd_srand_coor2
    input_list = args.input_list 

    start = args.start_iter
    end = args.end_iter
    thread_num = args.thread_num
    use_new = args.use_new_data
    screen_data = args.screen_data

    if use_new:    

        srand_data_load_save_coord2(input_list, groundtruth_dir,  output_dir, apo, bond_th, pocket_th, thread_num = thread_num)

    else:
        print(f"Move data to {output_dir}/raw/")
        data_in_thread_files(thread_num, output_dir)






# %%
