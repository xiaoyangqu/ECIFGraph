# Water Network-Augmented Two-State Model
ECIFGraph::HM-Holo-Apo Scoring Function


CONTENTS

Notebooks

    1. Generate hydration sites in apo and holo states by HydraMap analysis:
    sh gen_hydramap.sh apo
    sh gen_hydramap.sh holo
    
    2. Generate dataset for graph representation:
    python convert_data.py  --input_list=data/pdb_list/pdb_list_ --output_file=pdbbind_coor2 --use_new_data --groundtruth_dir=data/pdbbind/ --dataset_file=Graph_tmp --apo --thread_num=32

    3. Train water network-augmented two-state model:
    python train-gpu.py --d_graph_layer 256 --n_graph_layer 4 --dropout_rate 0.3 --data_path Graph_tmp/pdbbind_coor2 --output ./output.txt --pred_output ./prediction.txt --batch_size 64 --processed processed --graph_pooling sum --apo

    4. To run ECIFGraph::HM-Holo-Apo, please use the following command line：
    python train-gpu.py --d_graph_layer 256 --n_graph_layer 4 --dropout_rate 0.3 --data_path Graph_tmp/pdbbind_coor2 --pred_output ./prediction.txt --processed processed-score --graph_pooling sum --apo --pre_model model/ECIFGraph-Holo-Apo.pt --test_mode 

   
Folders
    
    data/pdb_list: PDBID list file and experimental binding affinity file
    data/pdbbind: Protein & ligand structure files
    
Notes:

    1. Protein PDB files are assumed to contain coordinates for all heavy atoms
    
