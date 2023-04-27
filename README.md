# Water Network-Augmented Two-State Model
## ECIFGraph::HM-Holo-Apo Scoring Function

Water network rearrangement, transitioning from the ligand-unbound state to the ligand-bound state, has been shown to significantly impact protein-ligandbinding   interactions. Despite this, current machine learning-based scoring functions fail to account for these effects. Our study aims to address this challengeby incorporating water network rearrangement information into deep learning-based scoring functions for protein-ligand binding prediction.<br>
Here, we attempt to construct a comprehensive representation that incorporates water network information into both ligand-unbound and bound states. Specifically,we integrated extended connectivity interaction features into the graph representation, while employing a graph transformer operator to extract features fromboth    ligand-unbound and bound states. Consequently, we devised a water network-augmented two-state model, designated as ECIFGraph::HM-Holo-Apo, which exhibitssatisfactory performance for scoring, ranking, docking, (reverse) screening, power tests on the CASF-2016 benchmark, and superior performance in larger-scaledocking-based virtual screening tests on the DEKOIS2.0 dataset. Our study emphasizes that the use of a water network-augmented two-state model can effectivelyenhance the robustness of machine learning-based scoring functions.



## CONTENTS

### Usage
    0. Setup dependencies
    conda env create -f environment-data.yml
    conda env create -f environment-train.yml
    
    1. Generate hydration sites in apo and holo states by HydraMap analysis:
    sh gen_hydramap.sh apo
    sh gen_hydramap.sh holo
    
    2. Generate dataset for graph representation:
    conda activate data
    python convert_data.py  --input_list=data/pdb_list/pdb_list_ --output_file=pdbbind_coor2 --use_new_data --groundtruth_dir=data/pdbbind/ --dataset_file=Graph_tmp --apo --thread_num=32

    3. Train water network-augmented two-state model:
    conda activate train
    python train-gpu.py --d_graph_layer 256 --n_graph_layer 4 --dropout_rate 0.3 --data_path Graph_tmp/pdbbind_coor2 --output ./output.txt --pred_output ./prediction.txt --batch_size 64 --processed processed --graph_pooling sum --apo

    4. To run trained model, please use the following command line：
    python train-gpu.py --d_graph_layer 256 --n_graph_layer 4 --dropout_rate 0.3 --data_path Graph_tmp/pdbbind_coor2 --pred_output ./prediction.txt --processed processed-score --graph_pooling sum --apo --pre_model model/ECIFGraph-Holo-Apo.pt --test_mode 

   
### Folders
    
    data/pdb_list: PDBID list file and experimental binding affinity file
    data/pdbbind: Protein & ligand structure files
    
### Notes:

    1. Protein PDB files are assumed to contain coordinates for all heavy atoms
    
