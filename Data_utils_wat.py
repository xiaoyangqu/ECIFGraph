#%%
import numpy as np
import json
from scipy.spatial import distance
import pandas as pd

from rdkit import Chem
from pymol import cmd, stored
import os
#%%
SPACE = 100.0
COV_BOND_TH = 2.5

pro_node_feat = {
"ALA-C" : "C;4;3;0;0;0","ALA-CA" : "C;4;3;1;0;0","ALA-CB" : "C;4;1;3;0;0","ALA-N" : "N;3;2;1;0;0","ALA-O" : "O;2;1;0;0;0","ALA-OXT" : "O;2;1;0;0;0","ARG-C" : "C;4;3;0;0;0","ARG-CA" : "C;4;3;1;0;0" \
,"ARG-CB" : "C;4;2;2;0;0","ARG-CD" : "C;4;2;2;0;0","ARG-CG" : "C;4;2;2;0;0","ARG-CZ" : "C;6;3;0;0;0","ARG-N" : "N;3;2;1;0;0","ARG-NE" : "N;4;2;1;0;0","ARG-NH1" : "N;4;1;2;0;0","ARG-NH2" : "N;4;1;2;0;0" \
,"ARG-O" : "O;2;1;0;0;0","ARG-OXT" : "O;2;1;0;0;0","ASN-C" : "C;4;3;0;0;0","ASN-CA" : "C;4;3;1;0;0","ASN-CB" : "C;4;2;2;0;0","ASN-CG" : "C;4;3;0;0;0","ASN-N" : "N;3;2;1;0;0","ASN-ND2" : "N;3;1;2;0;0" \
,"ASN-O" : "O;2;1;0;0;0","ASN-OD1" : "O;2;1;0;0;0","ASN-OXT" : "O;2;1;0;0;0","ASP-C" : "C;4;3;0;0;0","ASP-CA" : "C;4;3;1;0;0","ASP-CB" : "C;4;2;2;0;0","ASP-CG" : "C;5;3;0;0;0","ASP-N" : "N;3;2;1;0;0" \
,"ASP-O" : "O;2;1;0;0;0","ASP-OD1" : "O;2;1;0;0;0","ASP-OD2" : "O;2;1;0;0;0","ASP-OXT" : "O;2;1;0;0;0","CYS-C" : "C;4;3;0;0;0","CYS-CA" : "C;4;3;1;0;0","CYS-CB" : "C;4;2;2;0;0","CYS-N" : "N;3;2;1;0;0" \
,"CYS-O" : "O;2;1;0;0;0","CYS-OXT" : "O;2;1;0;0;0","CYS-SG" : "S;2;1;1;0;0","GLN-C" : "C;4;3;0;0;0","GLN-CA" : "C;4;3;1;0;0","GLN-CB" : "C;4;2;2;0;0","GLN-CD" : "C;4;3;0;0;0","GLN-CG" : "C;4;2;2;0;0" \
,"GLN-N" : "N;3;2;1;0;0","GLN-NE2" : "N;3;1;2;0;0","GLN-O" : "O;2;1;0;0;0","GLN-OE1" : "O;2;1;0;0;0","GLN-OXT" : "O;2;1;0;0;0","GLU-C" : "C;4;3;0;0;0","GLU-CA" : "C;4;3;1;0;0","GLU-CB" : "C;4;2;2;0;0" \
,"GLU-CD" : "C;5;3;0;0;0","GLU-CG" : "C;4;2;2;0;0","GLU-N" : "N;3;2;1;0;0","GLU-O" : "O;2;1;0;0;0","GLU-OE1" : "O;2;1;0;0;0","GLU-OE2" : "O;2;1;0;0;0","GLU-OXT" : "O;2;1;0;0;0","GLY-C" : "C;4;3;0;0;0" \
,"GLY-CA" : "C;4;2;2;0;0","GLY-N" : "N;3;2;1;0;0","GLY-O" : "O;2;1;0;0;0","GLY-OXT" : "O;2;1;0;0;0","HIS-C" : "C;4;3;0;0;0","HIS-CA" : "C;4;3;1;0;0","HIS-CB" : "C;4;2;2;0;0","HIS-CD2" : "C;4;2;1;1;1" \
,"HIS-CE1" : "C;4;2;1;1;1","HIS-CG" : "C;4;3;0;1;1","HIS-N" : "N;3;2;1;0;0","HIS-ND1" : "N;3;2;0;1;1","HIS-NE2" : "N;3;2;1;1;1","HIS-O" : "O;2;1;0;0;0","HIS-OXT" : "O;2;1;0;0;0","ILE-C" : "C;4;3;0;0;0" \
,"ILE-CA" : "C;4;3;1;0;0","ILE-CB" : "C;4;3;1;0;0","ILE-CD1" : "C;4;1;3;0;0","ILE-CG1" : "C;4;2;2;0;0","ILE-CG2" : "C;4;1;3;0;0","ILE-N" : "N;3;2;1;0;0","ILE-O" : "O;2;1;0;0;0","ILE-OXT" : "O;2;1;0;0;0" \
,"LEU-C" : "C;4;3;0;0;0","LEU-CA" : "C;4;3;1;0;0","LEU-CB" : "C;4;2;2;0;0","LEU-CD1" : "C;4;1;3;0;0","LEU-CD2" : "C;4;1;3;0;0","LEU-CG" : "C;4;3;1;0;0","LEU-N" : "N;3;2;1;0;0","LEU-O" : "O;2;1;0;0;0" \
,"LEU-OXT" : "O;2;1;0;0;0","LYS-C" : "C;4;3;0;0;0","LYS-CA" : "C;4;3;1;0;0","LYS-CB" : "C;4;2;2;0;0","LYS-CD" : "C;4;2;2;0;0","LYS-CE" : "C;4;2;2;0;0","LYS-CG" : "C;4;2;2;0;0","LYS-N" : "N;3;2;1;0;0" \
,"LYS-NZ" : "N;4;1;3;0;0","LYS-O" : "O;2;1;0;0;0","LYS-OXT" : "O;2;1;0;0;0","MET-C" : "C;4;3;0;0;0","MET-CA" : "C;4;3;1;0;0","MET-CB" : "C;4;2;2;0;0","MET-CE" : "C;4;1;3;0;0","MET-CG" : "C;4;2;2;0;0" \
,"MET-N" : "N;3;2;1;0;0","MET-O" : "O;2;1;0;0;0","MET-OXT" : "O;2;1;0;0;0","MET-SD" : "S;2;2;0;0;0","PHE-C" : "C;4;3;0;0;0","PHE-CA" : "C;4;3;1;0;0","PHE-CB" : "C;4;2;2;0;0","PHE-CD1" : "C;4;2;1;1;1" \
,"PHE-CD2" : "C;4;2;1;1;1","PHE-CE1" : "C;4;2;1;1;1","PHE-CE2" : "C;4;2;1;1;1","PHE-CG" : "C;4;3;0;1;1","PHE-CZ" : "C;4;2;1;1;1","PHE-N" : "N;3;2;1;0;0","PHE-O" : "O;2;1;0;0;0","PHE-OXT" : "O;2;1;0;0;0" \
,"PRO-C" : "C;4;3;0;0;0","PRO-CA" : "C;4;3;1;0;1","PRO-CB" : "C;4;2;2;0;1","PRO-CD" : "C;4;2;2;0;1","PRO-CG" : "C;4;2;2;0;1","PRO-N" : "N;3;3;0;0;1","PRO-O" : "O;2;1;0;0;0","PRO-OXT" : "O;2;1;0;0;0" \
,"SER-C" : "C;4;3;0;0;0","SER-CA" : "C;4;3;1;0;0","SER-CB" : "C;4;2;2;0;0","SER-N" : "N;3;2;1;0;0","SER-O" : "O;2;1;0;0;0","SER-OG" : "O;2;1;1;0;0","SER-OXT" : "O;2;1;0;0;0","THR-C" : "C;4;3;0;0;0" \
,"THR-CA" : "C;4;3;1;0;0","THR-CB" : "C;4;3;1;0;0","THR-CG2" : "C;4;1;3;0;0","THR-N" : "N;3;2;1;0;0","THR-O" : "O;2;1;0;0;0","THR-OG1" : "O;2;1;1;0;0","THR-OXT" : "O;2;1;0;0;0","TRP-C" : "C;4;3;0;0;0" \
,"TRP-CA" : "C;4;3;1;0;0","TRP-CB" : "C;4;2;2;0;0","TRP-CD1" : "C;4;2;1;1;1","TRP-CD2" : "C;4;3;0;1;1","TRP-CE2" : "C;4;3;0;1;1","TRP-CE3" : "C;4;2;1;1;1","TRP-CG" : "C;4;3;0;1;1","TRP-CH2" : "C;4;2;1;1;1" \
,"TRP-CZ2" : "C;4;2;1;1;1","TRP-CZ3" : "C;4;2;1;1;1","TRP-N" : "N;3;2;1;0;0","TRP-NE1" : "N;3;2;1;1;1","TRP-O" : "O;2;1;0;0;0","TRP-OXT" : "O;2;1;0;0;0","TYR-C" : "C;4;3;0;0;0","TYR-CA" : "C;4;3;1;0;0" \
,"TYR-CB" : "C;4;2;2;0;0","TYR-CD1" : "C;4;2;1;1;1","TYR-CD2" : "C;4;2;1;1;1","TYR-CE1" : "C;4;2;1;1;1","TYR-CE2" : "C;4;2;1;1;1","TYR-CG" : "C;4;3;0;1;1","TYR-CZ" : "C;4;3;0;1;1","TYR-N" : "N;3;2;1;0;0" \
,"TYR-O" : "O;2;1;0;0;0","TYR-OH" : "O;2;1;1;0;0","TYR-OXT" : "O;2;1;0;0;0","VAL-C" : "C;4;3;0;0;0","VAL-CA" : "C;4;3;1;0;0","VAL-CB" : "C;4;3;1;0;0","VAL-CG1" : "C;4;1;3;0;0","VAL-CG2" : "C;4;1;3;0;0" \
,"VAL-N" : "N;3;2;1;0;0","VAL-O" : "O;2;1;0;0;0","VAL-OXT" : "O;2;1;0;0;0"}

Atoms = ['N', 'C', 'O', 'S', 'Br', 'Cl', 'P', 'F', 'I']

three_one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y', \
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A', \
    'GLY':'G', 'PRO':'P', 'CYS':'C'}
three_letter_lower = [ k.lower() for k,v in three_one_letter.items()]
three_letter_lower.sort()
three_letter = list(three_one_letter.keys())
three_letter.sort()

#water0_gt.append((name, x, y, z, atom, idx, 'O;2;0;2;0;0'))
def get_wat0_feature(groundtruth_dir, groundtruth_suffix, pdb, cut=4.0):
    protein_f = groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[0]
    wat_f = groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[3]
    cmd.delete("all")
    cmd.load(protein_f)
    cmd.load(wat_f)
    pro=f"{os.path.splitext(os.path.basename(protein_f))[0]}"
    wat=f"{os.path.splitext(os.path.basename(wat_f))[0]}"
    #for w in cmd.select(wat):
    cmd.create('poc', f'br. {pro} w. {cut} of {wat}')


    stored.list = []
    cmd.iterate_state(1,f"(name o and {wat})","stored.list.append([name, x, y, z, name, resi])")
    #print(stored.list)

    resi_poc=[stored.list[i][0] for i in range(len(stored.list))]
    resi_poc.sort()
    #print(resi_poc)
    

    dict_count_wat={}
    for O in range(1,len(stored.list)+1):
        myspace = {'myfunc': []}
        cmd.iterate(f'br. {pro} w. 4 of resi {O} in {wat}', 'myfunc.append((resi,resn))', space = myspace)
        #print(myspace['myfunc'])
        tmp = list(set(myspace['myfunc']))
        #print(tmp)
        resi_poc=[tmp[i][1] for i in range(len(tmp))]
        occ = [resi_poc.count(x) for x in three_letter]
        stored.list[O-1].append(occ)

    return stored.list



#%%


def GetAtomType(atom):
    
    AtomType = [atom.GetSymbol(),
                str(atom.GetExplicitValence()),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                str(int(atom.GetIsAromatic())),
                str(int(atom.IsInRing())), 
               ]

    return(";".join(AtomType))


#%%
def line_to_coor(line, form):
    if form == 'protein':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = st[3]
        idx = int(line[22:26])
        return name, x, y, z, atom, idx

    if form == 'protein_atom_ecif':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        if st[-1].isalpha():
            atom = st[-1]
        else:
            atom = line[13:14]
        idx = int(line[22:26])
        aa_name = line[17:20]+"-"+line[12:16].replace(" ","")
        #ecif = pro_node_feat[aa_name]
        return name, x, y, z, atom, idx, aa_name
    if form == 'protein_atom':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = st[-1]
        idx = int(line[22:26])
        return name, x, y, z, atom, idx
    if form == 'ligand_pdb':
        st = line.split()
        name = line[12:17].strip().strip('\'')
        x = float(line[30:38]) # - c_x
        y = float(line[38:46]) # - c_y
        z = float(line[46:54]) # - c_z
        atom = st[-1]
    elif form == 'ligand_mol2':
        st = line.split()
        name = st[1].strip('\'') #移除字头尾的 '
        x = float(line[16:26])
        y = float(line[26:36])
        z = float(line[36:46])
        atom = st[5]
        atom = atom.split('.')[0]

    if len(name) > 3:
        name = name[:3]
    while not atom[0].isalpha():
        atom = atom[1:]
    while not atom[-1].isalpha():
        atom = atom[:-1]
    return name, x, y, z, atom


def centre_of_pocket(ligand):
    x = sum(line[1] for line in ligand) / len(ligand)
    y = sum(line[2] for line in ligand) / len(ligand)
    z = sum(line[3] for line in ligand) / len(ligand)
    
    return x, y, z



def file_to_gt_pose(groundtruth_dir, groundtruth_suffix, pdb, Atoms, Bonds, pocket_th): 

    atoms_idx = []
    protein_gt = []
    ligand_gt = []
    edge_gt = set()

    pocket_idx = []
    water_gt = []
    water_idx = []
    water0_gt = []
    water0_idx = []

    #'''
    try:
        ligand_file =  groundtruth_dir+'/'+pdb+'/'+pdb+'_ligand.sdf'
        m = Chem.MolFromMolFile(ligand_file, sanitize=False)
        m.UpdatePropertyCache(strict=False)
    except:
        ligand_file = groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[1]
        m = Chem.MolFromMol2File(ligand_file, sanitize=False)
        m.UpdatePropertyCache(strict=False)
    else:
        print(f'{pdb} ligand File is Normal')
    #m.UpdatePropertyCache(strict=False)
    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            #print('atom',atom)
            idx = int(atom.GetIdx())+1
            entry = [idx]
            #entry.append(GetAtomType(atom))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())    
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            entry.append(atom.GetSymbol())
            entry.append(GetAtomType(atom))
            atoms_idx.append(idx)
            neighbors = atom.GetNeighbors()
            edge_gt.update({(idx-1,nidx) for nidx in [ x.GetIdx() for x in neighbors if x.GetSymbol() != "H"]})
            #print('neighbor',idx,[(idx-1, nidx) for x in neighbors for nidx in x.GetIdx()])
            ligand_gt.append(tuple(entry))
    #print('neighbor',edge_gt)
    #print('neighbor',len(edge_gt))
    #print(ligand_gt)
    #print(atoms_idx)
    

    cx, cy, cz = centre_of_pocket(ligand_gt)
    #protein

    f = open(groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[0], 'r') #protein.pdb
    for st in f:
        ss = st.split()
        if (ss[0] == 'ATOM'): 
            name, x, y, z, atom, idx, aa_name = line_to_coor(st, 'protein_atom_ecif') #CA x y z C 99
            if (atom != 'H'):
                if aa_name in pro_node_feat:
                    ecif = pro_node_feat[aa_name]

                else:
                    ecif = "X;0;0;0;0;0"
                    print(f'{aa_name} in {pdb} is unknown atom-type!!')
                protein_gt.append((name, x, y, z, atom, idx, ecif))

                if name == 'CA' and distance.euclidean([x, y, z], [cx, cy, cz]) < pocket_th:
                    pocket_idx.append(idx)
                    
    f.close()
    protein_gt = [line for line in protein_gt if line[5] in pocket_idx]


    #water
    f = open(groundtruth_dir+'/'+pdb+'/'+pdb+groundtruth_suffix[2], 'r')
    for st in f:
        ss = st.split()
        if (ss[0] == 'ATOM'):
            name, x, y, z, atom, idx = line_to_coor(st, 'protein_atom')
            #print("water",name, x, y, z, atom, idx)
            if (atom != 'H'):
                water_gt.append((name, x, y, z, atom, idx, 'O;2;0;2;0;0'))
                if name == 'O' and distance.euclidean([x, y, z], [cx, cy, cz]) < pocket_th:
                    water_idx.append(idx)

    f.close()
    water_gt = [line for line in water_gt if line[5] in water_idx]

    #gt_pose = gen_3D_2_gt_pose(protein_gt, ligand_gt, Atoms, file_dir=None, use_protein=False) #edge_gt only contains ligand 
    return protein_gt, ligand_gt, edge_gt, water_gt
    

#%%
a = np.array([1,2,3])
b = np.array([1,1,1])
a-b
#%%
def D3_info(a, b, c):

    ab = b - a 
    ac = c - a 
    bc = b - c 
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)

    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # Euclidean distance
    bc_ = np.sqrt(np.sum(bc ** 2))
    sidelength = ab_ + ac_ + bc_ #side length
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, sidelength

def rou(dis):
    dis = round(dis*100000) / 100000
    return dis

def gen_3D_wat_atomwise(water, bond_th, file_dir):
    nodes = []

    node_index = []
    edges = []
    dist = []

    feats = []
    feats2 = []
    node_id = 0

    for line in water:
        res_feat = line[-1]

        x = line[1]
        y = line[2]
        z = line[3]

        nodes.append(node_id)
        feat = np.array([node_id,x / SPACE, y / SPACE, z / SPACE])
        feat2 = np.array([int(i) for i in res_feat])
        feats.append(feat)
        feats2.append(feat2)
        node_id += 1

    assert node_id == len(nodes) 

    tot = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            c_i = feats[i][-3:]
            c_j = feats[j][-3:]
            dis = distance.euclidean(c_i, c_j)
            dis = round(dis*100000) / 100000

            if dis * SPACE < bond_th: #if < 6A
                # G.add_edge(i, j)
                if dis *SPACE < COV_BOND_TH * 1.4: #2.5*1.4 = 3.5
                    edges.append(j)
                    tot += 1
                    Angles = []
                    Areas = []
                    Distances = []
                    for k in nodes:
                        if k == i or k == j:
                            continue
                        c_k = feats[k][-3:]
                        angle, area, sidelength = D3_info(c_i, c_k, c_j)
                        Angles.append(angle)
                        Areas.append(area)
                        Distances.append(sidelength)
                        
                        triangle = [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                        np.mean(Areas), np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
                        triangle = [rou(i) for i in triangle]
                    dist.append([dis]+triangle) 

    

        node_index.append(tot)   
     
    with open(file_dir+"_data-G-wat0.json", 'a') as f:
        json.dump(node_index, f)
        f.write('\n')
        json.dump(edges, f)
        f.write('\n')
        json.dump(dist, f)
        f.write('\n')


    with open(file_dir+"_data-feats-wat0", 'ab') as f:
        np.save(f, feats2)

    df = pd.DataFrame()
    df['node_features']=feats2
    df.to_csv(file_dir+"_data-feats-label-wat0.csv", index=False)

    return len(nodes)

#%%
def gen_3D_2_pose_atomwise(protein, ligand, Atoms, water, edge_gt, bond_th, file_dir):

    print('Convert a pose with pdb format to graph format')

    nodes = []

    node_index = []
    edges = []
    dist = []

    feats = []
    feats2 = []
    node_id = 0

    la = len(Atoms)

    for line in ligand:
        atom, valence, heavy_atom, hydrogen, aromaticity, ring = line[5].split(';')
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + 3 * la)
        feat[Atoms.index(atom) + la] = 1 #one hot
        feat2 = np.concatenate((feat, np.array([int(i) for i in [valence, heavy_atom, hydrogen, aromaticity, ring]])))

        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat) 
        feats2.append(feat2)
        node_id += 1

    ligand_nodes = node_id

    for line in protein:
        atom, valence, heavy_atom, hydrogen, aromaticity, ring = line[6].split(';')

        x = line[1]
        y = line[2]
        z = line[3]

        nodes.append(node_id)

        feat = np.zeros(3 + 3 * la)
        feat[Atoms.index(atom)] = 1
        feat2 = np.concatenate((feat, np.array([int(i) for i in [valence, heavy_atom, hydrogen, aromaticity, ring]])))
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat)
        feats2.append(feat2)
        node_id += 1

    protein_nodes = node_id #new

    for line in water:
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + 3 * la)
        feat[Atoms.index(atom) + 2 * la] = 1
        feat2 = np.concatenate((feat, np.array([2,0,2,0,0])))
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]

        feats.append(feat)
        feats2.append(feat2)
        node_id += 1

    assert node_id == len(nodes) 
    tot = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            c_i = feats[i][-3:]
            c_j = feats[j][-3:]
            dis = distance.euclidean(c_i, c_j)
            dis = round(dis*100000) / 100000
            if dis * SPACE < bond_th: #if < 6A
                # G.add_edge(i, j)
                if i < ligand_nodes and j < ligand_nodes: #i,j同属于配体
                    if (i, j) in edge_gt:
                        edges.append(j)
                        tot += 1
                        dist.append([dis, 0.0, 0.0, 0.0, 0.0, 0.0])
                        # TODO: covalent bond
                elif i >= ligand_nodes and j >= ligand_nodes:#i，j同属蛋白
                    if dis *SPACE < COV_BOND_TH: #2.5
                        edges.append(j)
                        tot += 1
                        dist.append([0.0, 0.0, dis, 0.0, 0.0, 0.0])
                elif i < ligand_nodes and protein_nodes > j >=ligand_nodes or protein_nodes > i >= ligand_nodes and j < ligand_nodes: # i属于配体 j属于蛋白
                    pass
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, dis, 0.0, 0.0, 0.0, 0.0])                
                elif i >= protein_nodes and j >= protein_nodes: # i属于水 j属于水
                    if dis *SPACE < COV_BOND_TH * 1.4: #2.5*1.4 = 3.5
                        edges.append(j)
                        tot += 1
                        dist.append([0.0, 0.0, 0.0, dis, 0.0, 0.0])                   
                elif i >= protein_nodes and  j < ligand_nodes or i >= protein_nodes and  j < ligand_nodes: #i属于水 j属于配体
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, 0.0, 0.0, 0.0, dis, 0.0])    
                    
                elif i >= protein_nodes and protein_nodes > j >= ligand_nodes or protein_nodes > i >= ligand_nodes and j >= protein_nodes: # i属于水 j属于蛋白
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, 0.0, 0.0, 0.0, 0.0, dis])  

        node_index.append(tot)



  
    with open(file_dir+"_data-G.json", 'a') as f:
        json.dump(node_index, f)
        f.write('\n')
        json.dump(edges, f)
        f.write('\n')
        json.dump(dist, f)
        f.write('\n')

    with open(file_dir+"_data-feats", 'ab') as f:
        np.save(f, feats2)


    df = pd.DataFrame()
    df['node_features']=feats2
    df.to_csv(file_dir+"_data-feats-label.csv", index=False)

    return len(nodes)


def gen_3D_2_pose_atomwise_pl(protein, ligand, Atoms, water, edge_gt, bond_th, file_dir):
    """ Convert a pose with pdb format to graph forma. 
    """

    print('Convert a pose with pdb format to graph format')

    nodes = []

    node_index = []
    edges = []
    dist = []

    feats = []
    feats2 = []
    node_id = 0

    la = len(Atoms)

    for line in ligand:
        atom, valence, heavy_atom, hydrogen, aromaticity, ring = line[5].split(';')
        x = line[1]
        y = line[2]
        z = line[3]
        atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + 3 * la)
        feat[Atoms.index(atom) + la] = 1 #one hot
        feat2 = np.concatenate((feat, np.array([int(i) for i in [valence, heavy_atom, hydrogen, aromaticity, ring]])))

        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat) #[[0,,,,1,0,0,x,y,z],[]]
        feats2.append(feat2)
        node_id += 1

    ligand_nodes = node_id

    for line in protein:
        atom, valence, heavy_atom, hydrogen, aromaticity, ring = line[6].split(';')

        x = line[1]
        y = line[2]
        z = line[3]
        ##atom = line[4]

        nodes.append(node_id)
        feat = np.zeros(3 + 3 * la)
        feat[Atoms.index(atom)] = 1
        feat2 = np.concatenate((feat, np.array([int(i) for i in [valence, heavy_atom, hydrogen, aromaticity, ring]])))
        feat[-3:] = [x / SPACE, y / SPACE, z / SPACE]
        feats.append(feat)
        feats2.append(feat2)
        node_id += 1

    protein_nodes = node_id #new

    assert node_id == len(nodes) 

    tot = 0
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            c_i = feats[i][-3:]
            c_j = feats[j][-3:]
            dis = distance.euclidean(c_i, c_j)
            dis = round(dis*100000) / 100000
            if dis * SPACE < bond_th: #if < 6A
                # G.add_edge(i, j)
                if i < ligand_nodes and j < ligand_nodes: #i,j同属于配体
                    if (i, j) in edge_gt:
                        edges.append(j)
                        tot += 1
                        dist.append([dis, 0.0, 0.0])
                        # TODO: covalent bond
                elif i >= ligand_nodes and j >= ligand_nodes:#i，j同属蛋白
                    if dis *SPACE < COV_BOND_TH: #2.5
                        edges.append(j)
                        tot += 1
                        dist.append([0.0, 0.0, dis])
                #elif i < ligand_nodes and protein_nodes > j >=ligand_nodes or protein_nodes > i >= ligand_nodes and j < ligand_nodes: # i属于配体 j属于蛋白
                else:    
                    edges.append(j)
                    tot += 1
                    dist.append([0.0, dis, 0.0])                


        node_index.append(tot)

  
    with open(file_dir+"_data-G.json", 'a') as f:
        json.dump(node_index, f)
        f.write('\n')
        json.dump(edges, f)
        f.write('\n')
        json.dump(dist, f)
        f.write('\n')


    with open(file_dir+"_data-feats", 'ab') as f:
        np.save(f, feats2)


    df = pd.DataFrame()
    df['node_features']=feats2
    df.to_csv(file_dir+"_data-feats-label.csv", index=False)

    return len(nodes)




