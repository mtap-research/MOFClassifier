from __future__ import print_function, division

import io, os, json, warnings, argparse, requests, zipfile

import numpy as np

from tqdm import tqdm

from pymatgen.core.structure import Structure
from ase.io import read, write
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pymatgen.io.cif"
)

package_directory = os.path.abspath(os.path.dirname(__file__))
models_dir = os.path.join(package_directory, "models")

if os.path.isdir(models_dir) and os.listdir(models_dir):
    pass
else:
    os.makedirs(models_dir, exist_ok=True)
    zip_url = (
        "https://github.com/mtap-research/MOFClassifier/archive/refs/heads/main.zip"
    )
    resp = requests.get(zip_url)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        prefix = "MOFClassifier-main/MOFClassifier/models/"
        for member in z.namelist():
            if not member.startswith(prefix) or member.endswith("/"):
                continue
            rel_path = member[len(prefix):]
            dest_path = os.path.join(models_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with z.open(member) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            print(f"Extracted {rel_path} to models/")

atom_url = "https://raw.githubusercontent.com/mtap-research/MOFClassifier/main/MOFClassifier/atom_init.json"
atom_path = os.path.join(package_directory, "atom_init.json")

if not os.path.exists(atom_path):
    resp = requests.get(atom_url)
    resp.raise_for_status()
    with open(atom_path, "wb") as f:
        f.write(resp.content)
    print("Downloaded atom_init.json")
else:
    pass

def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
            batch_cif_ids

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}
    def state_dict(self):
        return self._embedding
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

def preprocess(root_cif, atom_init_file):
    cif_id = os.path.basename(root_cif).replace(".cif", "")
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
    try:
        crystal = Structure.from_file(root_cif)
    except:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = read(root_cif)
            write(root_cif, atoms)
    atom_fea = np.vstack([ari.get_atom_fea(crystal[j].specie.number)
                          for j in range(len(crystal))])
    atom_fea = torch.Tensor(atom_fea)
    all_nbrs = crystal.get_all_neighbors(8, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < 12:
            warnings.warn('{} not find enough neighbors to build graph. '
                          'If it happens frequently, consider increase '
                          'radius.'.format(cif_id))
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (12 - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) + [8 + 1.] * (12 - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:12])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:12])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)
    atom_fea = torch.Tensor(atom_fea)
    nbr_fea = torch.Tensor(nbr_fea)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    preload_data = ((atom_fea, nbr_fea, nbr_fea_idx), cif_id)
    return preload_data

class ConvLayer(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.final_fea = 0
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        self.final_fea = crys_fea

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out
    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    def norm(self, tensor):
        return (tensor - self.mean) / self.std
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def predict(root_cif, atom_init_file=os.path.join(package_directory, "atom_init.json"), model_dir = os.path.join(package_directory, "models")):
    use_cuda = torch.cuda.is_available()
    models_100 = []
    for i in tqdm(range(1, 101)):
        collate_fn = collate_pool
        dataset_test = []
        dataset_test.append(preprocess(root_cif=root_cif, atom_init_file=atom_init_file))
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=True,
                                num_workers=1, collate_fn=collate_fn,
                                pin_memory=use_cuda)
        modelpath = os.path.join(model_dir, 'checkpoint_bag_'+str(i)+'.pth.tar')
        if os.path.isfile(modelpath):
            model_checkpoint = torch.load(modelpath, weights_only=False,
                                          map_location=lambda storage, loc: storage)
            model_args = argparse.Namespace(**model_checkpoint['args'])
        else:
            print("=> no model params found at '{}'".format(modelpath))
        structures, _ = dataset_test[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True)
        if use_cuda:
            model.cuda()
        normalizer = Normalizer(torch.zeros(3))
        if os.path.isfile(modelpath):
            checkpoint = torch.load(modelpath, weights_only=False,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            print("=> no model found at '{}'".format(modelpath))
        test_preds = []
        test_cif_ids = []
        model.eval()
        for _, (input, batch_cif_ids) in enumerate(test_loader):
            with torch.no_grad():
                if use_cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                Variable(input[1].cuda(non_blocking=True)),
                                input[2].cuda(non_blocking=True),
                                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3])
            output = model(*input_var)
            test_pred = torch.exp(output.data.cpu())
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
            test_cif_ids += batch_cif_ids
        models_100.extend(test_preds)
    CLscore = np.mean(models_100)
    return test_cif_ids[0], models_100, CLscore
