from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
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
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
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
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, n_t=1, funnel_rate=2, decode_rate=2, poolstyle = 0):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        n_t: int
          Number of output features
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)]) # batch norm
        self.conv_to_fc = nn.Linear(atom_fea_len * self.poolmultiplier(poolstyle), h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.poolstyle = poolstyle
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(max(int(h_fea_len//(funnel_rate**i)), n_t), max(int(h_fea_len//(funnel_rate**(i+1))), n_t))
                                      for i in range(n_h-1)])
            # self.fcs.append(nn.Linear(max(int(h_fea_len // (funnel_rate**(n_h-1))), n_t), max(int(h_fea_len // (funnel_rate**(n_h-1))), n_t)))
            # self.fcs.append(nn.Linear(max(h_fea_len // (funnel_rate**(n_h-1)), n_t), n_t))
            self.fcs.append(nn.Linear(int(max(h_fea_len // (funnel_rate ** (n_h - 1)), n_t)),
                                      int(max(h_fea_len // (funnel_rate ** (n_h - 1)), n_t) * decode_rate)))
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h)])
        if self.classification:
            self.fc_out = nn.Linear(max(h_fea_len // (funnel_rate**(n_h-1)), n_t), 2) # torch.dropout
        else:
            # self.fc_out = nn.Linear(max(h_fea_len // (funnel_rate**(n_h-1)), n_t), n_t)
            # self.fc_out = nn.Linear(n_t, n_t)
            self.fc_out = nn.Linear(int(max(h_fea_len // (funnel_rate ** (n_h - 1)), n_t) * decode_rate), n_t)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
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
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        if self.poolstyle == 0: # mean pool
            summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        elif self.poolstyle == 1: # min pool
            summed_fea = [torch.min(atom_fea[idx_map], dim=0, keepdim=True).values
                          for idx_map in crystal_atom_idx]
        elif self.poolstyle == 2: # max pool
            summed_fea = [torch.max(atom_fea[idx_map], dim=0, keepdim=True).values
                          for idx_map in crystal_atom_idx]
        elif self.poolstyle == 3: # max - min pool
            summed_fea = [torch.max(atom_fea[idx_map], dim=0, keepdim=True).values - torch.min(atom_fea[idx_map], dim=0, keepdim=True).values
                          for idx_map in crystal_atom_idx]
        elif self.poolstyle == 4: # sum pool
            summed_fea = [torch.sum(atom_fea[idx_map], dim=0, keepdim=True)
                          for idx_map in crystal_atom_idx]
        elif self.poolstyle == 5: # mean + stddev
            summed_fea = [torch.cat((torch.mean(atom_fea[idx_map], dim=0, keepdim=True), torch.std(atom_fea[idx_map], dim=0, keepdim=True)), 1)
                          for idx_map in crystal_atom_idx]
        elif self.poolstyle == 6:  # mean + stddev + max
            summed_fea = [torch.cat(
                            (torch.mean(atom_fea[idx_map], dim=0, keepdim=True),
                             torch.std(atom_fea[idx_map], dim=0, keepdim=True),
                             torch.max(atom_fea[idx_map], dim=0, keepdim=True).values
                             ),0
                         )for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
        #return torch.stack(summed_fea, dim=0)

    def poolmultiplier(self, poolstyle):
        if poolstyle in {0, 1, 2, 3, 4}:
            return 1
        elif poolstyle in {5}:
            return 2
        elif poolstyle in {6}:
            return 3
        else:
            raise ValueError("Pooling style does not exist")