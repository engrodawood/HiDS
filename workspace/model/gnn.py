from utils import *
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU, Tanh, LeakyReLU, ELU, SELU, GELU
from torch_geometric.nn import GINConv, EdgeConv, DynamicEdgeConv, TransformerConv, global_add_pool, global_mean_pool, global_max_pool,SAGPooling
from tqdm import tqdm
import torch.nn as nn
# %% Graph Neural Network


class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[16, 16, 8], pooling='max', dropout=0.0, conv='GINConv', gembed=False, **kwargs):
        """

        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.
        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Returns
        -------
        None.
        """
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.featd = 32
        self.pooling = {'max': global_max_pool,
                        'mean': global_mean_pool, 'add': global_add_pool}[pooling]
        # if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores
        self.gembed = gembed

        self.fc = Sequential(
             Linear(dim_target, dim_target),      
         )

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ELU())
                # output MLP for node level aggregation
                self.linears.append(
                    Sequential(
                    Linear(out_emb_dim, self.featd), BatchNorm1d(self.featd),ELU(),
                    Linear(self.featd, dim_target), BatchNorm1d(dim_target),
                    )
                )

            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(
                    Sequential(
                    Linear(out_emb_dim, self.featd), BatchNorm1d(self.featd),ELU(),
                    Linear(self.featd, dim_target), BatchNorm1d(dim_target),
                    )
                )
                if conv == 'GINConv':
                    subnet = Sequential(
                        Linear(input_emb_dim, self.featd), BatchNorm1d(self.featd), ELU(),
                        Linear(self.featd, out_emb_dim), BatchNorm1d(out_emb_dim), #ELU(),
                        
                        )
                    self.nns.append(subnet)
                    # Eq. 4.2 eps=100, train_eps=False
                    self.convs.append(GINConv(self.nns[-1], **kwargs))
                elif conv == 'EdgeConv':
                    subnet = Sequential(
                        Linear(2*input_emb_dim, self.featd), BatchNorm1d(self.featd), ELU(),
                        Linear(self.featd,out_emb_dim),BatchNorm1d(out_emb_dim),#ELU(),
                        )
                    self.nns.append(subnet)
                    # DynamicEdgeConv#EdgeConv                aggr='mean'
                    self.convs.append(EdgeConv(self.nns[-1], **kwargs))

                else:
                    raise NotImplementedError

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        # has got one more for initial input
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        pooling = self.pooling
        Z = 0
        import torch.nn.functional as F
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z += z
                dout = F.dropout(pooling(z, batch),
                                 p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x, edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z += z
                    dout = F.dropout(pooling(z, batch),
                                     p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](
                        pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout
        
        # out = self.fc(out)

        return out, Z, x

def decision_function(model, loader, device='cpu', outOnly=False, returnNumpy=True):
    """
    generate prediction score for a given model
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    loader : TYPE Dataset or dataloader
        DESCRIPTION.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    outOnly : TYPE, optional 
        DESCRIPTION. The default is True. Only return the prediction scores.
    returnNumpy : TYPE, optional
        DESCRIPTION. The default is False. Return numpy array or ttensor
    Returns
    -------
    Z : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    ZXn : TYPE
        DESCRIPTION. Empty unless outOnly is False
    """
    if type(loader) is not DataLoader:  # if data is given
        loader = DataLoader(loader)
    if type(device) == type(''):
        device = torch.device(device)
    ZXn = []
    model.to(device)
    model.eval()
    Pn = []
    with torch.no_grad():
        
        for i, data in enumerate(loader):
            data = data.to(device)
            output, zn, xn = model(data)
            Pn.extend(data.pid)
            if returnNumpy:
                zn, xn = toNumpy(zn), toNumpy(xn)
            if not outOnly:
                ZXn.append((zn, xn))
            if i == 0:
                Z = output
                Y = data.y
            else:
                Z = torch.cat((Z, output))
                Y = torch.cat((Y, data.y))
    if returnNumpy:
        Z, Y, Pn = toNumpy(Z), toNumpy(Y), toNumpy(Pn)
    return Z, Y, ZXn, Pn


def EnsembleDecisionScoring(Q, train_dataset, test_dataset, device='cpu', k=None):
    """
    Generate prediction scores from an ensemble of models 
    First scales all prediction scores to the same range and then bags them
    Parameters
    ----------
    Q : TYPE reverse deque or list or tuple
        DESCRIPTION.  containing models or output of train function
    train_dataset : TYPE dataset or dataloader 
        DESCRIPTION.
    test_dataset : TYPE dataset or dataloader 
        DESCRIPTION. shuffle must be false
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    k : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    Z : Numpy array
        DESCRIPTION. Scores
    yy : Numpy array
        DESCRIPTION. Labels
    """

    Z = 0
    if k is None:
        k = len(Q)
    for i, mdl in enumerate(Q):
        if type(mdl) in [tuple, list]:
            mdl = mdl[0]
        zz, yy, _, _ = decision_function(mdl, train_dataset, device=device)
        zz, yy, ZXn, Pn = decision_function(mdl, test_dataset, device=device)
        Z += zz
        if i+1 == k:
            break
    Z = Z/k
    return toNumpy(Z), toNumpy(yy), toNumpy(ZXn), toNumpy(Pn)
# %%


class NetWrapper:
    def __init__(self, model, loss_function=nn.BCEWithLogitsLoss(), device='cuda:0', classification=True):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification

    def _pair_train(self, data_loader, optimizer, clipping=None,train=True):
        """
        Performs pairwise comparisons with ranking loss
        """
        model = self.model.to(self.device)
        if train:
            model.train()
        else:
            model.eval()
        loss_all = 0
        # acc_all = 0
        assert self.classification
        pair_count = []
        for data in data_loader:
            data = data.to(self.device)
            target = data.y
            n_classes = target.shape[-1]
            if not len(pair_count):
                pair_count = np.ones(n_classes)
            if train:
                optimizer.zero_grad()
            ypred, _, _ = model(data)
            if self.loss_fun:
                loss = self.loss_fun(ypred, target.float())
            else:
                loss_pairs = torch.zeros(n_classes)
                for tid in range(n_classes):
                    y = target[:, tid]
                    output = ypred[:, tid]
                    # take only examples with valid labels
                    vidx = ~torch.isnan(y)
                    if len(y)-len(vidx):
                        import pdb
                        pdb.set_trace()
                    output = output[vidx]
                    y = y[vidx]
                    z = toTensor([0])
                    dY = (y.unsqueeze(1)-y)
                    dZ = (output.unsqueeze(1)-output)[dY > 0]
                    dY = dY[dY > 0]
                    if len(dY):
                        closs = torch.mean(torch.max(z, dY-dZ))
                    else:
                        closs = toTensor(0)
                    pair_count[tid] += len(dY)
                    loss_pairs[tid] = closs
                loss = torch.mean(loss_pairs)
            if train:
                loss.backward()
            try:
                num_graphs = data.num_graphs
            except TypeError:
                num_graphs = data.adj.size(0)

            loss_all += loss.item() * num_graphs
            # acc_all += acc.item() * num_graphs

            if clipping is not None:  # Clip gradient before updating weights
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
        # print(pair_count)
        #import pdb;pdb.set_trace()
        return  loss_all #acc_all,

    def regress_graphs(self, loader):
        Z, Y, _, _ = decision_function(self.model, loader, device=self.device)
        n_classes = Z.shape[-1]
        R = np.zeros((n_classes, 2))-2
        for i in range(n_classes):
            try:
                R[i] = np.array(
                    [spearmanr(Y[:, i], Z[:, i])[0], spearmanr(Y[:, i], Z[:, i])[1]])
            except:
                print('Correlation can not be defined')
        Rf = R[R[:,0]!=-2]
        return np.median(Rf, axis=0)[0],  np.median(Rf,axis=0)[1]

    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=100, return_best=True, log_every=0):
        """

        Parameters
        ----------
        train_loader : TYPE
            Training data loader.
        max_epochs : TYPE, optional
            DESCRIPTION. The default is 100.
        optimizer : TYPE, optional
            DESCRIPTION. The default is torch.optim.Adam.
        scheduler : TYPE, optional
            DESCRIPTION. The default is None.
        clipping : TYPE, optional
            DESCRIPTION. The default is None.
        validation_loader : TYPE, optional
            DESCRIPTION. The default is None.
        test_loader : TYPE, optional
            DESCRIPTION. The default is None.
        early_stopping : TYPE, optional
            Patience  parameter. The default is 100.
        return_best : TYPE, optional
            Return the models that give best validation performance. The default is True.
        log_every : TYPE, optional
            DESCRIPTION. The default is 0.
        Returns
        -------
        Q : TYPE: (reversed) deque of tuples (model,val_acc,test_acc)
            DESCRIPTION. contains the last k models together with val and test acc
        train_loss : TYPE
            DESCRIPTION.
        train_acc : TYPE
            DESCRIPTION.
        val_loss : TYPE
            DESCRIPTION.
        val_acc : TYPE
            DESCRIPTION.
        test_loss : TYPE
            DESCRIPTION.
        test_acc : TYPEimport pdb; pdb.set_trace()
        """

        from collections import deque
        Q = deque(maxlen=10)  # queue the last 10 models
        return_best = return_best and validation_loader is not None

        best_val_corr, best_val_pval, test_corr_at_best_val_corr, test_pval_at_best_val_corr = -1, -1,-1,-1

        train_loss,val_loss = None, None
        tr_corr, tr_pval,vl_corr,vl_pval,ts_corr,ts_pval = -1,-1,-1,-1,-1,-1

        time_per_epoch = []
        self.history = []
        patience = early_stopping
        best_epoch = np.inf
        iterator = tqdm(range(1, max_epochs+1))
        for epoch in iterator:
            updated = False

            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()

            train_loss = self._pair_train(
                train_loader, optimizer, clipping)
            
            val_loss = self._pair_train(
                validation_loader,optimizer,clipping,train=False
            )

            if train_loader is not None:
                tr_corr,tr_pval = self.regress_graphs(
                    train_loader)
            end = time.time() - start
            time_per_epoch.append(end)

            if validation_loader is not None:
                vl_corr, vl_pval = self.regress_graphs(
                    validation_loader)
            
            if test_loader is not None:
                ts_corr, ts_pval = self.regress_graphs(
                    test_loader)

            if vl_corr > best_val_corr:
                best_val_corr = vl_corr
                test_corr_at_best_val_corr = ts_corr
                test_pval_at_best_val_corr = ts_pval
                best_val_pval = vl_pval
                best_epoch = epoch
                updated = True
                if return_best:
                    best_model = deepcopy(self.model)
                    best_model.to('cpu')
                    Q.append((best_model, best_val_corr,best_val_pval, test_corr_at_best_val_corr,test_pval_at_best_val_corr))

                if False:
                    from vis import showGraphDataset, getVisData
                    fig = showGraphDataset(getVisData(
                        validation_loader, best_model, self.device, showNodeScore=False))
                    plt.savefig(f'./figout/{epoch}.jpg')
                    plt.close()

            if not return_best:
                Q.append((deepcopy(self.model), vl_corr,ts_corr))

            showresults = False
            if log_every == 0:  # show only if validation results improve
                showresults = updated
            elif (epoch-1) % log_every == 0:
                showresults = True

            if showresults:
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR perf: {tr_corr}, TR pval: {tr_pval}'\
                    f' VL loss: {val_loss} VL perf: {vl_corr} VL pval: {vl_pval} ' \
                    f'TE perf: {ts_corr}, TS pval: {ts_pval}'\
                    f'Best: VL perf: {best_val_corr} pvalue: {best_val_pval}'\
                    f'TE perf (AT BEST VAL) : {test_corr_at_best_val_corr} pval: {test_pval_at_best_val_corr}'
                tqdm.write('\n'+msg)
            self.history.append((train_loss,val_loss))

            if epoch-best_epoch > patience:
                iterator.close()
                break

        if return_best:
            vl_corr = best_val_corr
            vl_pval = best_val_pval
            ts_corr = test_corr_at_best_val_corr
            ts_pval = test_pval_at_best_val_corr

        Q.reverse()
        return Q, tr_corr, tr_pval, vl_corr, vl_pval, ts_corr, ts_pval