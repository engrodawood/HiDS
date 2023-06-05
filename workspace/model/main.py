'''
Importing packages
'''

from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.spatial import distance_matrix, Delaunay
import random
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from platscaling import PlattScaling
from utils import *
from gnn import *

if __name__ == '__main__':

    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 300 # Total number of epochs
    split_fold =5 # Stratified cross validation
    scheduler = None
    batch_size = 16
    layers=[16,16,16]
    returnNodeProba = True

    bdir = '/data/PanCancer/Hist-DS'
    drug_sens_file = f'{bdir}/data/BRCA_Drug_sensitivity.csv'
    DS = pd.read_csv(drug_sens_file)
    DS.set_index('Patient ID', inplace=True)
    
    # Converting sensitivity data into z-score
    from sklearn.preprocessing import StandardScaler
    DS.loc[:,DS.columns] = StandardScaler().fit_transform(DS)

    '''
    Graph data
    ''' 
    patch_size = (512,512)
    TAG = 'x'.join(map(str,patch_size))
    Repr = 'ShuffleNet'
    graphs_dir = f'{bdir}/data/Graphs/{TAG}/{Repr}'
    device = 'cuda:0'
    cpu = torch.device('cpu')

    Exid = f'{learning_rate}_{batch_size}_layers_{"_".join(map(str,layers))}'

    graphlist = sorted(glob(os.path.join(graphs_dir, "*.pkl")))
    dataset = []
    GN = []
    for graph in tqdm(graphlist):

        TAG = os.path.split(graph)[-1].split('_')[0][:12]

        # If Drug sensitivity data is missing
        if TAG not in DS.index:
            continue
        G = pickleLoad(graph)
        tstatus = DS.loc[TAG, :].tolist() # patient sensitivity to all drugs
        G.y = toTensor([tstatus], dtype=torch.int8, requires_grad=False)
        G.pid = TAG  # setting up patient id might be used for post processing.
        G.to(cpu)
        dataset.append(G)
    

    print(len(dataset))

    from sklearn.model_selection import KFold
    # Stratified cross validation
    skf = KFold(n_splits=split_fold, shuffle=False)
    TRcorr, TRpval, VLcorr, VLpval, TScorr, TSpval = [], [], [], [], [], []  # Intialise outputs

    Fdata = []
    fold = 1
    for trvi, test in skf.split(np.arange(0, len(dataset))):
        train, valid = train_test_split(
            trvi, test_size=0.20, shuffle=True)  # ,
        train_dataset = [dataset[i] for i in train]
        tr_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)#, batch_sampler=sampler)
        valid_dataset = [dataset[i] for i in valid]
        v_loader = DataLoader(valid_dataset, shuffle=True,batch_size=batch_size)
        test_dataset = [dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=False)

        model = GNN(dim_features=dataset[0].x.shape[1], dim_target=DS.shape[1],
                    layers=layers, dropout=0.1, pooling='mean', conv='EdgeConv', aggr='max')

        net = NetWrapper(model, loss_function=None,#nn.MSELoss(),
                         device=device)  # 
        model = model.to(device=net.device)
        optimizer = optim.Adam(
            model.parameters(),lr=learning_rate, weight_decay=weight_decay)

        best_model, tr_corr, tr_pval,  vl_corr, vl_pval, ts_corr, ts_pval = net.train(
            train_loader=tr_loader,
            max_epochs=epochs,
            optimizer=optimizer,
            scheduler=None,#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            clipping=None,
            validation_loader=v_loader,
            test_loader=tt_loader,
            early_stopping=20,
            return_best=False,
            log_every=5)
        # import pdb; pdb.set_trace()
        Fdata.append((best_model, test_dataset, valid_dataset))
        TRcorr.append(tr_corr)
        TRpval.append(tr_pval)
        VLcorr.append(vl_corr)
        VLpval.append(vl_pval)
        TScorr.append(ts_corr)
        TSpval.append(ts_pval)
        print("\nfold complete ", len(VLcorr),'Train: corr ',tr_corr,' pval: ', tr_pval,
                'Val: corr: ',vl_corr, ' pval: ',vl_pval,
                 ' Test: corr ', ts_corr, ' pval: ',ts_pval
            )
        print('.....'*20,'Saving Convergence Curve','........'*20)

        path_plot_conv = f'{bdir}/outputs/converg_curve/{Exid}/'
        mkdirs(path_plot_conv)
        import matplotlib.pyplot as plt
        ep_loss = np.array(net.history)
        plt.plot(ep_loss)#[:,0]); plt.plot(ep_loss[:,1]); plt.legend(['train','val']);
        plt.savefig(path_plot_conv+str(len(VLcorr))+'.png')
        plt.close()

        print('.....'*20,'Saving Best model Weights','........'*20)
        weights_path = f'{bdir}/outputs/weights/'+Exid+'/'
        mkdirs(weights_path)

        torch.save(best_model[0][0].state_dict(), weights_path+str(fold))
        fold+=1

    # Averaged results of 5 folds
    print("avg Train Corr= ", np.median(TRcorr), "+/-", np.std(TRcorr))
    print("avg Train Pval= ", np.median(TRpval), "+/-", np.std(TRpval))

    print("avg Valid Corr= ", np.median(VLcorr), "+/-", np.std(VLcorr))
    print("avg Valid Pval= ", np.median(VLpval), "+/-", np.std(VLpval))

    print("avg Test Corr= ", np.median(TScorr), "+/-", np.std(TScorr))
    print("avg Test Pval= ", np.median(TSpval), "+/-", np.std(TSpval))

    # Use top 10 models in each fold and re-calculate the averaged results of 5 folds
    import pandas as pd
    # number of splits, AUROC-mean, AUROC-std AUC-PR-mean, AUC-PR-std topics
    RR = np.zeros((split_fold, DS.shape[1], 2))

    node_pred_dir = f'{bdir}/outputs/NodePredictions/{Exid}/'
    mkdirs(node_pred_dir)
    
    res_dir = f'{bdir}/outputs/Results/{Exid}/'
    mkdirs(res_dir)

    foldPredDir = f'{bdir}/outputs/foldPred/{Exid}/'
    mkdirs(foldPredDir)

    for idx in range(len(Fdata)):
        Q, test_dataset, valid_dataset = Fdata[idx]
        zz, yy, zxn, pn = EnsembleDecisionScoring(
            Q, train_dataset, test_dataset, device=net.device, k=10)

        if returnNodeProba:
            for i, G in enumerate(tqdm(test_dataset)):
                G.to(cpu)
                # import pdb; pdb.set_trace()
                G.nodeproba = zxn[i][0]
                G.class_label = DS.columns.to_numpy()
                G.fc = zxn[i][1]
                ofile = f'{node_pred_dir}/{G.pid}.pkl'
                with open(ofile, 'wb') as f:
                    pickle.dump(G, f)

        n_classes = zz.shape[-1]
        R = np.zeros((n_classes, 2))
        for i in range(n_classes):
            try:
                R[i] = np.array(
                    [spearmanr(yy[:, i], zz[:, i])[0], spearmanr(yy[:, i], zz[:, i])[1]])
            except:
                print('Constant value')

        df = pd.DataFrame(R, columns=['SpCorr', 'SpCorr-pval'])
        df.index = DS.columns.tolist()
        RR[idx] = R
        df.to_csv(f'{res_dir}{idx}.csv')

        # saving results of fold prediction
        foldPred = np.hstack((pn[:, np.newaxis], zz, yy))
        columns = ['Patient ID'] + ['P_'+DS.columns.tolist()[i] for i in range(
            DS.shape[1])] + ['T_'+DS.columns.tolist()[i] for i in range(DS.shape[1])]
        foldDf = pd.DataFrame(foldPred, columns=columns)
        foldDf.set_index('Patient ID', inplace=True)
        foldDf.to_csv(f'{foldPredDir}/{idx}.csv')
    RRm = np.median(RR,0)
    RRstd = np.median(RR,0)
    results = pd.DataFrame(np.hstack((RRm, RRstd)))
    results.columns = ['SpCorr-median', 'SpCorr-median-pval', 'SpCorr-std', 'SpCorr-pval-std']
    results.index = DS.columns.tolist()
    results.to_csv(f'{bdir}/outputs/Results/{Exid}.csv')
    print('Results written to csv on disk')
    print(results)
    print(results.mean())
