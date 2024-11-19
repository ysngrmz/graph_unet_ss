# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
class GenerateDs:
    def __init__(self):
        path=""
    def generateFea(self,l):
        fea = []
        for elem in l:
            fea.append(float(elem))
        return fea
    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def generateEdges(self,_id_,minaa,maxaa,edges):
        for i in range(minaa,maxaa):
            if(i!=_id_):
                edges.append([_id_,i])
        return edges
    def generateAdjForMufoldNwsNconnect(self,ws,ncon,naa):
        edges = []
        for i in range(naa):
            if(i<=ncon):
                edges = self.generateEdges(i,0,i+ncon+1,edges)
            elif(i<naa-ncon):
                edges = self.generateEdges(i,i-ncon,i+ncon+1,edges)
            else:
                edges = self.generateEdges(i,i-ncon,naa,edges)
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(ws, ws), dtype=np.float32)
        return adj
    def generateAdjNConntected(self,n_sample,N_AA_LIST,ncon):
            _id = 0
            edges = []
            for naa in N_AA_LIST:
                first_id = _id
                for i in range(naa):
                    if(i<=ncon):
                        edges = self.generateEdges(_id,_id-i,_id+ncon+1,edges)
                    elif(i<naa-ncon):
                        edges = self.generateEdges(_id,_id-ncon,_id+ncon+1,edges)
                    else:
                        edges = self.generateEdges(_id,_id-ncon,naa + first_id,edges)
                
                    _id = _id + 1
            edges = np.array(edges)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_sample, n_sample), dtype=np.float32)
            return adj
            
    def generateInpNWindow(self,N_AA_LIST,inputs,ws):  
        n_fea = np.shape(inputs[0])[0]
        _id = 0
        inp = []
        for naa in N_AA_LIST:
            first_id = _id
            for i in range(naa):
                _feaLine = []
                if(i<=ws):
                    n_zero = n_fea * (ws - i)
                    startNode = first_id
                    endNode = _id + ws + 1
                    for zero in range(n_zero):
                        _feaLine.append(0.0)
                    for nodeN in range(startNode,endNode):
                        _sample = inputs[nodeN]
                        for _fea in _sample:
                            _feaLine.append(_fea)
                elif(i<naa-ws):
                    startNode = _id - ws
                    endNode = _id + ws + 1
                    for nodeN in range(startNode,endNode):
                        _sample = inputs[nodeN]
                        for _fea in _sample:
                            _feaLine.append(_fea)
                else:
                    startNode = _id - ws
                    endNode = first_id + naa
                    n_zero = n_fea * (ws - ((first_id + naa -1 ) - _id) )
                    for nodeN in range(startNode,endNode):
                        _sample = inputs[nodeN]
                        for _fea in _sample:
                            _feaLine.append(_fea)
                    for zero in range(n_zero):
                        _feaLine.append(0.0)
                _id = _id + 1
                inp.append(_feaLine)
        return np.array(inp)
    def generateAdjFullyConntected(self,N_AA_LIST,n_sample):
        edges = []
        first_elem = 0
        for elem in N_AA_LIST:
            last_elem = first_elem + elem
            for i in range(first_elem,last_elem):
                for j in range(i+1,last_elem):
                    edges.append([i,j])
            first_elem = last_elem
            
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_sample, n_sample), dtype=np.float32)    
        return adj
   
    
    def dsGenerateFeatureAndTarget(self,path,testRat = 0.15,valRat = 0.05):
        fread = open(path,"r")
        number_of_pro = int(fread.readline().replace("\n", ""))
        
        n_test = int(number_of_pro * testRat)
        n_val = int(number_of_pro * valRat)
        n_tr = number_of_pro - (n_test + n_val)
        
        Inputs = []
        Targets = []
        N_AA_LIST = []
        _id = 0
        tr_ind = 0
        val_ind = 0
        for npro in range(number_of_pro):           
            number_of_aa = int(fread.readline().replace("\n", ""))
            N_AA_LIST.append(number_of_aa)
            for naa in range(number_of_aa): 
                _id = _id + 1
                temp_line = fread.readline().replace("\n", "")
                temp_line = temp_line.split()
                clabel = temp_line[0]
                Inputs.append(self.generateFea(temp_line[1:]))
                Targets.append(int(clabel))
            if(npro==n_tr):            
                tr_ind = _id
            elif(npro==n_tr+n_val):
                val_ind = _id
        Targets = self.encode_onehot(Targets)
        return np.array(Inputs), Targets, N_AA_LIST, tr_ind, val_ind
    
    def generateFullyConntectedDsFromLstmFormat(self,path,testRat = 0.15,valRat = 0.05):
        Inputs, Targets, N_AA_LIST, tr_ind, val_ind = self.dsGenerateFeatureAndTarget(path,testRat,valRat)
        n_sample = Targets.shape[0]
        adj = self.generateAdjFullyConntected(N_AA_LIST,n_sample)
        return Inputs, Targets, adj, tr_ind, val_ind
    def generateNConnectedNwindowSizeDsFromLstmFormat(self,path,ncon,ws,testRat = 0.15,valRat = 0.05):
        Inputs, Targets, N_AA_LIST, tr_ind, val_ind = self.dsGenerateFeatureAndTarget(path,testRat,valRat)
        adj = self.generateAdjNConntected(Targets.shape[0],N_AA_LIST,ncon)
        inp = self.generateInpNWindow(N_AA_LIST,Inputs,ws)
        
        return inp, Targets, adj, tr_ind, val_ind
    

        
        
        
        