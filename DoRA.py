import torch
import torch.nn as nn

import os
import sys
import numpy as np
from tqdm import tqdm
import pickle as pkl
from nltk import flatten

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR
from pytorch_metric_learning import losses

import utils.utils as utils
from utils.model import DNN, Encoder
from utils.eval_metrics import *

BuildingType = sys.argv[1]  ## e.g. 'building'、'apartment'、'house'
NumShot = sys.argv[2]  ## e.g. 1, 5, 10 ..

PreTrainedCol = 'town_nm'
DataDate = '2015-7-1-2021-1-1-2021-4-1-2021-7-1'
ReportPath = f'./report/DoRA_{BuildingType}_{NumShot}'
DataPath = f'./data/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#####################
BATCH_SIZE_PRE = 512
LEARNING_RATE_PRE = 0.005
EPOCHS_PRE = 100
LAMBDA_PRE = 0.7

BATCH_SIZE_DOWN = 512
LEARNING_RATE_DOWN = 0.05
EPOCHS_DOWN = 200
LAMBDA_DOWN = 1.0

RERUN_PRETRAINED = True
#####################

utils.set_random_seed(10)

def process(engBuilding, dataPath, shots):
    unlabeledFeature =  utils.loadData_unlabeled(dataPath+'few_shot', f'{engBuilding}', 'train')
    trainFeature, trainPrice = utils.loadData_shots(dataPath+'few_shot', f'{engBuilding}', 'train', shots)
    valFeature, valPrice = utils.loadData(dataPath+DataDate, f'{engBuilding}', 'val')
    testFeature, testPrice = utils.loadData(dataPath+DataDate, f'{engBuilding}', 'test')

    ## filter specific columns
    unlabeledFeature, trainFeature, valFeature, testFeature = utils.column_filter(unlabeledFeature), utils.column_filter(trainFeature), utils.column_filter(valFeature), utils.column_filter(testFeature)

    unlabeledFeature_train = unlabeledFeature.sample(frac=.8, random_state=10)
    unlabeledFeature_test = unlabeledFeature.loc[~unlabeledFeature.index.isin(unlabeledFeature_train.index)]

    return unlabeledFeature_train, unlabeledFeature_test, trainFeature, trainPrice, valFeature, valPrice, testFeature, testPrice

class EsunDataset(Dataset):
    def __init__(self, z, y):
        super().__init__()
        self.data = z
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class Trainer:
    def __init__(self, model, criterion, optimizer, bs, isClass=False, device=device):
        self.model = model
        self.criterion = criterion
        self.opt = optimizer
        self.device = device
        self.model.to(self.device)
        self.bs = bs
        self.scheduler = None
        self.supconloss = losses.SupConLoss().to(self.device)
        self.pretext_is_classification = isClass
        
    def fit(self, train_loader, val_loader, test_loader, epochs):
        print('Downstream #Parameters to train:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        best_val_mape, best_test_mape, best_test_hit10, best_test_hit20, best_test_mae, best_test_rmse = np.inf, np.inf, 0.0, 0.0, np.inf, np.inf
        self.scheduler = LinearLR(self.opt, total_iters=len(train_loader)*epochs)
        pbar = tqdm(range(epochs), desc='Epoch: ')
        for epoch in pbar:
            train_mape, train_hit10, train_hit20, train_mae, train_rmse, train_loss, train_mse_loss, train_scl_loss = self._train(train_loader)
            val_mape, val_hit10, val_hit20, val_mae, val_rmse = self._validate(val_loader)
            test_mape, test_hit10, test_hit20, test_mae, test_rmse = self._validate(test_loader)

            with open(ReportPath+'/trainReport.csv', 'a') as f:
                f.write(f'{epoch}, {train_mape}, {train_hit10}, {train_hit20}, {train_mae}, {train_rmse} \n')
            with open(ReportPath+'/valReport.csv', 'a') as f:
                f.write(f'{epoch}, {val_mape}, {val_hit10}, {val_hit20}, {val_mae}, {val_rmse} \n')
            with open(ReportPath+'/testReport.csv', 'a') as f:
                f.write(f'{epoch}, {test_mape}, {test_hit10}, {test_hit20}, {test_mae}, {test_rmse} \n')
            with open(ReportPath+'/loss.csv', 'a') as f:
                f.write(f'{epoch}, {train_loss}, {train_mse_loss}, {train_scl_loss} \n')

            if best_val_mape > val_mape:
                best_val_mape, best_test_mape, best_test_hit10, best_test_hit20, best_test_mae, best_test_rmse = val_mape, test_mape, test_hit10, test_hit20, test_mae, test_rmse
        
        print(f'[Best testing] MAPE: {best_test_mape}% | hit10: {best_test_hit10}% | hit20: {best_test_hit20}% | MAE: {test_mae} | RMSE: {test_rmse}')
        return best_val_mape, best_test_mape, best_test_hit10, best_test_hit20, best_test_mae, best_test_rmse
        
    def _train(self, train_dataloader):
        self.model.train()
        tot_loss, tot_mse_loss, tot_scl_loss = 0., 0., 0.
        tot_pred, tot_price = [], []
        for _, item in enumerate(train_dataloader):
            self.opt.zero_grad()
            feat = item[0].clone().detach()
            
            if self.pretext_is_classification == True:
                cat_feature, num_feature, price = feat[:, :16].long().to(self.device), feat[:, 16:].float().to(self.device), item[1].clone().detach().float().to(self.device)
            else:
                cat_feature, num_feature, price = feat[:, :17].long().to(self.device), feat[:, 17:].float().to(self.device), item[1].clone().detach().float().to(self.device)

            pred, pretrained_output = self.model(cat_feature, num_feature)
            for param in self.model.parameters():
                if not param.requires_grad:
                    print(param)
            mse_loss = self.criterion(pred, price.unsqueeze(-1))

            target_col = feat[:, 0] # city_nm2
            scl_pretrained_loss = self.supconloss(pretrained_output, target_col)
            loss = LAMBDA_DOWN * mse_loss + (1-LAMBDA_DOWN) * scl_pretrained_loss 

            loss.backward()
            tot_loss += loss.item()
            tot_mse_loss += mse_loss.item()
            tot_scl_loss += scl_pretrained_loss.item()
            self.opt.step()
            self.scheduler.step()

            pred = pred.cpu().detach().flatten().tolist()
            price = price.cpu().detach().tolist()

            tot_pred.append(pred)
            tot_price.append(price)

        tot_pred = flatten(tot_pred)
        tot_price = flatten(tot_price)

        mape = calMAPE(tot_price, tot_pred)
        hit10 = calHitRate(tot_price, tot_pred, 0.1)
        hit20 = calHitRate(tot_price, tot_pred, 0.2)
        mae = calMAE(tot_price, tot_pred)
        rmse = calRMSE(tot_price, tot_pred)

        avg_loss = round(tot_loss/(len(train_dataloader)*self.bs), 4)
        avg_mse_loss = round(tot_mse_loss/(len(train_dataloader)*self.bs), 4)
        avg_scl_loss = round(tot_scl_loss/(len(train_dataloader)*self.bs), 4)
        print(f'Train Loss: {avg_loss:.2f} | MAPE: {mape}% | hit10: {hit10}% | hit20: {hit20}% | MAE: {mae} | rmse: {rmse}')
        return mape, hit10, hit20, mae, rmse, avg_loss, avg_mse_loss, avg_scl_loss

    def _validate(self, val_dataloader):
        self.model.eval()
        with torch.no_grad():
            tot_pred, tot_price = [], []
            for _, item in enumerate(val_dataloader):
                feat = item[0].clone().detach()

                if self.pretext_is_classification == True:
                    cat_feature, num_feature, price = feat[:, :16].long().to(self.device), feat[:, 16:].float().to(self.device), item[1].clone().detach().float().to(self.device)
                else:
                    cat_feature, num_feature, price = feat[:, :17].long().to(self.device), feat[:, 17:].float().to(self.device), item[1].clone().detach().float().to(self.device)

                pred, _ = self.model(cat_feature, num_feature)

                pred = pred.cpu().detach().flatten().tolist()
                price = price.cpu().detach().tolist()

                tot_pred.append(pred)
                tot_price.append(price)

            tot_pred = flatten(tot_pred)
            tot_price = flatten(tot_price)

            mape = calMAPE(tot_price, tot_pred)
            hit10 = calHitRate(tot_price, tot_pred, 0.1)
            hit20 = calHitRate(tot_price, tot_pred, 0.2)
            mae = calMAE(tot_price, tot_pred)
            rmse = calRMSE(tot_price, tot_pred)
            print(f'MAPE: {mape}% | hit10: {hit10}% | hit20: {hit20}% | MAE: {mae} | rmse: {rmse}')
        return mape, hit10, hit20, mae, rmse

def dnn(train_dataset, val_dataset, test_dataset, feat_dim, bs=64, lr=0.005, epochs=100, pretrained_model=None, isClass=False):
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model = DNN(feat_dim, pretrained_model=pretrained_model)
    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(model=model, criterion=criterion, optimizer=opt, bs=bs, isClass=isClass)
    best_val_mape, best_test_mape, best_test_hit10, best_test_hit20, best_test_mae, best_test_rmse = trainer.fit(train_dataloader, val_dataloader, test_dataloader, epochs)
    return best_val_mape, best_test_mape, best_test_hit10, best_test_hit20, best_test_mae, best_test_rmse

class Pretrained_Trainer:
    def __init__(self, model, criterion, optimizer, bs, isClass=False, device=device):
        self.model = model
        self.criterion = criterion
        self.opt = optimizer
        self.device = device
        self.model.to(self.device)
        self.bs = bs
        self.supconloss = losses.SupConLoss().to(self.device)
        self.pretext_is_classification = isClass
        
    def _train(self, train_dataloader, val_dataloader, epochs):
        scheduler = LinearLR(self.opt, total_iters=len(train_dataloader)*epochs)
        print('Pretrained #Parameters to train:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        best_mse, best_f1 = np.inf, 0.0
        pbar = tqdm(range(epochs), desc='Epoch: ')
        for epoch in pbar:
            self.model.train()
            total_loss, tot_original_loss, tot_scl_loss = 0, 0, 0
            for data in train_dataloader:
                self.opt.zero_grad()
                feat = data[0].clone().detach()
                if self.pretext_is_classification == True:
                    cat_feat, num_feat, label = feat[:, :16].long().to(self.device), feat[:, 16:].float().to(self.device), data[1].to(self.device)
                else:
                    cat_feat, num_feat, label = feat[:, :17].long().to(self.device), feat[:, 17:].float().to(self.device), data[1].type(torch.FloatTensor).to(self.device)

                logits, pretrained_output = self.model(cat_feat, num_feat)

                orginal_loss = self.criterion(logits, label.squeeze(-1))
                target_col = label.squeeze(-1)
                scl_pretrained_loss = self.supconloss(pretrained_output, target_col)
                loss = LAMBDA_PRE * orginal_loss + (1-LAMBDA_PRE) * scl_pretrained_loss 

                loss.backward()
                total_loss += loss.item()
                tot_original_loss += orginal_loss.item()
                tot_scl_loss += scl_pretrained_loss.item()
                self.opt.step()
                scheduler.step()
        
            self.model.eval()
            preds, labels, logits = [], [], []
            for data in val_dataloader:
                feat = data[0].clone().detach()

                if self.pretext_is_classification:
                    cat_feat, num_feat, label = feat[:, :16].long().to(self.device), feat[:, 16:].float().to(self.device), data[1].to(self.device)
                else:
                    cat_feat, num_feat, label = feat[:, :17].long().to(self.device), feat[:, 17:].float().to(self.device), data[1].type(torch.FloatTensor).to(self.device)

                with torch.no_grad():
                    logits, _ = self.model(cat_feat, num_feat)
                    if self.pretext_is_classification: ## classification task
                        pred = torch.argmax(logits, dim=-1).cpu().detach().tolist()
                    else:
                        pred = logits.cpu().detach().flatten().tolist()
                preds.append(pred)
                labels.append(label.cpu().detach().tolist())
            
            preds, labels = flatten(preds), flatten(labels)

            avg_loss = round(total_loss/(len(train_dataloader)*self.bs), 4)
            avg_original_loss = round(tot_original_loss/(len(train_dataloader)*self.bs), 4)
            avg_scl_loss = round(tot_scl_loss/(len(train_dataloader)*self.bs), 4)

            if self.pretext_is_classification: ## classification task
                f1_macro, f1_micro = calF1Macro(labels, preds), calF1Micro(labels, preds)
                pbar.set_description(f"Epoch: {epoch}, F1-Macro: {f1_macro}, F1-Micro: {f1_micro}, Loss: {avg_loss}, MSE Loss: {avg_original_loss}, SCL Loss: {avg_scl_loss}", refresh=False)
                if f1_macro > best_f1:
                    print(f'Epoch {epoch}: F1-Macro: {f1_macro}, save the pretrained model.')
                    pkl.dump(self.model, open(f'{PreTrainedCol}.pkl', 'wb'))

                with open(ReportPath+'/pretext_Class_Report.csv', 'a') as f:
                    f.write(f'{epoch}, {f1_macro}, {f1_micro}, {avg_loss}, {avg_original_loss}, {avg_scl_loss}\n')
            else:
                mse, mape, rmse = calMSE(labels, preds), calMAPE(labels, preds), calRMSE(labels, preds)
                pbar.set_description(f"Epoch: {epoch}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}, Loss: {avg_loss}, MSE Loss: {avg_original_loss}, SCL Loss: {avg_scl_loss}", refresh=False)
                if mse < best_mse:
                    print(f'Epoch {epoch}: MSE = {mse}, save the pretrained model.')
                    pkl.dump(self.model, open(f'{PreTrainedCol}.pkl', 'wb'))
                with open(ReportPath+'/pretext_Reg_Report.csv', 'a') as f:
                    f.write(f'{epoch}, {mse}, {avg_loss}, {avg_original_loss}, {avg_scl_loss}\n')

        return self.model

def pretext_task_class(train_dataset, test_dataset, cat_dim, num_dim, out_dim, bs=64, lr=0.0001, epochs=100):
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model = Encoder(cat_dim, num_dim, out_dim)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Pretrained_Trainer(model=model, criterion=criterion, optimizer=opt, bs=bs, isClass=True)
    model = trainer._train(train_dataloader, test_dataloader, epochs)
    
    return model

def pretext_task_reg(train_dataset, test_dataset, cat_dim, num_dim, out_dim, bs=64, lr=0.0001, epochs=100):
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model = Encoder(cat_dim, num_dim, out_dim)
    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Pretrained_Trainer(model=model, criterion=criterion, optimizer=opt, bs=bs, isClass=False)
    model = trainer._train(train_dataloader, test_dataloader, epochs)
    
    return model

if __name__ == '__main__':
    if not os.path.exists(ReportPath):
        utils.checkAndMakeDir(ReportPath)
        utils.initReport(f'{ReportPath}/trainReport.csv')
        utils.initReport(f'{ReportPath}/valReport.csv')
        utils.initReport(f'{ReportPath}/testReport.csv')
        utils.initReport(f'{ReportPath}/best_score.csv')
        utils.initPretextRegReport(f'{ReportPath}/pretext_Reg_Report.csv')
        utils.initPretextClassReport(f'{ReportPath}/pretext_Class_Report.csv')

    if PreTrainedCol in utils.CatFeatList: ## pretext task is classification task
        isClass = True        
    else:
        isClass = False

    unlabeledFeature_train, unlabeledFeature_test, trainFeature, trainPrice, valFeature, valPrice, testFeature, testPrice = process(engBuilding = BuildingType, dataPath = DataPath, shots = NumShot)
    if RERUN_PRETRAINED == True:
        unlabeledFeature_train_target = unlabeledFeature_train[[PreTrainedCol]]
        unlabeledFeature_test_target = unlabeledFeature_test[[PreTrainedCol]]
        unlabeledFeature_train = unlabeledFeature_train.iloc[:,:].drop([PreTrainedCol], axis=1)
        unlabeledFeature_test = unlabeledFeature_test.iloc[:,:].drop([PreTrainedCol], axis=1)

        pretext_train_dataset = EsunDataset(unlabeledFeature_train.values.astype(np.float32), unlabeledFeature_train_target.values)
        pretext_test_dataset = EsunDataset(unlabeledFeature_test.values.astype(np.float32), unlabeledFeature_test_target.values)
        
        num_cat_list = []
        for key in utils.CatFeatMapping.keys():
            if key != PreTrainedCol:
                num_cat_list.append(utils.CatFeatMapping[key])

        if PreTrainedCol in utils.CatFeatList: ## pretext task is classification task
            pretext_task_model = pretext_task_class(train_dataset=pretext_train_dataset, test_dataset=pretext_test_dataset, cat_dim=num_cat_list, num_dim=unlabeledFeature_train.shape[1]-16, out_dim=utils.CatFeatMapping[PreTrainedCol], bs=BATCH_SIZE_PRE, lr=LEARNING_RATE_PRE , epochs=EPOCHS_PRE)
        else:
            pretext_task_model = pretext_task_reg(train_dataset=pretext_train_dataset, test_dataset=pretext_test_dataset, cat_dim=num_cat_list, num_dim=unlabeledFeature_train.shape[1]-17, out_dim=1, bs=BATCH_SIZE_PRE, lr=LEARNING_RATE_PRE , epochs=EPOCHS_PRE)

    else: ## load pretrained model
        pretext_task_model = pkl.load(open(f'{PreTrainedCol}.pkl', 'rb'))

    trainFeature = trainFeature.drop([PreTrainedCol], axis=1)
    valFeature = valFeature.drop([PreTrainedCol], axis=1)
    testFeature = testFeature.drop([PreTrainedCol], axis=1)

    train_dataset = EsunDataset(trainFeature.values, trainPrice)
    val_dataset = EsunDataset(valFeature.values, valPrice)
    test_dataset = EsunDataset(testFeature.values, testPrice)

    print('====================')

    pretrained_dim = 256
    best_val_mape, best_test_mape, best_test_hit10, best_test_hit20, best_test_mae, best_test_rmse = dnn(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, feat_dim=pretrained_dim, bs=BATCH_SIZE_DOWN, lr=LEARNING_RATE_DOWN, epochs=EPOCHS_DOWN, pretrained_model=pretext_task_model, isClass=isClass)

    with open(ReportPath+'/best_score.csv', 'a') as f:
        f.write(f'{best_test_mape}, {best_test_hit10}, {best_test_hit20}, {best_test_mae}, {best_test_rmse} \n')
