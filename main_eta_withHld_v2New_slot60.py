from utils_cat21.utils_data_cat21 import *
from utils_cat21.utils_eta_modeling_v2 import *

import pandas as pd
import numpy as np
import os
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import pytorch_lightning as pl
from PIL import Image
from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse



# --------------------------------------------  0. config  -----------------------------------------------------
dir_image_holdingModeling = dir_trajImage_1
dir_image_etaModeling = dir_trajImage_2
dir_etaLabel_features = dir_csvETA_datesCombine_v2

# ------------------------------------------   1. custom dataset  ---------------------------------------------
# custom dataset for eta
class CustomETADataset(Dataset):
    def __init__(self, slot, sta, dir_etaLabel_features):
        self.etaFile = dir_etaLabel_features + "Nov/" + "slot_" + str(slot) + "_sta_" + str(sta) + ".csv"
        self.features = ['image',
                        'recat_label', 'is_weekday', 'is_peakhour', 
                        'drct', 'sknt', 'gust', 'vsby', 'skyc1', 'skyl1', 
                        'runwayChange', 'numARR_on_02L20R', 'numARR_on_02R20L', 
                        ]
        self.hld_features = ['arr_runway', 'gapTBC', 'deltaV_avg', 'deltaV_lead', 'hld_lead']
        self.date_ac = ['date', 'ac']
        self.df_eta = pd.read_csv(self.etaFile, index_col=0)
        
    def __len__(self):
        print("######################################")
        print("######################################")
        print(f"data size: {len(self.df_eta)}")
        return len(self.df_eta)
          
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self.df_eta.loc[index, 'image']
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image)
        # image = image.transpose(2, 0, 1)
        image = image.astype(np.float32)
        image /= 255
        image = transforms.functional.to_tensor(image)
        tabularFeatures = self.df_eta[self.features[1:]]#no image here
        tabular = tabularFeatures.loc[index, self.features[1:]].tolist()
        tabular = torch.FloatTensor(tabular)

        hld_tabFeatures = self.df_eta[self.hld_features]
        hld_tab = hld_tabFeatures.loc[index, self.hld_features].tolist()
        hld_tab = torch.FloatTensor(hld_tab)

        # date = self.df_eta.loc[index, self.date_ac[0]].tolist()
        # date = torch.FloatTensor(date)
        # ac = self.df_eta.loc[index, self.date_ac[1]].tolist()
        # ac = torch.FloatTensor(ac)

        #index in the original data
        # ind = self.df_eta.loc[index, 'ind'].tolist()
        # ind = torch.FloatTensor(ind)

        eta = torch.tensor(float(self.df_eta.loc[index, 'eta']))

        return image, tabular, hld_tab, eta


# -----------------------------     2. define network for holding and eta modeling --------------------------------

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)),
        nn.ReLU(),
        nn.BatchNorm2d(output_size),
        nn.MaxPool2d((2, 2)))
    return block

class LitClassifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, num_workers: int = 20, batch_size: int = 32, slot: int = 60, sta: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.results = {
                        'y': [],
                        'y_hat': [],
                        # 'ind': [],
                        # 'ac': []
                        }
        

        self.slot = slot
        self.sta = sta

        # holding
        self.image_effnet = models.efficientnet_b0(num_classes=32)
        self.hld_regression = nn.Sequential(
            nn.Linear(in_features=5, out_features=16),
            # nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
        )
        self.hold_classification = nn.Sequential(
            nn.Linear(in_features=48, out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )

        # eta
        self.image_mobilenet = models.mobilenet_v2(pretrained=True)
        self.image_mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=64),#v2
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )     
        self.tab_regression = nn.Sequential(
            nn.Linear(in_features=12, out_features=16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
        )

        self.eta_regression = nn.Sequential(
            nn.Linear(in_features=80, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32),
        )

        self.final_regression = nn.Sequential(
        nn.Linear(in_features=40, out_features=1),
        # nn.BatchNorm1d(16),
        # nn.LeakyReLU(),
        # nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, img, tab, hld_tab):
        img_hld = self.image_effnet(img)
        tab_hld = self.hld_regression(hld_tab)
        hld = torch.cat((img_hld, tab_hld), dim=1)
        hld = self.hold_classification(hld)

        img = self.image_mobilenet(img)
        tab = self.tab_regression(tab)
        old_eta = torch.cat((img, tab), dim=1)
        old_eta = self.eta_regression(old_eta)
        x = torch.cat((hld, old_eta), dim=1)
        x = self.final_regression(x)

        return x
    
           
    def setup(self,stage):
        image_data =  CustomETADataset(self.slot, self.sta, dir_etaLabel_features)
        train_size = int(0.70 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size - val_size))

        generator = torch.Generator().manual_seed(123)
        self.train_set, self.val_set, self.test_set = random_split(image_data, (train_size, val_size, test_size), generator=generator)
        print("\n ---------------model setup finished---------------------")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size = self.batch_size)
    
    def configure_optimizers(self):
        print("\n ---------------configure optimizers---------------------")
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def training_step(self, batch, batch_idx):
        image, tabular, tab_hld, y = batch
        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(image, tabular, tab_hld))
        y_pred = y_pred.double()
        loss = criterion(y_pred, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, tabular, tab_hld, y= batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(image, tabular, tab_hld))
        y_pred = y_pred.double()
        val_loss = criterion(y_pred, y)
        self.validation_step_outputs.append(val_loss)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        image, tabular, tab_hld, y = batch
        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(image, tabular, tab_hld))
        y_pred = y_pred.double()

        test_loss = criterion(y_pred, y)
        self.test_step_outputs.append(test_loss)
        self.results['y'] = self.results['y'] + torch.flatten(y).double().tolist()
        self.results['y_hat'] = self.results['y_hat'] + y_pred.tolist()
        # self.results['ind'] = self.results['ind'] + torch.flatten(ind).int().tolist()
        # self.results['ac'] = self.results['ac'] + torch.flatten(ac).int().tolist()

        print(f"in test step, y=  {y}, y_hat = {y_pred}")
        return {"test_loss": test_loss}
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        df_ret = pd.DataFrame(self.results)
        df_ret['err'] = df_ret['y_hat'] - df_ret['y']
        df_ret['absErr_ratio'] = abs(df_ret['err'])/df_ret['y']
        df_ret['mape'] = df_ret['absErr_ratio'].mean()
        df_ret['mae'] = df_ret['err'].abs().mean()
        df_ret['mse'] = mse(df_ret['y'], df_ret['y_hat'], squared=True)
        df_ret['rmse'] = mse(df_ret['y'], df_ret['y_hat'], squared=False)
        file = dir_eta_pred_withHold+ f"slot_{str(self.slot)}_sta_{str(self.sta)}_withHld_v2New.csv"
        df_ret.to_csv(file)
        print("\n   --------------- test data saved to csv file --------------")
        return avg_loss.tolist()


if __name__ == "__main__":

    # csvETA_combine(dates = range(221025, 221031), prefix="Oct")
    csvETA_combine_v2(dates = range(221101, 221131), prefix="Nov")

    slots = [60] 
    stas = [10, 15, 20, 25, 30]

    for slot in slots:
        for sta in stas:
            trainer = pl.Trainer(accelerator="gpu", max_epochs = 1000)
            model = LitClassifier(slot=slot, sta=sta)
            trainer.fit(model)
            print(f"train finished for slot {slot}, sta {sta}")
            trainer.save_checkpoint(f"./models/model_slot_{str(slot)}_sta_{sta}_withHld_v2New.ckpt")
            # new_model = model.load_from_checkpoint(checkpoint_path=f"./models/model_slot_{str(slot)}_sta_{sta}.ckpt")
            # loss = trainer.test(new_model)
            loss = trainer.test(model)
            print(f"test finished for slot {slot}, sta {sta}")
            print("----------------------------\n\n")







    


        






