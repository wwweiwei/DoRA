import os
import random
import numpy as np
import pandas as pd
import torch

CityList = [
    '臺北市', '臺中市', '基隆市', '臺南市', '高雄市', '新北市', '宜蘭縣', '桃園市', '嘉義市', 
    '新竹縣', '苗栗縣', '南投縣', '彰化縣', '新竹市', '雲林縣', '嘉義縣', '屏東縣', '花蓮縣', 
    '臺東縣', '金門縣', '澎湖縣'
]

CatFeatList = [
    'city_nm2', 'town_nm', '交易車位', '小坪數物件', '建物型態', '主要用途', '主要建材', '有無管理組織', 
    '車位類別', '電梯', 'firstfloor_ind', 'shop_ind', 'building_type2', 'col2_ind', 'villname', 
    '都市土地使用分區', '非都市土地使用編定'
]

CatFeatMapping = {'city_nm2':22, 'town_nm':350, '交易車位':2, '小坪數物件':2, '建物型態':5, '主要用途':1622, '主要建材':220, '有無管理組織':2, 
    '車位類別':3, '電梯':3, 'firstfloor_ind':2, 'shop_ind':2, 'building_type2':3, 'col2_ind':2, 'villname':4650, 
    '都市土地使用分區':44, '非都市土地使用編定':16
}

CityNameMapping = {
    '臺北市': 'Taipei',
    '臺中市': 'Taichung',
    '基隆市': 'Keelung',
    '臺南市': 'Tainan',
    '高雄市': 'Kaohsiung',
    '新北市': 'NewTaipei',
    '宜蘭縣': 'Yilan',
    '桃園市': 'Taoyuan',
    '新竹縣': 'HsinchuCounty',
    '新竹市': 'HsinchuCity',
    '嘉義縣': 'ChiayiCounty',
    '嘉義市': 'ChiayiCity',
    '苗栗縣': 'Miaoli',
    '南投縣': 'Nantou',
    '彰化縣': 'Changhua',
    '雲林縣': 'Yunlin',
    '屏東縣': 'Pingtung',
    '花蓮縣': 'Hualien',
    '臺東縣': 'Taitung',
    '金門縣': 'Kinmen',
    '澎湖縣': 'Penghu'
}

BuildingTypeMapping = {
    '大樓': 'Building',
    '公寓': 'Apartment',
    '透天厝': 'House'
}

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def checkAndMakeDir(directory):
    if not os.path.exists(directory):
        print(f'Make directory: {directory}')
        os.makedirs(directory)

def loadData_shots(dataPath, engCityBuilding, mode, shots):
    feature_df = pd.read_csv(f'{dataPath}/{shots}_{engCityBuilding}_{mode}Feat.csv')
    price = pd.read_csv(f'{dataPath}/{shots}_{engCityBuilding}_{mode}Price.csv').values.reshape(-1)

    return feature_df, price

def loadData_unlabeled(dataPath, engCityBuilding, mode):
    feature_df = pd.read_csv(f'{dataPath}/unlabeled_{engCityBuilding}_{mode}Feat.csv')

    return feature_df

def loadData(dataPath, engCityBuilding, mode):
    feature_df = pd.read_csv(f'{dataPath}/{engCityBuilding}_{mode}Feat.csv')
    price = pd.read_csv(f'{dataPath}/{engCityBuilding}_{mode}Price.csv').values.reshape(-1)
    
    return feature_df, price

def column_filter(df):
    return df.loc[:, ['city_nm2','town_nm','交易車位','小坪數物件','建物型態','主要用途','主要建材','有無管理組織','車位類別','電梯','firstfloor_ind','shop_ind','building_type2',
        'col2_ind','villname','都市土地使用分區','非都市土地使用編定','土地移轉總面積(坪)','建物移轉總面積(坪)','建物現況格局-房','建物現況格局-廳','建物現況格局-衛','建物現況格局-隔間',
        '車位移轉總面積(坪)','主建物面積','附屬建物面積','陽台面積','house_age','交易筆棟數_土地','交易筆棟數_建物','交易筆棟數_停車位','building_area_no_park','single_floor_area','far','floor','total_floor',
        'x座標','y座標','larea_utilize_ratio','park_cnt_flat','park_cnt_mach','n_a_10', 'n_a_50', 'n_a_100', 'n_a_250', 'n_a_500', 'n_a_1000', 'n_a_5000', 'n_a_10000','n_c_10', 'n_c_50', 'n_c_100', 'n_c_250', 'n_c_500', 'n_c_1000', 'n_c_5000', 'n_c_10000',
        'area_kilometer','population_density','house_price_index','unemployment_rate','econ_rate','lending_rate','land_tx_count','land_price','steel_id'
    ]]

def initReport(reportWholePath):
    with open(reportWholePath, 'w') as f:
        f.write('MAPE, Hit Rate 10%, Hit Rate 20%, MAE, RMSE\n')

def initPretextReport(reportWholePath):
    with open(reportWholePath, 'w') as f:
        f.write('epoch, precision, recall, f1, support, avg_loss\n')

def initPretextRegReport(reportWholePath):
    with open(reportWholePath, 'w') as f:
        f.write('epoch, mse, avg_loss, avg_mse_loss, avg_scl_loss\n')        

def initPretextClassReport(reportWholePath):
    with open(reportWholePath, 'w') as f:
        f.write('epoch, f1_macro, f1_micro, avg_loss, avg_ce_loss, avg_scl_loss\n')

def addItemToReport(score, reportWholePath):
    if score is not None:
        with open(reportWholePath, 'a') as f:
            f.write(f'{score[0]},{score[1][0]:.2f},{score[1][1]:.2f},{score[1][2]:.2f},{score[1][3]:.2f},{score[1][4]:.2f}\n')