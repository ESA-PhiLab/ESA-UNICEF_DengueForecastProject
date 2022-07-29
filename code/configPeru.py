GROUPED_VARS = {

    'EXCLUDED':[
        'PopTotal_Urban_UF',#
        'PopTotal_Rural_UF',#
        'total_precipitation_d',#
        'surface_pressure_d',#
        'area_km2',#
        'humidity_d',#
        'temperature_2m_d',#
        'min_temperature_2m_d',#
    ],

    'CLIMATIC VARIABLES': [
        'dewpoint_temperature_2m_d',#
        'max_temperature_2m_d',#
        'u_component_of_wind_10m_d',#
        'v_component_of_wind_10m_d'#
    ],

    'GEO VARIABLES': [
        'NDVI_d',#
        'max_elevation_d',#
        'mean_elevation_d',#
        'min_elevation_d',#
        'stdDev_elevation_d',#
        'variance_elevation_d',#
        'Forest_Cover_Percent',#
        'Urban_Cover_Percent'#
    ],

    'SOCIO VARIABLES':[
        'Urban_Cover_Percent',#
        'WaterSupply_PublicNetworkInside',
        'WaterSupply_PublicNetworkBuilding',
        'WaterSupply_PoolPublicUse',
        'WaterSupply_Tanker',
        'WaterSupply_Well',
        'WaterSupply_Aqueduct',
        'WaterSupply_RiverLake',
        'WaterSupply_Other',
        'WaterSupply_Closeby',
        'WithElectricity',
        'WithoutElectricity',
        'Hygienic_PublicNetworkInside',
        'Hygienic_PublicNetworkBuilding',
        'Hygienic_SepticTank',
        'Hygienic_Latrine',
        'Hygienic_Well',
        'Hygienic_RiverLake',
        'Hygienic_OpenField',
        'Hygienic_Other',
        'HouseType_Independent',
        'HouseType_Flat',
        'HouseType_Farm',
        'HouseType_HoodAlley',
        'HouseType_Hut',
        'HouseType_Improvised',
        'HouseType_NonHumanHabitation',
        'HouseType_Other',
        'HouseType_Collective'
    ],

    'AUXILIAR':[
        'Month',#
        'cases20_99',#
        'cases0_19'#
    ],

    'DENGUE':[
        'rate_total',#
        'rate_019'#
    ]
}

DATA_REDUCER_SETTINGS = {
    'TYPE': 'PLS', # 'PCA',
    'NUMBER OF COMPONENTS': {
        'CLIMATIC VARIABLES': 4,
        'GEO VARIABLES': 6,
        'SOCIO VARIABLES':10
    }
}

DATA_PROCESSING_SETTINGS = {
    'T LEARNING': 12,
    'T PREDICTION': 1,
    'AUGMENTATION': 3
}

LSTM_SETTINGS = {
    'EPOCHS': 200,
    'LEARNING RATE': 0.0001,
    'BATCH SIZE': 16,
    'OPTIMZER': 'rmsprop', #'adam',
    'LOSS':'mae',
    'EVALUATION METRIC':['mse'],
    'EARLY STOPPING': 12
}

CATBOOST_SETTINGS = {
    'EPOCHS': 1000,
    'DEVICE':'CPU',
    'LEARNING RATE': 0.001,
    'LOSS':'MultiRMSE',
    'RANDOM SEED': 42,
    'MAX DEPTH': 6,
    'EVALUATION METRIC': 'MultiRMSE',
    'EARLY STOPPING': 300
}

SVM_SETTINGS = {
    'EPOCHS': 1000,
    'RANDOM STATE': 2,
    'N ITER': 5,
    'CV': 5,
    'HYPERPARAM':{'estimator__kernel': ['poly','sigmoid','rbf'],
                       'estimator__C': [0.01,0.1,1,10],
                       'estimator__gamma': [0.01,0.1,1],
                       'estimator__epsilon': [0.01,0.1,1]}
}

RF_SETTINGS = {
    'NB_ESTIMATORS': 100,
    'MAX DEPTH': 6
}

ENSAMBLE_SETTINGS = {
    'EPOCHS': 81000,
    'LEARNING RATE': 0.0002,
    'BATCH SIZE': 16,
    'OPTIMZER': 'rmsprop', #'adam',
    'LOSS':'mae',
    'EVALUATION METRIC':['mse'],
    'EARLY STOPPING': 50
}

CNN_SETTINGS = {
    'EPOCHS': 300,
    'LEARNING RATE': 0.0001,
    'BATCH SIZE': 4,
    'OPTIMZER': 'rmsprop', #'adam',
    'LOSS':'mse',
    'EVALUATION METRIC':['mae'],
    'EARLY STOPPING': 100
}

DEP_NAMES = {
    0:'Loreto',
    1:'Madre de Dios',
    2:'Piura',
    3:'San Martin',
    4:'Tumbes',
    5:'Ucayali'
}
