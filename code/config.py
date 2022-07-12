GROUPED_VARS = {

    'EXCLUDED':[
        't_fundc_ocup18m',
        't_medioc_ocup18m',
        'PopTotal_Urban_UF',
        'PopTotal_Rural_UF',
        'total_precipitation_d',
        'surface_pressure_d',
        'area_km2',
        'humidity_d',
        'temperature_2m_d',
        'min_temperature_2m_d',
        'CNN_all',
        'CNN_0-19'
    ],

    'CLIMATIC VARIABLES': [
        'dewpoint_temperature_2m_d',
        'max_temperature_2m_d',
        'u_component_of_wind_10m_d',
        'v_component_of_wind_10m_d'
    ],

    'GEO VARIABLES': [
        'NDVI_d',
        'max_elevation_d',
        'mean_elevation_d',
        'min_elevation_d',
        'stdDev_elevation_d',
        'variance_elevation_d',
        'Forest_Cover_Percent',
        'Urban_Cover_Percent'
    ],

    'SOCIO VARIABLES':[
        'Urban_Cover_Percent',
        'ivs',
        'ivs_infraestrutura_urbana',
        'ivs_capital_humano',
        'ivs_renda_e_trabalho',
        't_sem_agua_esgoto',
        't_sem_lixo',
        't_vulner_mais1h',
        't_analf_15m',
        't_cdom_fundin',
        't_p15a24_nada',
        't_vulner',
        't_desocup18m',
        't_p18m_fundin_informal',
        'idhm',
        'idhm_long',
        'idhm_educ',
        'idhm_renda',
        'idhm_educ_sub_esc',
        't_pop18m_fundc',
        'idhm_educ_sub_freq',
        'renda_per_capita',
        'pea10a14',
        'pea15a17',
        'pea18m',
        't_eletrica',
        't_densidadem2',
        'rdpc_def_vulner',
        't_analf_18m',
        't_formal_18m'
    ],

    'AUXILIAR':[
        'Month',
        'cases20_99',
        'cases0_19'
    ],

    'DENGUE':[
        'rate_total',
        'rate_019'
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
    'EPOCHS': 27000,
    'DEVICE':'CPU',
    'LEARNING RATE': 0.001,
    'LOSS':'MultiRMSE',
    'RANDOM SEED': 42,
    'MAX DEPTH': 6,
    'EVALUATION METRIC': 'MultiRMSE',
    'EARLY STOPPING': 300
}

SVM_SETTINGS = {
    'EPOCHS': 20000,
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
    0:'Rondônia',
    1:'Acre',
    2:'Amazonas',
    3:'Roraima',
    4:'Pará',
    5:'Amapá',
    6:'Tocantins',
    7:'Maranhão',
    8:'Piauí',
    9:'Ceará',
    10:'Rio Grande do Norte',
    11:'Paraíba',
    12:'Pernambuco',
    13:'Alagoas',
    14:'Sergipe',
    15:'Bahia',
    16:'Minas Gerais',
    17:'Espírito Santo',
    18:'Rio de Janeiro',
    19:'São Paulo',
    20:'Paraná',
    21:'Santa Catarina',
    22:'Rio Grande do Sul',
    23:'Mato Grosso do Sul',
    24:'Mato Grosso',
    25:'Goiás',
    26:'Distrito Federal'
}
