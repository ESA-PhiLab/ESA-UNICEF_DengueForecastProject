import geopandas
import contextily as cx
import pandas as pd

def plotShape():
    df = geopandas.read_file('states/BR_UF_2020.shp')
    df_wm = df.to_crs(epsg=3857)
    ax = df_wm.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    ax.axis(False)
    cx.add_basemap(ax)


def plist(l):
    for i, x in enumerate(l):
        print('----- {} {}'.format(i+1,x))
    

def clean(db):
    db['rate_total'] =  (db['cases_total']/db['PopTotal_UF'])*(100000.0)
    db['rate_019']   =  (db['cases0_19']/db['Pop0_19_UF'])*(100000.0)
    db = db.drop(columns=['Pop0_19_Urban_UF', 'Pop0_19_Rural_UF', 'Pop0_19_UF', 'PopTotal_UF', 'cases_total'])
    return db
