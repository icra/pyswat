import pandas as pd
from sqlalchemy import create_engine
import geopandas as gpd
from shapely.geometry import Point


def generate_pollution_observations(contaminant):
    
    engine = create_engine('postgresql://traca_user:EdificiH2O!@217.61.208.188:5432/traca_1')
    #observacions = pd.read_sql(f"SELECT fecha, estacion, cod_estaci, utm_x, utm_y, variable, unidad_med, valor_alfa, valor FROM estacions_full where variable = '{contaminant}'", engine)
    observacions = pd.read_sql(f"SELECT fecha, estacion, cod_esta_1, utm_x, utm_y, variable, unidad_med, valor_alfa, valor, origen FROM estacions_full_1 where variable = '{contaminant}'", engine)
    observacions = observacions.rename(columns={'cod_esta_1':'cod_estaci'})

    observacions['fecha'] = pd.to_datetime(observacions['fecha'], format='mixed')
    #Convert concentration of observations to mg/L
    def f(unit, value):
        if "µg" in unit:
            return float(value) / 1000
        elif "ng" in unit:
            return float(value) / 1000000
        else:           
            return float(value)

    observacions['valor'] = observacions.apply(lambda x: f(x['unidad_med'], x['valor']), axis=1)
    observacions = observacions.drop_duplicates(subset=['fecha', 'cod_estaci', 'variable', 'valor'], keep='first')

    return observacions



def observacions_from_conca(channels_geom_path, observacions, conca):

    #read channels geometry
    gdf = gpd.read_file(channels_geom_path, driver="ESRI Shapefile")
        
    #x, y in EPSG:25831 coords (UTM31T)
    def coords_to_channel(x, y):
        point = gpd.GeoSeries([Point(x, y)], crs="EPSG:25831").to_crs(gdf.crs).iloc[0]
        df = gdf.copy()
        df['distance'] = gdf.geometry.distance(point)
        observacio = df.sort_values(by='distance').head(1)
        return observacio['Channel'].values[0], observacio['layer'].values[0]
        
    #assign observations to closest channel
    observacions_aux = observacions.copy()
    observacions_aux[['gis_id', 'conca']] = [coords_to_channel(*a) for a in zip(observacions['utm_x'], observacions['utm_y'])]

    #filter observacions by conca
    if isinstance(conca, list):
        observacions_conca = observacions_aux[observacions_aux['conca'].isin(conca)]
    else:
        observacions_conca = observacions_aux[observacions_aux['conca'] == conca]
    
    observacions_conca = observacions_conca.copy()

    #divide fecha into year, month and day
    observacions_conca['year'] = observacions_conca['fecha'].dt.year
    observacions_conca['month'] = observacions_conca['fecha'].dt.month
    observacions_conca['day'] = observacions_conca['fecha'].dt.day

    return observacions_conca


def generate_wwtp_observations(contaminant):
    
    engine = create_engine('postgresql://traca_user:EdificiH2O!@217.61.208.188:5432/traca_1')
    observacions = pd.read_sql(f"""SELECT * FROM edars_effluent where "Substance name" = '{contaminant}'""", engine)
    
    observacions = observacions.rename(columns={'Unit':'unit'})
    observacions = observacions.rename(columns={'Substance name':'substance_name'})
    observacions = observacions.rename(columns={'Value':'value'})


    observacions['fecha'] = pd.to_datetime(observacions['fecha'], format='mixed')
    
    #Convert concentration of observations to mg/L
    def f(unit, value):
        if "µg" in unit:
            return float(value) / 1000
        elif "ng" in unit:
            return float(value) / 1000000
        else:           
            return float(value)

    observacions['valor'] = observacions.apply(lambda x: f(x['unit'], x['value']), axis=1)
    
    return observacions

def observacions_wwtp_from_conca(observacions, wwtp_df, conca):

    wwtp_df = wwtp_df[['codi_eu', 'conca']]
    merged = pd.merge(observacions, wwtp_df, how='left', left_on='cod_eu', right_on='codi_eu')
    return merged[merged['conca'] == conca].copy()

