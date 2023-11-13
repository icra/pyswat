from src.SWATPollution import SWATPollution
from src.SWATPollutionOptimizer import SWATPollutionOptimizer
from pathlib import Path
from sqlalchemy import create_engine



if __name__ == '__main__':

    contaminant = 'Diuron'
    conca = 'fluvia'

    #cwd = Path('/mnt/c/Users/jsalo/Desktop/ICRA/pyswat')
    cwd = Path(__file__).parent
    txtinout_folder = cwd / 'data' / 'txtinouts' / f"TxtInOut_{conca}"
    channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'
    tmp_path = cwd / 'data' / 'txtinouts' / 'tmp_parallel'
    compound_features_path = cwd / 'data' / 'compound_features.xlsx'


    a = SWATPollutionOptimizer(
        conca, 
        contaminant, 
        txtinout_folder = txtinout_folder,
        channels_geom_path = channels_geom_path,
        tmp_path = tmp_path,
        compound_features_path = compound_features_path,
        lb_factor = 0.2,
        ub_factor = 0.5,
        n_gen = 30,
        n_workers = 12,
    )

    res = a.minimize()
    print(res)




