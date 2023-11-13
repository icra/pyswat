from src.SWATPollution import SWATPollution
from src.SWATPollutionOptimizer import SWATPollutionOptimizer
from pathlib import Path
from sqlalchemy import create_engine
import sys
import datetime
import shutil

if __name__ == '__main__':


    args = sys.argv[1:]
    contaminant = args[0]
    conques = args[1:]

    if len(contaminant) == 0:
        raise ValueError('Especifica valor de contaminant')

    if len(conques) == 0:
        raise ValueError('Especifica valor de conques a generar simulacio')
    
        
    #cwd = Path('/mnt/c/Users/jsalo/Desktop/ICRA/pyswat')
    cwd = Path(__file__).parent
    txt_in_outs_paths = []
    for conca in conques:
        txt_in_outs_paths.append(cwd / 'data' / 'txtinouts' / f"TxtInOut_{conca}")

    
    #conca = 'fluvia'
    #txtinout_folder = cwd / 'data' / 'txtinouts' / f"TxtInOut_{conca}"
    channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'

    #make folder name day with day and time
    current_datetime = datetime.datetime.now()
    folder_name = contaminant + '_' + '_'.join(conques) + '_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    tmp_path = cwd / 'data' / 'txtinouts' / 'sims' / folder_name
    
    compound_features_path = cwd / 'data' / 'compound_features.xlsx'

    lod_path = cwd / 'data' / 'lod.xlsx'

    a = SWATPollutionOptimizer(
        conques, 
        contaminant, 
        txtinout_folder = txt_in_outs_paths,
        channels_geom_path = channels_geom_path,
        tmp_path = tmp_path,
        compound_features_path = compound_features_path,
        lod_path = lod_path,
        lb_factor = 0.5,
        ub_factor = 0.5,
        n_gen = 20,
        n_workers = 8
    )

    print(f"Començant simulació del contaminant {contaminant} a la conca {conca}:")

    path_sims, error = a.minimize()

    #get paths of best simulations
    paths = path_sims.values()

    #delete all the files in tmp_path except the ones in paths
    for path in tmp_path.iterdir():
        if path not in paths:
            shutil.rmtree(path)

    #rename best simulations temporal name to conca name
    for conca, path_name in path_sims.items():
        path = Path(path_name)
        path.rename(path.parent / conca)

    
    print(f"Millor simulació guardada a {tmp_path}, amb error {error}")





