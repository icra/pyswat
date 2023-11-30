

# %%
from src.GenerationAttenuationOptimizer import GenerationAttenuationOptimizer
from pathlib import Path
import sys
import datetime

# %%
cwd = Path.cwd()

# %%
args = sys.argv[1:]
contaminant = args[0]
#conca = args[1:]
conca = args[1]

if len(contaminant) == 0:
    raise ValueError('Especifica valor de contaminant')

if len(conca) == 0:
    raise ValueError('Especifica valor de conques a generar simulacio')
#txtinout_path = Path().resolve() / 'data' / 'txtinouts' / f'Txtinout_{conca}'
#compound_generator_path = Path().resolve().parent / 'traca' / 'traca'/ 'inputs compound generator'

txtinout_path = Path().resolve() / 'scripts' / 'traca_contaminacio' / 'data' / 'txtinouts' / f'TxtInOut_{conca}'
compound_generator_path = Path().resolve() / 'scripts' / 'traca' / 'traca'/ 'inputs compound generator'


removal_rate_path = compound_generator_path / 'inputs' / 'atenuacions_generacions.xlsx'
channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'


recall_points_path = compound_generator_path / 'inputs' / 'recall_points.xlsx'
edar_data_path = compound_generator_path / 'inputs' / 'edar_data.xlsx'


#make folder name day with day and time
current_datetime = datetime.datetime.now()
parent_dir = '/home/jsalo/calibracio_contaminants/scripts/tmp/'
folder_name = contaminant + '_' + conca + '_' + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
folder_path = Path(parent_dir +  folder_name)
folder_path.mkdir(parents=True, exist_ok=True) 


a = GenerationAttenuationOptimizer(
    conca, 
    contaminant, 
    txtinout_folder = txtinout_path,
    removal_rate_path = removal_rate_path,
    compound_generator_path = compound_generator_path,
    recall_points_path = recall_points_path,
    edar_data_path = edar_data_path,
    channels_geom_path = cwd/'scripts'/'traca_contaminacio'/'data'/'rivs1'/'canals_tot_ci.shp',
    tmp_path = folder_path,
    compound_features_path = cwd/'scripts'/'traca_contaminacio'/'data'/'compound_features.xlsx',
    lod_path = cwd/'scripts'/'traca_contaminacio'/'data'/'lod.xlsx',
    n_gen = 1,
    n_workers = 10,
)

# %%
res = a.minimize()

# %%
res

# %%



