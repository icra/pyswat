from src.SWATPollution import SWATPollution
from src.pollution_utils import generate_pollution_observations, observacions_from_conca
from pathlib import Path
import sys
import pandas as pd
import tempfile
import os
import subprocess
from pySWATPlus.TxtinoutReader import TxtinoutReader
import json

if __name__ == '__main__':

    """
    args = sys.argv[1:]
    contaminant = args[0]
    conca = args[1]
    """
    contaminant = 'Venlafaxina'
    conca = 'besos'

    cwd = Path(__file__).parent
    txt_in_out_path = cwd / 'data' / 'txtinouts' / f"TxtInOut_{conca}"
    tmp_path = cwd / 'data' / 'txtinouts' / 'sims' / contaminant / conca 

    channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'

    compound_generator_path = cwd.parent / 'traca' / 'traca' / 'inputs compound generator'
    removal_rate_path = compound_generator_path / 'inputs' / 'atenuacions_generacions.xlsx'


    #Crear fitxer auxiliar


    try: 
        tmpdir = tempfile.mkdtemp(dir = tmp_path)
    except FileNotFoundError:
        os.makedirs(tmp_path, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir = tmp_path)

    tmpfile=tempfile.NamedTemporaryFile(dir = tmp_path, delete=False, suffix='.xlsx')
    tmpfile.close()

    #Definir parametres generatio i sanejament ['UV', 'CL', 'SF', 'UF', 'GAC', 'RO', 'AOP', 'O3', 'OTHER', 'Primari', 'C', 'CN', 'CNP', 'cxgx']

    #new_values = [99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 99.99, 0.96]

    #new_values = [40, 50, 0, 0, 50, 60, 50, 60, 20, 5, 15, 30, 15, 0.5]   # hauria de ser prediccio mes alta que observacio

    #['CL', 'UF', 'Primari', 'C', 'CN', 'CNP', 'coef']
    new_values = [90, 100, 30, 30, 100, 100, 100, 100, 50, 20, 40, 100,60, 4e-6]   # hauria de ser prediccio mes baixa que observacio

    new_values = [contaminant] + new_values + [1]   #contaminant + parametres atenuacio + coeficient error industrial

    removal_rate_df = pd.read_excel(removal_rate_path)

    #for each column, if type is integer, convert it to float
    for column in removal_rate_df:
        if removal_rate_df[column].dtype.kind == 'i':
            removal_rate_df[column] = removal_rate_df[column].astype(float)


    removal_rate_df = removal_rate_df.loc[removal_rate_df['contaminant'] == contaminant].copy()
    removal_rate_df.loc[removal_rate_df['contaminant'] == contaminant] = new_values
    removal_rate_df.to_excel(tmpfile.name, index=False)
    
    #Generar nou txtinout
    txtinout_src = TxtinoutReader(txt_in_out_path)
    txtinout_dst = txtinout_src.copy_swat(tmpdir, overwrite=True)

    try:
        os.chdir(compound_generator_path)
        result = subprocess.run(['python3', 'pymain_min.py', conca, txtinout_dst, tmpfile.name, contaminant], capture_output=True, text=True, check=True)
        os.chdir(cwd)

    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e.stderr)


    #Esborrar excel
    os.unlink(tmpfile.name)

    #Executar SWAT
    compound_features = {
        'pollutants.def': ('name', [
            (contaminant, 'solub', 268.87921403643594),
            (contaminant, 'aq_hlife',  5.8163288973662475),
            (contaminant, 'aq_volat', 2.426892003951511e-05),
            (contaminant, 'aq_resus', 0.016572877755983338),
            (contaminant, 'aq_settle', 3.323523663451719),
            (contaminant, 'ben_act_dep', 2.8013198398771575),
            (contaminant, 'ben_bury', 0.015674893121805523),
            (contaminant, 'ben_hlife', 4.8360458988330155),
            ])
    }

    observacions = generate_pollution_observations(contaminant)
    lod_path = cwd / 'data' / 'lod.xlsx'

    lod_path = cwd / 'data' / 'lod.xlsx'
    lod_df = pd.read_excel(lod_path, index_col=0)
    lod = lod_df.loc[contaminant, 'LOD (mg/L)']


    df = observacions_from_conca(channels_geom_path, observacions, conca)
    first_observation = df.year.min()
    year_end = 2022
    year_start = max(first_observation-3, 2000) #3 years warm-up
    warmup = max(1, first_observation - year_start)
    

    pollution_generator = SWATPollution(
        conca, 
        contaminant, 
        txtinout_dst, 
        channels_geom_path,
        tmp_path,
        True,
        compound_features = compound_features,
        show_output = False,
        copy_txtinout = False,
        overwrite_txtinout = False,
        observacions=observacions,
        lod=lod, 
        year_start=year_start,
        year_end=year_end,
        warmup=warmup,
    )
    
    print(f"Finalitzada simulació del contaminant {contaminant} a la conca {conca}. Resultats guardats a :")
    print(pollution_generator.get_txtinout_path())