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
    contaminant = 'Ciprofloxacina'
    conca = 'besos'

    cwd = Path(__file__).parent
    txt_in_out_path = cwd / 'data' / 'txtinouts' / f"TxtInOut_{conca}"
    tmp_path = cwd / 'data' / 'txtinouts' / 'sims' / contaminant / conca 

    channels_geom_path = cwd / 'data' / 'rivs1' / 'canals_tot_ci.shp'

    compound_generator_path = cwd.parent / 'traca' / 'traca' / 'inputs compound generator'
    removal_rate_path = compound_generator_path / 'inputs' / 'atenuacions_generacions.xlsx'


    #Crear fitxer auxiliar
    tmpdir = tempfile.mkdtemp(dir = tmp_path)
    tmpfile=tempfile.NamedTemporaryFile(dir = tmp_path, delete=False, suffix='.xlsx')
    tmpfile.close()

    #Definir parametres generatio i sanejament ['UV', 'CL', 'SF', 'UF', 'GAC', 'RO', 'AOP', 'O3', 'OTHER', 'Primari', 'C', 'CN', 'CNP', 'cxgx']
    new_values = [10.061382349987, 3.68910396e+00, 0, 9.53763700e-01, 90, 98, 90, 90, 0, 3.37157970e+01, 9.98377496e+01, 9.99835912e+01, 9.99294450e+01, 1.65777664e-01]
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
            (contaminant, 'solub', 5.89175657e+03),
            (contaminant, 'aq_hlife',  7.87844189e-01),
            (contaminant, 'aq_volat', 5.36559494e-05),
            (contaminant, 'aq_resus', 1.39548344e-02),
            (contaminant, 'aq_settle', 3.41342884e-02),
            (contaminant, 'ben_act_dep', 1.98351488e+00),
            (contaminant, 'ben_bury', 9.04084839e-03),
            (contaminant, 'ben_hlife', 4.34015290e-01),
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