import os
import pandas as pd
from src.pollution_utils import generate_pollution_observations
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.termination import get_termination
import numpy as np
from pymoo.util.normalization import denormalize
from src.pollution_utils import observacions_from_conca
from pathlib import Path
import tempfile
from pySWATPlus.TxtinoutReader import TxtinoutReader
import subprocess
from src.SWATPollution import SWATPollution
#from pySWATPlus.SWATProblem import SWATProblem, minimize_pymoo
from src.SWATProblem import SWATProblem, minimize_pymoo
from pymoo.core.callback import Callback
import random


parametres_a_optimitzar = None
temporary_results_path = None

def wrapper_get_error_pollution_generation(dict_args):
    return get_error_pollution_generation(**dict_args)

def run_swat_pollution(conca_aux, contaminant, path_aux, channels_geom_path, tmp_path, run, compound_features, observacions, lod, year_start, year_end, warmup):
    
    pollution_generator = SWATPollution(
        conca_aux, 
        contaminant, 
        path_aux, 
        channels_geom_path,
        tmp_path,
        run,
        compound_features,
        show_output = False,
        copy_txtinout = False,
        overwrite_txtinout = False,
        observacions = observacions,        
        lod = lod ,
        year_start = year_start,
        year_end = year_end,
        warmup = warmup
    )

    return pollution_generator

def get_error_pollution_generation(
        conca, 
        contaminant, 
        txtinout_folder, 
        channels_geom_path = os.path.join('data', 'rivs1', 'canals_tot_ci.shp'), 
        tmp_path = os.path.join('data', 'txtinouts', 'tmp'), 
        run = True, 
        compound_features = {}, #{filename: (id_col, [(id, col, value)])}
        observacions = None,
        lod = None,
        year_start = 2000,
        year_end = 2022,
        warmup = 1
        ):
    

    txt_in_out_paths = {}

    #if conca is a list, then we have to generate the pollution for each conca:
    
    errors = []
    if isinstance(conca, list):        
        
        for conca_aux, path_aux in zip(conca, txtinout_folder):
            
            try:
                pollution_generator = run_swat_pollution(conca_aux, contaminant, path_aux, channels_geom_path, tmp_path, run, compound_features, observacions, lod, year_start, year_end, warmup)
                errors.append(pollution_generator.get_error())
                txt_in_out_paths[conca_aux] = pollution_generator.get_txtinout_path()
                
                #errors.append(0)
                #txt_in_out_paths[conca_aux] = 'aaaaaa'


            except Exception as e:
                print(f'error in {conca_aux}')
                print(e)
                errors.append(np.nan)
                txt_in_out_paths[conca_aux] = None
                continue
            
        try:
            np_errors = np.array(errors)
            errors_no_inf = np_errors[~np.isnan(np_errors) & ~np.isinf(np_errors)]
            errors_mean = np.mean(errors_no_inf)
        except:
            errors_mean = np.nan

        return errors_mean, txt_in_out_paths
    
    else:
        try:

            pollution_generator = run_swat_pollution(conca, contaminant, txtinout_folder, channels_geom_path, tmp_path, run, compound_features, observacions, lod, year_start, year_end, warmup)
            txt_in_out_paths[conca] = pollution_generator.get_txtinout_path()
            error = pollution_generator.get_error()
            
            #error = random.randint(0, 1000)
            #txt_in_out_paths[conca] = 'aaaaaa'

        except Exception as e:
            print(f'error in {conca}')
            print(e)
            txt_in_out_paths[conca] = None
            error = np.nan


        return error, txt_in_out_paths

def function_prior_swat_execution(
        X, 
        removal_rate_path, 
        contaminant, 
        txtinout_path,
        parent_dir, 
        compound_generator_path, 
        conca, 
        cwd, 
        technologies_order, 
        default_value, 
        X_order):
    
    """
    Update the best solution based on the provided paths and errors.

    Parameters:
    - X (np.ndarray): An array of the values to set in the removal_rate file, in the same order as X_order
    - removal_rate_path (str): Path of the removal_rate file
    - contaminant (str): Name of the contaminant
    - txtinout_path (str): Path of the txtinout file
    - conca (str): Name of the watershed
    - cwd (str): Current working directory
    - technologies_order (list): List of technologies in the order they appear in the removal_rate file, usually ['UV', 'CL', 'SF', 'UF', 'GAC', 'RO', 'AOP', 'O3', 'OTHER', 'Primari', 'C', 'CN', 'CNP']
    - default_value (Dict[str, float]): Default value to set in the removal_rate file for each technology
    - X_order (List[str]): List of technologies in the order they appear in X, for example (['UV', 'CL', 'Primari', 'C', 'CN', 'CNP'])


    Returns:
    Path of the new txtinout file
    """
        
    tmpdir = tempfile.mkdtemp(dir = parent_dir)
    tmpfile=tempfile.NamedTemporaryFile(dir = parent_dir, delete=False, suffix='.xlsx')
    tmpfile.close()

    new_values = []

    """
    1 - llegir ordre de technologies order
        Si valor esta dins de X_order, agafar-lo de X
        Altrament, agafar-lo de default_value
    """
    for technology in technologies_order:
        if technology in X_order:
            new_values.append(X[X_order.index(technology)])
        else:
            new_values.append(default_value[technology])

    new_values = [contaminant] + new_values + [1] #set error industrial equal to 1

    #replace parameters of current pollutant with the ones in X
    removal_rate_df = pd.read_excel(removal_rate_path)

    #for each column, if type is integer, convert it to float
    for column in removal_rate_df:
        if removal_rate_df[column].dtype.kind == 'i':
            removal_rate_df[column] = removal_rate_df[column].astype(float)


    removal_rate_df = removal_rate_df.loc[removal_rate_df['contaminant'] == contaminant].copy()
    removal_rate_df.loc[removal_rate_df['contaminant'] == contaminant] = new_values
    removal_rate_df.to_excel(tmpfile.name, index=False)

    txtinout_src = TxtinoutReader(txtinout_path)
    txtinout_dst = txtinout_src.copy_swat(tmpdir, overwrite=True)

    #run python script
    try:
        os.chdir(compound_generator_path)
        result = subprocess.run(['python3', 'pymain_min.py', conca, txtinout_dst, tmpfile.name, contaminant], capture_output=True, text=True, check=True)
        os.chdir(cwd)

    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e.stderr)

    os.unlink(tmpfile.name)

    return txtinout_dst

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):

        #get argmin
        min_idx = np.argmin(algorithm.pop.get("F"))

        x = algorithm.pop.get("X")[min_idx]
        y = algorithm.pop.get("F")[min_idx]
        #path = algorithm.pop.get("path")[min_idx]

        print(f"Best solution found: \nX = {list(zip(parametres_a_optimitzar, x))} \nF = {y}\n")

        #append results to file
        with open(temporary_results_path, 'a') as f:
            f.write(f"Best solution found: \nX = {list(zip(parametres_a_optimitzar, x))} \nF = {y}\n\n")
        

        #print(f'callback: {path}')

class GenerationAttenuationOptimizer:

    def __init__(self, 
                 conca, 
                 contaminant, 
                 txtinout_folder, 
                 removal_rate_path,
                 compound_generator_path,
                 recall_points_path,
                 edar_data_path,
                 results_file_path,
                 channels_geom_path = os.path.join('data', 'rivs1', 'canals_tot_ci.shp'),
                 tmp_path = os.path.join('data', 'txtinouts', 'tmp'),
                 compound_features_path = os.path.join('data', 'compound_features.xlsx'),
                 lod_path = os.path.join('data', 'lod.xlsx'),
                 n_gen = 30,
                 n_workers = 5,

                 ):
                

        technologies_order = ['UV', 'CL', 'SF', 'UF', 'GAC', 'RO', 'AOP', 'O3', 'OTHER', 'Primari', 'C', 'CN', 'CNP', 'coef']
        global temporary_results_path 
        temporary_results_path = results_file_path
        
        #get wwtp in watersheds
        recall_points_df = pd.read_excel(recall_points_path)
        edars_in_conca = recall_points_df[(recall_points_df['edar_code'].notnull()) & (recall_points_df['conca'] == conca)].edar_code.unique()
        
        #get unique technologies from wwtp in watersheds
        edar_data_df = pd.read_excel(edar_data_path)
        edar_data_df = edar_data_df[edar_data_df['codi_eu'].isin(edars_in_conca)]
        primari = edar_data_df.primari.dropna().unique()
        secundari = edar_data_df.secundari.dropna().unique()
        terciari_splitted = [s.split(',') for s in edar_data_df.terciari.dropna()]
        terciari = set([tecnologia for tren_tractament in terciari_splitted for tecnologia in tren_tractament])
        technologies_used_in_watershed_wwtp = set([*primari, *secundari, *terciari]) #{'CL', 'OTHER', 'P', 'SC', 'SF', 'SN', 'SP', 'UF', 'UV'}

        X_order = []    #these are the values that will be optimized
        for technology in technologies_order:
            if technology == 'Primari':
                if 'P' in technologies_used_in_watershed_wwtp:
                    X_order.append(technology)
            elif technology == 'C':
                if 'SC' in technologies_used_in_watershed_wwtp:
                    X_order.append(technology)
            elif technology == 'CN':
                if 'SN' in technologies_used_in_watershed_wwtp:
                    X_order.append(technology)
            elif technology == 'CNP':
                if 'SP' in technologies_used_in_watershed_wwtp:
                    X_order.append(technology)
            elif technology == 'coef':
                X_order.append(technology)
            else:
                if technology in technologies_used_in_watershed_wwtp:
                    X_order.append(technology)

        #Llegir removal rate
        removal_rate_df = pd.read_excel(removal_rate_path, index_col=0)
        removal_rate_df = removal_rate_df[technologies_order]
        removal_rate_df = removal_rate_df.loc[contaminant]
        default_value = removal_rate_df.to_dict()   #default value that parameters have according to the removal_rate file


        print(X_order)


        removal_rate = removal_rate_df[X_order].values

        #prior_swat_execution_ub = [min(1.3*x, 100) for x in removal_rate]
        #prior_swat_execution_lb = list(removal_rate * 0.7)

        #['CL', 'UF',  'Primari', 'C', 'CN', 'CNP', 'cxgx']
        prior_swat_execution_lb = [50,  0,  5, 15, 30, 15, 4e-6]   # hauria de ser prediccio mes alta que observacio
        prior_swat_execution_ub = [100, 30, 20, 40, 100, 60, 0.4]


        #Llegir compound features
        compound_features_df = pd.read_excel(compound_features_path).dropna()
        df_contaminant = compound_features_df[compound_features_df['name'] == contaminant]

        #delete column name
        df_contaminant = df_contaminant[['solub', 'aq_hlife', 'aq_volat', 'aq_resus', 'aq_settle', 'ben_act_dep', 'ben_bury', 'ben_hlife']]

        #Global variables (only for printing purposes)
        global parametres_a_optimitzar
        parametres_a_optimitzar = X_order + ['solub', 'aq_hlife', 'aq_volat', 'aq_resus', 'aq_settle', 'ben_act_dep', 'ben_bury', 'ben_hlife']

        #convert to dict
        param_dict = df_contaminant.to_dict('records')[0]

        #definir lower bounds i upperbounds, posar en format {filename: (id_col, [(id, col, lb, up)])}
        params_aux = []

        for key, value in param_dict.items():
            if key == 'solub':
                params_aux.append((contaminant, key, 0.5*value, 1.5*value))   #+-50%
            elif key == 'aq_hlife' or key == 'ben_hlife':
                params_aux.append((contaminant, key, 0, 3*value))   #0-3 times the value
            else:
                params_aux.append((contaminant, key, 0, 10*value))  #0-10 times the value


        self.param_dict = param_dict

        params_feature_compounds = {
            'pollutants.def': ('name', params_aux)
        }


        lod_df = pd.read_excel(lod_path, index_col=0)
        lod = lod_df.loc[contaminant, 'LOD (mg/L)']


        self.n_workers = n_workers
        self.n_gen = n_gen


        observacions = generate_pollution_observations(contaminant)
        df = observacions_from_conca(channels_geom_path, observacions, conca)


        first_observation = df.year.min()
        year_end = 2022
        year_start = max(first_observation-3, 2000) #3 years warm-up
        warmup = max(1, first_observation - year_start)

        args_function_to_evaluate_prior = {
            'removal_rate_path': removal_rate_path, 
            'contaminant': contaminant, 
            'txtinout_path': txtinout_folder, 
            'parent_dir': tmp_path,
            'compound_generator_path': compound_generator_path, 
            'conca': conca,
            'cwd': Path.cwd(),
            'technologies_order': technologies_order, 
            'default_value': default_value, 
            'X_order': X_order
        }


        #Create SWATProblem with bounds and function to call and get error
        self.swat_problem = SWATProblem(
            params = params_feature_compounds, #{filename: (id_col, [(id, col, lb, up)])}
            function_to_evaluate = wrapper_get_error_pollution_generation,
            param_arg_name = 'compound_features',
            n_workers = n_workers,
            ub_prior = prior_swat_execution_ub,
            lb_prior =  prior_swat_execution_lb,
            function_to_evaluate_prior = function_prior_swat_execution,
            args_function_to_evaluate_prior = args_function_to_evaluate_prior,
            param_arg_name_to_modificate_by_prior_function = 'txtinout_folder',
            conca = conca,     #beginning of kwargs
            contaminant = contaminant, #kwargs
            txtinout_folder = txtinout_folder,  #kwargs
            channels_geom_path = channels_geom_path, #kwargs
            tmp_path = tmp_path, #kwargs
            run = True, #kwargs
            compound_features = {},   #kwargs
            observacions = observacions.copy(),   #kwargs
            lod = lod, #kwargs
            year_start = year_start,  #kwargs
            year_end = year_end,    #kwargs
            warmup = warmup          #kwargs
        )


    def minimize(self):

        
        x0 = denormalize(np.random.random(self.swat_problem.n_var), self.swat_problem.xl, self.swat_problem.xu)
        max_evals = self.swat_problem.n_var * 100
        
        algorithm = CMAES(x0=x0,
                        maxfevals=max_evals)   
                     
        termination = get_termination("n_eval", max_evals)

        x, path, error = minimize_pymoo(self.swat_problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True,
                       callback=MyCallback(),
                       )
                

        return path, error        

  

    