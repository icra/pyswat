import os
import pandas as pd
from src.SWATPollution import SWATPollution
from src.pollution_utils import generate_pollution_observations
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
import sys
import numpy as np
from pymoo.util.normalization import denormalize
from src.pollution_utils import observacions_from_conca

#from pySWATPlus.SWATProblem import SWATProblem, minimize_pymoo
from src.SWATProblem import SWATProblem, minimize_pymoo




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
        copy_txtinout = True,
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
    if isinstance(conca, list):
        errors = []        
        
        for conca_aux, path_aux in zip(conca, txtinout_folder):
            
            try:
                pollution_generator = run_swat_pollution(conca_aux, contaminant, path_aux, channels_geom_path, tmp_path, run, compound_features, observacions, lod, year_start, year_end, warmup)
                errors.append(pollution_generator.get_error())
                txt_in_out_paths[conca_aux] = pollution_generator.get_txtinout_path()

            except Exception as e:
                print(f'error in {conca_aux}')
                print(e)
                errors.append(np.nan)
                txt_in_out_paths[conca_aux] = None
                continue
            
        
        np_errors = np.array(errors)
        errors_no_inf = np_errors[~np.isnan(np_errors) & ~np.isinf(np_errors)]
        errors_mean = np.mean(errors_no_inf)
        #errors_mean = random.randint(0, 1000)
        return errors_mean, txt_in_out_paths
    
    else:
        try:
            pollution_generator = run_swat_pollution(conca, contaminant, txtinout_folder, channels_geom_path, tmp_path, run, compound_features, observacions, lod, year_start, year_end, warmup)
            txt_in_out_paths[conca] = pollution_generator.get_txtinout_path()
            error = pollution_generator.get_error()
        except Exception as e:
            print(f'error in {conca}')
            print(e)
            txt_in_out_paths[conca] = None
            error = np.nan


        #error = random.randint(0, 1000)
        return error, txt_in_out_paths

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

        print(f"Best solution found: \nX = {x} \nF = {y}\n")
        

        #print(f'callback: {path}')

class SWATPollutionOptimizer:

    def __init__(self, 
                 conca, 
                 contaminant, 
                 txtinout_folder, 
                 channels_geom_path = os.path.join('data', 'rivs1', 'canals_tot_ci.shp'),
                 tmp_path = os.path.join('data', 'txtinouts', 'tmp'),
                 compound_features_path = os.path.join('data', 'compound_features.xlsx'),
                 lod_path = os.path.join('data', 'lod.xlsx'),
                 lb_factor = 0.2,
                 ub_factor = 0.5,
                 n_gen = 30,
                 n_workers = 12,
                 ):

        compound_features_df = pd.read_excel(compound_features_path).dropna()

        df_contaminant = compound_features_df[compound_features_df['name'] == contaminant]

        #delete column name
        df_contaminant = df_contaminant[['solub', 'aq_hlife', 'aq_volat', 'aq_resus', 'aq_settle', 'ben_act_dep', 'ben_bury', 'ben_hlife']]

        #convert to dict
        param_dict = df_contaminant.to_dict('records')[0]

        #definir lower bounds i upperbounds, posar en format {filename: (id_col, [(id, col, lb, up)])}
        params_aux = []
        for key, value in param_dict.items():
            if key == 'solub':
                params_aux.append((contaminant, key, (1-lb_factor)*value, (1+ub_factor)*value))
            elif key == 'aq_hlife' or key == 'ben_hlife':
                params_aux.append((contaminant, key, max((1-3)*value, 0), (1+3)*value))
            else:
                params_aux.append((contaminant, key, max((1-10)*value, 0), (1+10)*value))


        self.param_dict = param_dict

        params_feature_compounds = {
            'pollutants.def': ('name', params_aux)
        }


        lod_df = pd.read_excel(lod_path, index_col=0)
        lod = lod_df.loc[contaminant, 'LOD (mg/L)']


        #PROCESSES
        """
        pool = multiprocessing.Pool(n_workers)
        runner = StarmapParallelization(pool.starmap)
        """
        
        #THREADS
        """
        pool = ThreadPool(n_workers)
        runner = StarmapParallelization(pool.starmap)
        """

        self.n_workers = n_workers
        self.n_gen = n_gen


        observacions = generate_pollution_observations(contaminant)

        df = observacions_from_conca(channels_geom_path, observacions, conca)

        first_observation = df.year.min()
        year_end = 2022
        year_start = max(first_observation-3, 2000) #3 years warm-up
        warmup = max(1, first_observation - year_start)

        #Create SWATProblem with bounds and function to call and get error
        self.swat_problem = SWATProblem(
            params = params_feature_compounds, #{filename: (id_col, [(id, col, lb, up)])}
            function_to_evaluate = wrapper_get_error_pollution_generation,
            param_arg_name = 'compound_features',
            n_workers = n_workers,
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

        #print(f'beginning optimization with {self.n_workers} workers')

        """
        algorithm = GA(
            pop_size=self.n_workers,
            eliminate_duplicates=True,
            )
        """

        
        """
        algorithm = DE(
            pop_size=self.n_workers,
        )
        """
        
        x0 = denormalize(np.random.random(self.swat_problem.n_var), self.swat_problem.xl, self.swat_problem.xu)
        max_evals = self.swat_problem.n_var * 100

        """
        algorithm = CMAES(x0=x0,
                        sigma=0.5,
                        restarts=2,
                        maxfevals=max_evals,
                        restart_from_best=True,
                        bipop=True)   
        """
        algorithm = CMAES(x0=x0,
                        maxfevals=max_evals)   
        
                     
        termination = get_termination("n_eval", max_evals)

        """
        res = pymoo.optimize.minimize(self.swat_problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True,
                       callback=MyCallback(),
                       )
        """

        x, path, error = minimize_pymoo(self.swat_problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True,
                       callback=MyCallback(),
                       )
                

        return path, error

        
        

            

        """
        param_dict = self.param_dict
        return {
            'best_params': dict(zip(param_dict.keys(), res.X)),
            'best_error': res.F[0],
            #'best_path': res.algorithm.callback.data["best"]
        }
        """
        

  

    


        

