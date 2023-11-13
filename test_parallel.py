
from src.TxtinoutReader import TxtinoutReader
from pathlib import Path
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, LocalCluster

if __name__ == '__main__':
    contaminant = 'Diuron'
    conca = 'fluvia'

    cwd = Path(__file__)

    dir_conca = cwd.parent / 'data' / 'txtinouts' / 'TxtInOut_fluvia_2'
    reader = TxtinoutReader(dir_conca)

    #set up
    reader.set_beginning_and_end_year(2000, 2022)
    reader.enable_object_in_print_prt('channel_sd', True, False, False, False)
    reader.enable_object_in_print_prt('poll', True, False, False, False)
    reader.disable_csv_print()

    #delete pollutants that are not the current one
    file = reader.register_file('pollutants.def', has_units = False, index = 'name')
    df = file.df
    df = df[df['name'] == contaminant]
    file.df = df.copy()
    file.overwrite_file()

    file = reader.register_file('pollutants_om.exc', has_units = False)
    df = file.df
    df = df[df['pollutants_pth'] == contaminant]
    file.df = df.copy()
    file.overwrite_file()


    #set up
    print('setting up ...')
    
    params  = [
        {'pollutants.def': ('name', [('Diuron', 'solub', 450)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 600)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 750)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 1000)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 1200)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 1400)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 1600)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 1800)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 2000)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 2002)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 2004)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 2006)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 2008)])},
        {'pollutants.def': ('name', [('Diuron', 'solub', 2001)])},
        ]
    

    
    #check if slurm is running
    """
    try:
        print('entered in slurm environment')
        cluster = SLURMCluster(
            cores=7,
            memory='24GB'
        )
        cluster.scale(1)
        client = Client(cluster)

    except FileNotFoundError as e:    
        print('entered in non slurm environment')
        client = Client()
    """
    
    n_workers = min(len(params), 6)
    cluster = LocalCluster()
    client = Client(cluster)
    cluster.scale(6)  

    #run
    print('running ...')
    dir_parallel = cwd.parent / 'data' / 'txtinouts' / 'tmp_parallel'

    import datetime
    beginning = datetime.datetime.now()
    results = reader.run_parallel_swat(params, [], n_workers=n_workers, dir = dir_parallel, client = client)
    end = datetime.datetime.now()
    seconds = (end - beginning).total_seconds()
    print('elapsed time: ', seconds)
    print(results)




