# Pollution calibration for TRACA project
This package implements the code for calibrating and analysing the results of SWAT pollution simulation. Has to be used with [input compound generator](https://github.com/icra/traca). Uses [pyswatplus](https://github.com/icra/pySWATPlus) for calibrating, and our modified version of (SWAT+ for modeling point-source pollution)[https://github.com/icra/swatplus]

The main files are:
- optimize_pollutant.py: calibrate pollutant parameters related to transport and attenuation in river
- optimize_generation_attenuation.py: calibrate both paramaters related to transport and attenuation in river, as well as parameters related to generation and filtration in WWTP's.
- plot_cic.ipynb: plot the spatial results of the swat simulation
- plot_temporal.ipynb: plot the results of a given channel
- optimize_edar_effluent: compare predictions in wwtp's effluent with observations (will be used for paretto optimisation in wwtp and rivers)
