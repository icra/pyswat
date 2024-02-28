import os
import pandas as pd
from src.IHA import IHA
import numpy as np
from sqlalchemy import create_engine
import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from bokeh.models import HoverTool
import geoviews as gv
gv.extension('bokeh', 'matplotlib')
import plotly.graph_objects as go
from src.pollution_utils import observacions_from_conca
from pySWATPlus.TxtinoutReader import TxtinoutReader
#from src.TxtinoutReader import TxtinoutReader


  

class SWATPollution:

    def __init__(self, 
                 conca, 
                 contaminant, 
                 txtinout_folder, 
                 channels_geom_path = os.path.join('data', 'rivs1', 'canals_tot_ci.shp'),
                 tmp_path = os.path.join('data', 'txtinouts', 'tmp'),
                 run = False,
                 compound_features = {},    #{filename: (id_col, [(id, col, value)])}
                 show_output = True,
                 copy_txtinout = True,
                 overwrite_txtinout = False,
                 observacions = None,
                 lod = None,
                 year_start = 2000,
                 year_end = 2022,
                 warmup = 1

    ):
        
        
        if observacions is None:
            #error
            raise Exception("Observacions not provided")
        
        
        pollutant_name_joined = contaminant

        if run:
            reader = TxtinoutReader(txtinout_folder)

            if copy_txtinout:
                tmp_path = reader.copy_swat(tmp_path, overwrite = overwrite_txtinout)
                reader = TxtinoutReader(tmp_path)

            #set up
            reader.set_beginning_and_end_year(year_start, year_end)
            reader.set_warmup(warmup)
            reader.enable_object_in_print_prt('channel_sd', True, False, False, False)
            reader.enable_object_in_print_prt('poll', True, False, False, False)
            reader.disable_csv_print()


            #delete pollutants that are not the current one
            file = reader.register_file('pollutants.def', has_units = False, index = 'name')
            df = file.df
            df = df[df['name'] == pollutant_name_joined]
            file.df = df.copy()
            file.overwrite_file()

            file = reader.register_file('pollutants_om.exc', has_units = False)
            df = file.df
            df = df[df['pollutants_pth'] == pollutant_name_joined]
            file.df = df.copy()
            file.overwrite_file()
        
            #guardar abans parametres a escriure
            txt_in_out_result = reader.run_swat(compound_features, show_output=show_output)

        else:
            txt_in_out_result = txtinout_folder


        #read pollutants

        reader = TxtinoutReader(txt_in_out_result)
        self.reader = reader
        self.txtinout_path = txt_in_out_result

        pollutant_namde_joined = contaminant
        
        poll = reader.register_file('channel_poll_day.txt', 
                            has_units=False,
                            usecols=['mon', 'day', 'yr', 'gis_id', 'pollutant', 'sol_out_mg', 'sor_out_mg'],
                            filter_by={
                                'pollutant': pollutant_namde_joined
                            })
        
        poll_df = poll.df
        
        poll_df['tot_out_mg'] = poll_df['sol_out_mg'] + poll_df['sor_out_mg'] #mg per day
        poll_df['tot_out_mg_sec'] = poll_df['tot_out_mg'] / (24 * 60 * 60) #mg per second
        
        #read flow
        flow = reader.register_file('channel_sdmorph_day.txt', 
                    has_units=True,
                    usecols=['mon', 'day', 'yr', 'gis_id', 'flo_out'])
        
        self.flow = flow.df
        
        flow_df = flow.df
      
        #merge
        df = poll_df.merge(flow_df, left_on=['mon', 'day', 'yr', 'gis_id'], right_on=['mon', 'day', 'yr', 'gis_id'])
        df['mg_m3'] = df['tot_out_mg_sec'] / df['flo_out']
        df['mg_l'] = df['mg_m3'] / 1000
        df = df[['mon', 'day', 'yr', 'gis_id', 'pollutant', 'mg_l', 'flo_out', 'tot_out_mg']]


        #print(df['pollutant'])
        #convert to string
        #df['pollutant'] = df['pollutant'].astype(str)
        df['pollutant'] = df['pollutant'].str.replace(pollutant_name_joined, contaminant)
        

        gdf = gpd.read_file(channels_geom_path, driver="ESRI Shapefile")

        observacions_conca = observacions_from_conca(channels_geom_path, observacions, conca)

        #merge observacions with model results
        merged_df = observacions_conca.merge(df, left_on=['variable', 'gis_id', 'year', 'month', 'day'], right_on=['pollutant', 'gis_id', 'yr', 'mon', 'day'])

        
        self.a = merged_df

        #calculate error
        merged_df['error'] = merged_df['valor'] - merged_df['mg_l']
        merged_df['error'] = merged_df['error'].abs()
        #merged_df['error'] = merged_df['error'] / merged_df['valor']
        #merged_df['error'] = merged_df['error'].replace([np.inf, -np.inf], np.nan).dropna()
        #merged_df['error'] = merged_df['error'] * 100

        
        aux_df = merged_df[['valor', 'mg_l']]
        aux_df = aux_df.dropna()
        observations = aux_df['valor'].values
        predictions = aux_df['mg_l'].values


        #filter channels by conca and reporject to EPSG:4326
        gdf = gdf[gdf['layer'] == conca]
        gdf = gdf.to_crs(epsg=4326)
        
        #merge results with channel geometry and convert to geodataframe
        df = df.merge(gdf[['Channel', 'geometry', 'layer']], left_on = ['gis_id'], right_on=['Channel'])
        df.rename(columns = {'mon':'month', 'yr':'year'}, inplace=True)
        df["Date"] = pd.to_datetime(df[["year", "month", "day"]])
        gdf_map = gpd.GeoDataFrame(df, geometry='geometry')
        gdf_map['date_str'] = gdf_map['Date'].dt.strftime('%Y-%m-%d')

        self.df = df


        #convert observations to geodataframe
        gdf_observacions = gpd.GeoDataFrame(
            merged_df, geometry=gpd.points_from_xy(merged_df.utm_x, merged_df.utm_y), crs="EPSG:25831"
        )


        gdf_observacions = gdf_observacions.to_crs(epsg=4326)
        gdf_observacions['fecha_str'] = gdf_observacions['fecha'].dt.strftime('%Y-%m-%d')

        df_error = pd.DataFrame({'obs':observations, 'pred':predictions})
        df_error = df_error.replace([np.inf, -np.inf], np.nan).dropna()

        #drop rows where pred is 0 (we would be wasting time trying to optimize this points, if it's 0 is because we are not generating anything)
        df_error = df_error[df_error['pred'] > 0]

        #drop min of observations (LOD)
        #df_error = df_error[df_error['obs'] > df_error['obs'].min()]

        self.lod = lod
        if np.isnan(lod):
            self.lod = None

        #if not non or nan
        if self.lod is not None:
            df_error = df_error[df_error['obs'] > lod]

        try:
            #self.error = mean_squared_error(df_error['obs'].values, df_error['pred'].values, squared=False)
            self.error = -1 * r2_score(df_error['obs'].values, df_error['pred'].values)

            self.rmse = mean_squared_error(df_error['obs'].values, df_error['pred'].values, squared=False)


        except:
            self.error = np.nan
        

        self.river_map = gdf
        self.gdf_map = gdf_map
        self.gdf_observacions = gdf_observacions
        self.contaminant = contaminant
        self.conca = conca

    
    def get_df(self):
        return self.df
    
    def get(self, df_name, has_units = False):
        reader = self.reader
        file = reader.register_file(df_name, has_units=has_units)
        return file.df
    
    def get_observacions(self):
        return self.gdf_observacions

    def get_txtinout_path(self):
        return self.txtinout_path
    
    def simple_map(self, edars):
    
        # Create the 'observations_map' plot    
        hover = HoverTool(tooltips=[("observacio (mg/l)", "@valor"),
                                    ("prediccio (mg/l)", "@mg_l"),
                                    ("error (mg/l)", "@error")
                                ])


        gdf_observacions = self.gdf_observacions[['geometry', 'valor', 'mg_l', 'error']]
        gdf_observacions = gdf_observacions.replace([np.inf, -np.inf], np.nan).dropna()

        #pts = pts.groupby(['id', 'geometry']).agg(prom_revenue=('prix',np.mean)).reset_index() 
        #gdf_observacions = self.gdf_observacions.groupby("geometry").max()[['mg_l', 'valor', 'error']]


        #for each point with multiple observations, take the mean for color
        error_gdf = gdf_observacions.groupby(['geometry']).agg(max_error=('error', np.max)).reset_index()
        error_gdf = error_gdf.replace([np.inf, -np.inf], np.nan).dropna()
        error_gdf = gpd.GeoDataFrame(error_gdf, geometry='geometry')

        #merge on geometry with gdf_observacions
        gdf_observacions = gdf_observacions.merge(error_gdf, left_on = ['geometry'], right_on=['geometry'])
        
        observations_map = gv.Points(
            gdf_observacions
        ).opts(tools=[hover], color='max_error', show_legend=True, cmap = 'Reds', line_color = 'black', size=7, colorbar=True)

        
        hover_2 = HoverTool(tooltips=[("canal", "@Channel"),
                                ])

        # Create the 'river_map' plot
        river_map = gv.Path(
            self.river_map,
            ).opts(tools=[hover_2])

    
        tiles = gv.tile_sources.CartoLight()  

        
        gdf_edars = gpd.GeoDataFrame(
            edars, geometry=gpd.points_from_xy(edars.lon, edars.lat), crs="EPSG:4326"
        )
        edars_map = gv.Points(
            gdf_edars
        ).opts( show_legend=True, marker = 's', color = 'blue', line_color = 'black', size=4, colorbar=True)


        
        return (tiles * river_map * observations_map * edars_map).opts(width=800, height=500)
          
    def scatter_plot(self):
        df = self.gdf_observacions.copy()

        #df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df.rename(columns = {'mg_l':'prediccio (mg/l)', 'valor':'observacio (mg/l)'})

        if self.lod is not None:
            df = df[df['observacio (mg/l)'] > self.lod]

        df = df[df['prediccio (mg/l)'] > 0]

        fig = go.Figure()
        
        #fig = px.scatter(df, x="observacio (mg/l)", y="prediccio (mg/l)", hover_data=["gis_id"], color='origen')
        fig = px.scatter(df, x="observacio (mg/l)", y="prediccio (mg/l)", hover_data=["gis_id"])

        #add trace x=y in gray and dashed
        fig.add_trace(
            go.Scatter(x=df['observacio (mg/l)'], y=df['observacio (mg/l)'],  line=dict(width=1, dash='dot', color='black'), marker=dict(opacity=0))
            )
        
        
        """
        fig.add_trace(
            go.Scatter(x=df['observacio (mg/l)'], y=df['prediccio (mg/l)'], mode='markers',  hover_data=["gis_id"])
        )
        """

        fig.update_traces(marker=dict(size=9,
                                      #color = 'orange',
                                      line=dict(width=1)),
                            selector=dict(mode='markers'))


        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(t=10,l=10,b=10,r=10)
        )
        
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        
        return fig
    
    def r2_mass(self):

        df = self.gdf_observacions.copy()
        df['mg'] = df['mg_l'] * df['flo_out'] * 1000
        df['valor'] = df['valor'] * df['flo_out'] * 1000


        return (-1 * r2_score(df['valor'].values, df['mg'].values))


    
    def scatter_plot_mass(self):
        
        
        df = self.gdf_observacions.copy()
        
        #df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df.rename(columns = {'mg_l':'prediccio (mg/l)', 'valor':'observacio (mg/l)'})

        df['prediccio (mg)'] = df['prediccio (mg/l)'] * df['flo_out'] * 1000
        df['observacio (mg)'] = df['observacio (mg/l)'] * df['flo_out'] * 1000

        
        if self.lod is not None:
            df = df[df['observacio (mg)'] > self.lod]

        df = df[df['prediccio (mg)'] > 0]

        fig = go.Figure()
        
        #fig = px.scatter(df, x="observacio (mg)", y="prediccio (mg)", hover_data=["gis_id"], color='origen')
        fig = px.scatter(df, x="observacio (mg)", y="prediccio (mg)", hover_data=["gis_id"])


        #add trace x=y in gray and dashed
        fig.add_trace(
            go.Scatter(x=df['observacio (mg)'], y=df['observacio (mg)'],  line=dict(width=1, dash='dot', color='black'), marker=dict(opacity=0))
            )
        

        fig.update_traces(marker=dict(size=9,
                                      #color = 'orange',
                                      line=dict(width=1)),
                            selector=dict(mode='markers'))


        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(t=10,l=10,b=10,r=10)
        )
        
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            gridcolor='lightgrey'
        )

        return fig
        
        
        

    def get_error(self):
        return self.error
        #return self.r2_mass()
        
    def visualise_map(self, attribute, name, units, day, month, year):

        #return self.visualise_map_matplotlib('mg_l', 30, day, month, year)
        gdf_map = self.gdf_map[(self.gdf_map['Date'].dt.year == year) & (self.gdf_map['Date'].dt.month == month) & (self.gdf_map['Date'].dt.day == day)]

        if (len(gdf_map) == 0):
            raise Exception("No data for this date")

        gdf_map = gdf_map.copy()

        #apply sqrt scale
        gdf_map['color'] = gdf_map[attribute].apply(lambda x: np.sqrt(x))

        gdf_map = gdf_map.replace([np.inf, -np.inf], np.nan).dropna()
        
        river_map = gdf_map.copy()

        tiles = gv.tile_sources.CartoLight()  # You can choose from different tile sources
        
        atribute_map = gv.Path(
            gdf_map,
            ).opts(color='color', show_legend=True, cmap = 'Reds', colorbar=True, line_width=2.5)

        hover = HoverTool(tooltips=[("cabal", "@flo_out"),
                                    (f"{name} ({units})", f"@{attribute}"),
                                    ])
        river = gv.Path(
            river_map,
            ).opts(tools=[hover], color='black', line_width=3, alpha=0.5)

        return (river * atribute_map * tiles).opts(width=800, height=500)


    def visualise_load(self, day = 1, month = 1, year = 2001):
        return self.visualise_map('tot_out_mg', 'load', 'mg/day', day, month, year)
    
    def visualise_concentration(self, day = 1, month = 1, year = 2001):
        return self.visualise_map('mg_l', 'concentation', 'mg/l', day, month, year)

    def get_df(self):
        return self.gdf_map
    
    def get_a(self):
        return self.a
    
    def plot_channel(self, gis_id):

        predictions = self.get_df()  
        observations = self.gdf_observacions.copy()

        predictions_channel = predictions[predictions['gis_id'] == gis_id][['mg_l', 'Date']]
        observations_channel = observations[observations['gis_id'] == gis_id]

        fig = px.line(predictions_channel, x='Date', y="mg_l", )

        fig.add_trace(
            px.scatter(observations_channel, x='fecha', y="valor").update_traces(marker=dict(color='orange'), mode='markers').data[0]
        )

        fig.update_traces(line=dict(color='black'))

        rmse = self.rmse

        upper_bound = predictions_channel['mg_l'] + 2*rmse
        lower_bound = predictions_channel['mg_l'] - 2*rmse
        lower_bound = lower_bound.apply(lambda x: max(0, x))

        fig.add_traces([go.Scatter(x = predictions_channel.Date, y = upper_bound,
                                mode = 'lines', line_color = 'rgba(0,0,0,0)',
                                showlegend = False),
                        go.Scatter(x = predictions_channel.Date, y = lower_bound,
                                mode = 'lines', line_color = 'rgba(0,0,0,0)',
                                name = '95% confidence interval',
                                fill='tonexty', fillcolor = 'rgba(128, 128, 128, 0.2)')])


        fig.update_traces(marker=dict(size=9,
                                            color = 'orange',
                                            line=dict(width=1)),
                                    selector=dict(mode='markers'))


        fig.update_layout(
                    showlegend=False,
                    plot_bgcolor="white",
                    margin=dict(t=10,l=10,b=10,r=10)
                )
                
        fig.update_layout(
                    plot_bgcolor='white'
                )
        fig.update_xaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey'
                )
        fig.update_yaxes(
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgrey'
                )



        return fig


        


    


        

