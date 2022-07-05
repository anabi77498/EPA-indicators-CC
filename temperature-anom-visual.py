import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import xarray as xr
import seaborn as sns
from matplotlib import style

%matplotlib inline

xr_df = xr.open_dataset('gistemp1200_GHCNv4_ERSSTv5.nc')
xr_df

climate = xr_df.resample(time='Y').mean()
anomaly = climate['tempanomaly']

#Creating an image of the global temperature anomaly for a given year.

#Creating the colorbar
colarbar = {
    'orientation':'horizontal',
    'fraction': 0.45,
    'pad': 0.03,
    'extend':'both'
}

# Creating a plot and subplot (colarbar) of temperature anonmallies over the world map
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())
ax.add_feature(NaturalEarthFeature('cultural', 'admin_0_countries', '10m'),
                       facecolor='none', edgecolor='black')
ax.set_extent([-150, 150, -55, 85])

timeof2021 = -2
date =  pd.to_datetime(anomaly.isel(time=timeof2021)['time'].values)
ax.set_title("Temperature Anomaly in "+ str(date.year) + " [°C]")

anomaly.isel(time=timeof2021).plot.imshow(ax=ax, add_labels=False, add_colorbar=True,
               vmin=-4, vmax=4, cmap='coolwarm',
               cbar_kwargs=colarbar, interpolation='Gaussian')

plt.show()
