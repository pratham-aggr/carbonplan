import React from 'react';
import { Map, Raster } from '@carbonplan/maps';
import { useColormap } from '@carbonplan/colormaps';
import './App.css';
import zarr from './zarr-js'
import '@carbonplan/maps/mapbox.css'


const TemperatureMap = () => {
  const colormap = useColormap('warm', { count: 256 });
  const dataSource = "http://localhost:8000/data/regridded.zarr";  

  return (
    <Map>
      <Raster
        colormap={colormap}
        clim={[-20,30]}
        source={dataSource}
        variable="pr"
        dimensions={['y', 'x']}
      />
    </Map>
  );
};

export default function App() {
  return (
    <div className="App">
      <h1>Climate Data Visualization</h1>
      <TemperatureMap />
    </div>
  );
}
