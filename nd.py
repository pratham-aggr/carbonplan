import dask  # noqa
import xarray as xr
import zarr
from ndpyramid.regrid import pyramid_regrid

ds = xr.open_zarr('gs://cmip6/CMIP6/ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp245/r1i1p1f1/Amon/pr/gr/v20190119',decode_cf=False)
ds = ds.rio.write_crs("EPSG:4326")

levels = 2
regridded_pyramid = pyramid_regrid(ds.isel(time=slice(0,10)), levels=levels, method="bilinear",parallel_weights=False)
regridded_pyramid.to_zarr("public/data/regridded.zarr", consolidated=True, mode="w")
