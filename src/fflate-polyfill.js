import { unzlibSync } from 'fflate';

// Expose globally so CarbonPlan Maps and Zarr libs can find it
window.unzlibSync = unzlibSync;
