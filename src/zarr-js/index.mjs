import v2 from './src/v2.js'
import v3 from './src/v3.js'

const zarr = (request, version, config) => {
  if (!version || version === 'v2') {
    return v2(request)
  } else if (version === 'v3') {
    return v3(request, config)
  } else {
    throw new Error(`version ${version} not recognized`)
  }
}

export default zarr
