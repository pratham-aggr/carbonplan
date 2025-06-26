/* eslint-env es2020 */

import { gunzipSync, inflateSync } from 'fflate'
import ndarray from 'ndarray'
import product from 'cartesian-product'

const constructors = {
  uint8: Uint8Array,
  int16: Int16Array,
  int32: Int32Array,
  float32: Float32Array,
  float64: Float64Array,
}

/**
 * Convert multidim index to linear index for given shape
 */
function ndToLinearIndex(shape, index) {
  let stride = 1
  let linearIndex = 0
  for (let i = shape.length - 1; i >= 0; i--) {
    linearIndex += index[i] * stride
    stride *= shape[i]
  }
  return linearIndex
}

/**
 * List all chunk keys given array shape, chunk shape, and separator
 */
function listKeys(arrayShape, chunkShape, separator) {
  const zipped = []
  for (let i = 0; i < arrayShape.length; i++) {
    const counts = []
    let iter = 0
    let total = 0
    while (total < arrayShape[i]) {
      counts.push(iter)
      total += chunkShape[i]
      iter++
    }
    zipped.push(counts)
  }
  return product(zipped).map((name) => name.join(separator))
}

/**
 * Parse JSON metadata string
 */
function parseMetadata(json) {
  return JSON.parse(json)
}

/**
 * Parse chunk data buffer to ndarray with dtype and chunk shape,
 * decompressing using codec info, or filling with fillValue if null chunk.
 */
function parseChunk(chunk, dtype, chunkShape, fillValue, codec) {
  if (chunk) {
    chunk = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
    if (codec.name === 'gzip') {
      chunk = gunzipSync(chunk)
    } else if (codec.name === 'blosc' && codec.configuration.cname === 'zlib') {
      chunk = inflateSync(chunk)
    } else if (codec.name !== 'none') {
      throw new Error('compressor ' + codec.name + ' is not supported')
    }
    chunk = new constructors[dtype](chunk.buffer)
  } else {
    const length = chunkShape.reduce((a, b) => a * b, 1)
    chunk = new constructors[dtype](length).fill(fillValue)
  }
  return ndarray(chunk, chunkShape)
}

/**
 * Fetch resource with given request function and options,
 * and return response body as arrayBuffer or text.
 * Throws on error or unsupported types.
 */
async function loader(request, src, options, type) {
  try {
    const response = await request(src, options)
    if (response.status === 200 || response.status === 206) {
      if (type === 'text') {
        return await response.text()
      } else if (type === 'arraybuffer') {
        return await response.arrayBuffer()
      } else {
        throw new Error('unsupported file format')
      }
    } else if ([403, 404].includes(response.status) && type === 'arraybuffer') {
      // Return null if resource not found for chunks
      return null
    } else {
      throw new Error('resource not found')
    }
  } catch (err) {
    if (type === 'arraybuffer' && err.code === 'ENOENT') {
      return null
    }
    throw err
  }
}

/**
 * Open a zarr array, return async function to get chunk by index array
 * Supports sharded or normal chunk storage based on metadata
 */
async function open(request, path, config = {}, metadata = null) {
  const useSuffixRequest = config.useSuffixRequest ?? true
  const indexCache = {}

  // Load metadata if not provided
  if (!metadata) {
    const metaText = await loader(request, `${path}/zarr.json`, {}, 'text')
    metadata = parseMetadata(metaText)
  }

  const isSharded = metadata.codecs[0].name === 'sharding_indexed'
  const arrayShape = metadata.shape
  const chunkShape = isSharded
    ? metadata.codecs[0].configuration.chunk_shape
    : metadata.chunk_grid.configuration.chunk_shape
  const dataType = metadata.data_type
  const separator = metadata.chunk_key_encoding.configuration.separator
  const fillValue = metadata.fill_value
  const codec = isSharded
    ? metadata.codecs[0].configuration.codecs[0]
    : metadata.codecs[0]
  const keys = listKeys(arrayShape, chunkShape, separator)

  // Helper to fetch and parse a chunk from given key array
  async function getChunk(k) {
    if (k.length !== arrayShape.length || k.length !== chunkShape.length) {
      throw new Error('key dimensionality must match array shape and chunk shape')
    }
    const key = k.join(separator)
    if (!keys.includes(key)) throw new Error('storage key ' + key + ' not found')
    const chunkBuffer = await loader(request, `${path}/c/${key}`, {}, 'arraybuffer')
    return parseChunk(chunkBuffer, dataType, chunkShape, fillValue, codec)
  }

  // For sharded chunk storage: fetch shard index and extract chunk by offsets
  async function getShardedChunk(k) {
    if (k.length !== arrayShape.length || k.length !== chunkShape.length) {
      throw new Error('key dimensionality must match array shape and chunk shape')
    }
    const lookup = []
    const chunksPerShard = metadata.chunk_grid.configuration.chunk_shape.map(
      (d, i) => d / chunkShape[i]
    )
    for (let i = 0; i < k.length; i++) {
      lookup.push(Math.floor(k[i] / chunksPerShard[i]))
    }
    const key = lookup.join(separator)
    const src = `${path}/c/${key}`
    const checksumSize = 4
    const indexSize = 16 * chunksPerShard.reduce((a, b) => a * b, 1)

    if (!keys.includes(key)) throw new Error('storage key ' + key + ' not found')

    function getUsingIndex(index) {
      const reducedKey = k.map((d, i) => d % chunksPerShard[i])
      const start = ndToLinearIndex(chunksPerShard, reducedKey)
      if (
        index[start * 2] === 18446744073709551615n &&
        index[start * 2 + 1] === 18446744073709551615n
      ) {
        // Chunk is empty, fill with fillValue
        return parseChunk(null, dataType, chunkShape, fillValue, codec)
      } else {
        const range = `bytes=${index[start * 2]}-${
          Number(index[start * 2]) + Number(index[start * 2 + 1]) - 1
        }`
        return loader(request, src, { headers: { Range: range } }, 'arraybuffer').then(
          (chunkBuffer) => parseChunk(chunkBuffer, dataType, chunkShape, fillValue, codec)
        )
      }
    }

    if (indexCache[key]) {
      return getUsingIndex(indexCache[key])
    } else {
      if (useSuffixRequest) {
        const res = await loader(
          request,
          src,
          { headers: { Range: `bytes=-${indexSize + checksumSize}` } },
          'arraybuffer'
        )
        const index = new BigUint64Array(Buffer.from(res).buffer.slice(0, indexSize))
        indexCache[key] = index
        return getUsingIndex(index)
      } else {
        // Use HEAD request to get Content-Length and then fetch index range
        const headResponse = await request(src, { method: 'HEAD' })
        const contentLength = headResponse.headers.get('Content-Length')
        if (!contentLength) throw new Error('Content-Length header missing')
        const fileSize = Number(contentLength)
        const startRange = fileSize - (indexSize + checksumSize)
        const res = await loader(
          request,
          src,
          { headers: { Range: `bytes=${startRange}-${fileSize - checksumSize - 1}` } },
          'arraybuffer'
        )
        const index = new BigUint64Array(Buffer.from(res).buffer)
        indexCache[key] = index
        return getUsingIndex(index)
      }
    }
  }

  return {
    getChunk: isSharded ? getShardedChunk : getChunk,
    metadata,
  }
}

export default function zarr(request, config = {}) {
  if (!request) {
    if (typeof window !== 'undefined' && window.fetch) {
      request = window.fetch.bind(window)
    } else {
      throw new Error('No request function defined and no fetch available')
    }
  }
  return { open: (path, metadata) => open(request, path, config, metadata) }
}

