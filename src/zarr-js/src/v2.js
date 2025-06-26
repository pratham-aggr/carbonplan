/* eslint-env es2020 */

import { gunzipSync, inflateSync } from 'fflate'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import pool from 'ndarray-scratch'
import product from 'cartesian-product'

const constructors = {
  '<i1': Int8Array,
  '<u1': Uint8Array,
  '|b1': (buffer) => {
    const result = new Uint8Array(buffer)
    return Array.from(result).map((d) => d === 1)
  },
  '|u1': Uint8Array,
  '<i2': Int16Array,
  '<u2': Uint16Array,
  '<i4': Int32Array,
  '<u4': Uint32Array,
  '<f4': Float32Array,
  '<f8': Float64Array,
  '<U': (length, bytes) => (buffer) => {
    // StringArray for unicode, returns array of strings
    const count = buffer.byteLength / (length * bytes)
    const array = []
    for (let s = 0; s < count; s++) {
      const subBuffer = buffer.slice(s * bytes * length, (s + 1) * bytes * length)
      const substring = []
      for (let c = 0; c < length; c++) {
        const parsed = new TextDecoder('utf-8').decode(subBuffer.slice(c * bytes, (c + 1) * bytes))
        // eslint-disable-next-line no-control-regex
        substring.push(parsed.replace(/\x00/g, ''))
      }
      array.push(substring.join(''))
    }
    return array
  },
  '|S': (length, bytes) => (buffer) => {
    // StringArray for bytes, similar to above but 1-byte encoding
    const count = buffer.byteLength / (length * bytes)
    const array = []
    for (let s = 0; s < count; s++) {
      const subBuffer = buffer.slice(s * bytes * length, (s + 1) * bytes * length)
      const substring = []
      for (let c = 0; c < length; c++) {
        const parsed = new TextDecoder('utf-8').decode(subBuffer.slice(c * bytes, (c + 1) * bytes))
        // eslint-disable-next-line no-control-regex
        substring.push(parsed.replace(/\x00/g, ''))
      }
      array.push(substring.join(''))
    }
    return array
  },
}

/**
 * Fetch resource and return text or arraybuffer
 */
async function loader(request, src, type) {
  try {
    const response = await request(src)
    if (response.status === 200) {
      if (type === 'text') return await response.text()
      else if (type === 'arraybuffer') return await response.arrayBuffer()
      else throw new Error('unsupported file format')
    } else if ([403, 404].includes(response.status) && type === 'arraybuffer') {
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
 * List keys for chunks based on metadata
 */
function listKeys(metadata) {
  const zipped = []
  for (let i = 0; i < metadata.shape.length; i++) {
    const counts = []
    let iter = 0
    let total = 0
    while (total < metadata.shape[i]) {
      counts.push(iter)
      total += metadata.chunks[i]
      iter++
    }
    zipped.push(counts)
  }
  return product(zipped).map((name) => name.join('.'))
}

/**
 * Parse metadata JSON
 */
function parseMetadata(json) {
  return JSON.parse(json)
}

/**
 * Parse chunk buffer into ndarray, decompressing if needed
 */
function parseChunk(chunk, metadata) {
  if (chunk) {
    chunk = chunk instanceof Buffer ? chunk : Buffer.from(chunk)
    if (metadata.compressor) {
      if (metadata.compressor.id === 'zlib') {
        chunk = inflateSync(chunk)
      } else if (metadata.compressor.id === 'gzip') {
        chunk = gunzipSync(chunk)
      } else {
        throw new Error(`compressor ${metadata.compressor.id} is not supported`)
      }
    }

    const dtype = metadata.dtype
    if (dtype.startsWith('|S')) {
      const length = parseInt(dtype.split('|S')[1])
      chunk = constructors['|S'](length, 1)(chunk.buffer)
    } else if (dtype.startsWith('<U')) {
      const length = parseInt(dtype.split('<U')[1])
      chunk = constructors['<U'](length, 4)(chunk.buffer)
    } else {
      chunk = new constructors[dtype](chunk.buffer)
    }
  } else {
    const length = metadata.chunks.reduce((a, b) => a * b, 1)
    chunk = Array(length).fill(metadata.fill_value)
  }
  return ndarray(chunk, metadata.chunks)
}

/**
 * Merge chunk objects into a single ndarray
 */
function mergeChunks(chunks, metadata) {
  const dtype = Object.values(chunks[0])[0].dtype
  const shape = metadata.shape.map((d, i) => {
    const c = metadata.chunks[i]
    return Math.floor(d / c) * c + (d % c > 0 ? c : 0)
  })

  let merged
  if (dtype === 'array') {
    merged = ndarray(new Array(shape.reduce((a, b) => a * b, 1)), shape)
  } else {
    merged = pool.zeros(shape, dtype)
  }

  chunks.forEach((chunkObj) => {
    const key = Object.keys(chunkObj)[0]
      .split('.')
      .map((k) => parseInt(k))
    const value = Object.values(chunkObj)[0]
    const lo = key.map((k, i) => k * metadata.chunks[i])
    const hi = metadata.chunks
    let view = merged.lo(...lo).hi(...hi)
    ops.assign(view, value)
  })

  if (metadata.shape.every((d, i) => d === merged.shape[i])) {
    return merged
  } else {
    const result = pool.zeros(metadata.shape, dtype)
    ops.assign(result, merged.hi(...metadata.shape))
    pool.free(merged)
    return result
  }
}

/**
 * Main zarr function: exposes load, open, openGroup, loadGroup as async functions
 */
export default function zarr(request) {
  if (!request) {
    if (typeof window !== 'undefined' && window.fetch) {
      request = window.fetch.bind(window)
    } else {
      throw new Error('no request function defined')
    }
  }

  async function load(path, metadata = null) {
    const onload = async (metadata) => {
      const keys = listKeys(metadata)
      const chunks = await Promise.all(
        keys.map(async (k) => {
          const res = await loader(request, `${path}/${k}`, 'arraybuffer')
          return { [k]: parseChunk(res, metadata) }
        })
      )
      return mergeChunks(chunks, metadata)
    }
    if (metadata) {
      return onload(metadata)
    } else {
      const metaText = await loader(request, `${path}/.zarray`, 'text')
      const metadata = parseMetadata(metaText)
      return onload(metadata)
    }
  }

  async function open(path, metadata = null) {
    const onload = async (metadata) => {
      const keys = listKeys(metadata)
      metadata.keys = keys

      return async function getChunk(k) {
        const key = k.join('.')
        if (!keys.includes(key)) {
          throw new Error(`chunk ${key} not found`)
        }
        const res = await loader(request, `${path}/${key}`, 'arraybuffer')
        return parseChunk(res, metadata)
      }
    }
    if (metadata) {
      return onload(metadata)
    } else {
      const metaText = await loader(request, `${path}/.zarray`, 'text')
      const metadata = parseMetadata(metaText)
      return onload(metadata)
    }
  }

  async function openGroup(path, list = [], metadata = null) {
    const onload = async (metadata) => {
      if (!Object.keys(metadata).includes('zarr_consolidated_format')) {
        throw new Error('metadata is not consolidated')
      }
      const arrays = Object.fromEntries(
        Object.entries(metadata.metadata).filter(([k]) => k.endsWith('.zarray'))
      )
      let keys = Object.keys(arrays).map((k) => k.replace(/\/\.zarray$/, ''))
      if (Array.isArray(list) && list.length > 0) {
        keys = keys.filter((k) => list.includes(k))
      }
      

      const result = {}
      for (const k of keys) {
        result[k] = await open(`${path}/${k}`, arrays[`${k}/.zarray`])
      }
      return [result, metadata]
    }
    if (metadata) {
      return onload(metadata)
    } else {
      const metaText = await loader(request, `${path}/.zmetadata`, 'text')
      const metadata = parseMetadata(metaText)
      return onload(metadata)
    }
  }

  async function loadGroup(path, list = [], metadata = null) {
    const onload = async (metadata) => {
      if (!Object.keys(metadata).includes('zarr_consolidated_format')) {
        throw new Error('metadata is not consolidated')
      }
      const arrays = Object.fromEntries(
        Object.entries(metadata.metadata).filter(([k]) => k.endsWith('.zarray'))
      )
      let keys = Object.keys(arrays).map((k) => k.replace(/\/\.zarray$/, ''))
      if (Array.isArray(list) && list.length > 0) {
        keys = keys.filter((k) => list.includes(k))
      }
      
      const result = {}
      for (const k of keys) {
        result[k] = await load(`${path}/${k}`, arrays[`${k}/.zarray`])
      }
      return [result, metadata]
    }
    if (metadata) {
      return onload(metadata)
    } else {
      const metaText = await loader(request, `${path}/.zmetadata`, 'text')
      const metadata = parseMetadata(metaText)
      return onload(metadata)
    }
  }

  return { load, open, openGroup, loadGroup }
}
