// flow-overlay: an image with motion arrows on top (nonoha view SDK 1).
//
// Plain ES module without imports: the engine serves it as a static file and
// the browser imports it directly. Helpers are top-level functions.
//
// Inputs
//   image    u8 [H x W] gray, [H x W x 3] (BGR from OpenCV by default) or [H x W x 4]
//   vectors  f32 [R x C x 2] (dx, dy) per cell in px/frame, e.g. MotionGrid's 6 x 8 grid,
//            or a dense flow field (OpticalFlowFarneback), averaged down to about
//            one arrow per `spacing` px.
// Only the last batch row of each value is shown.

const BUCKETS = 12
const HEAD_COS = Math.cos(0.45)
const HEAD_SIN = Math.sin(0.45)

/** "#rgb[a]", "#rrggbb[aa]", "rgb[a](...)" -> [r, g, b, a] (0-255), or null. */
function parseColor(css) {
  const s = String(css).trim().toLowerCase()
  if (s[0] === "#") {
    const h = s.slice(1)
    if (!/^[0-9a-f]+$/.test(h)) return null
    const short = h.length === 3 || h.length === 4
    if (!short && h.length !== 6 && h.length !== 8) return null
    const at = i => (short ? parseInt(h[i] + h[i], 16) : parseInt(h.slice(2 * i, 2 * i + 2), 16))
    return [at(0), at(1), at(2), h.length === 4 || h.length === 8 ? at(3) : 255]
  }
  const m = /^rgba?\(([^)]*)\)$/.exec(s)
  if (!m) return null
  const p = m[1].split(/[\s,/]+/).filter(Boolean)
  if (p.length < 3) return null
  const ch = (t, scale) => (t.endsWith("%") ? (parseFloat(t) / 100) * scale : parseFloat(t))
  const c = [ch(p[0], 255), ch(p[1], 255), ch(p[2], 255), p.length > 3 ? ch(p[3], 1) * 255 : 255]
  return c.every(isFinite) ? c.map(v => Math.max(0, Math.min(255, Math.round(v)))) : null
}

let probe = null
/** Any CSS color -> [r, g, b, a], via the browser when parseColor does not know it. */
function resolveColor(css) {
  const parsed = parseColor(css)
  if (parsed) return parsed
  if (!probe) probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true })
  probe.clearRect(0, 0, 1, 1)
  probe.fillStyle = "#808080"
  probe.fillStyle = css
  probe.fillRect(0, 0, 1, 1)
  const d = probe.getImageData(0, 0, 1, 1).data
  return [d[0], d[1], d[2], d[3]]
}

function rgba(a, b, t, alpha) {
  const mix = i => Math.round(a[i] + (b[i] - a[i]) * t)
  return `rgba(${mix(0)},${mix(1)},${mix(2)},${alpha.toFixed(3)})`
}

/** Largest rect with aspect srcW:srcH centered in dstW x dstH -> out {x, y, w, h}. */
function fitRect(srcW, srcH, dstW, dstH, out) {
  const scale = srcW > 0 && srcH > 0 ? Math.min(dstW / srcW, dstH / srcH) : 0
  out.w = srcW * scale
  out.h = srcH * scale
  out.x = (dstW - out.w) / 2
  out.y = (dstH - out.h) / 2
  return out
}

/** Grid to draw for a rows x cols field shown at w x h px: the field itself, or fewer cells ~spacing px apart. */
function targetGrid(rows, cols, w, h, spacing, out) {
  const maxRows = Math.max(1, Math.round(h / spacing))
  const maxCols = Math.max(1, Math.round(w / spacing))
  out.rows = Math.min(rows, maxRows)
  out.cols = Math.min(cols, maxCols)
  return out
}

/**
 * Average a rows x cols x 2 field (starting at `offset`) over an
 * outRows x outCols grid into `out` (outRows * outCols * 2). Every source cell
 * lands in exactly one output cell; NaN cells are skipped.
 */
function poolGrid(src, offset, rows, cols, outRows, outCols, out) {
  for (let gr = 0; gr < outRows; ++gr) {
    const r0 = Math.floor((gr * rows) / outRows)
    const r1 = Math.floor(((gr + 1) * rows) / outRows)
    for (let gc = 0; gc < outCols; ++gc) {
      const c0 = Math.floor((gc * cols) / outCols)
      const c1 = Math.floor(((gc + 1) * cols) / outCols)
      let sx = 0
      let sy = 0
      let n = 0
      for (let r = r0; r < r1; ++r) {
        let i = offset + (r * cols + c0) * 2
        for (let c = c0; c < c1; ++c, i += 2) {
          const dx = src[i]
          const dy = src[i + 1]
          if (dx === dx && dy === dy) {
            sx += dx
            sy += dy
            ++n
          }
        }
      }
      const o = (gr * outCols + gc) * 2
      out[o] = n > 0 ? sx / n : 0
      out[o + 1] = n > 0 ? sy / n : 0
    }
  }
}

/** u8 pixels (1, 3 or 4 channels) -> RGB of an RGBA buffer whose alpha is already 255. */
function convertToRgba(src, offset, count, channels, bgr, dst) {
  if (channels === 1) {
    for (let i = 0, j = 0; i < count; ++i, j += 4) {
      const v = src[offset + i]
      dst[j] = v
      dst[j + 1] = v
      dst[j + 2] = v
    }
    return
  }
  const r = bgr ? 2 : 0
  const b = bgr ? 0 : 2
  for (let i = 0, s = offset, j = 0; i < count; ++i, s += channels, j += 4) {
    dst[j] = src[s + r]
    dst[j + 1] = src[s + 1]
    dst[j + 2] = src[s + b]
  }
}

function readNumber(settings, key, fallback, min, max) {
  const v = typeof settings[key] === "number" ? settings[key] : parseFloat(settings[key])
  return isFinite(v) ? Math.min(max, Math.max(min, v)) : fallback
}

export default {
  sdk: 1,
  id: "flow-overlay",
  title: "Flow overlay",
  description: "Image with one motion arrow per grid cell; arrow length and color follow the flow magnitude.",
  inputs: [
    {
      name: "image",
      accepts: ["u8[?x?]", "u8[?x?x3]", "u8[?x?x4]"],
      optional: true,
      description: "Frame to draw under the arrows (gray, or 3/4 channels in colorOrder)",
    },
    {
      name: "vectors",
      accepts: ["f32[?x?x2]"],
      description: "(dx, dy) per cell in px/frame: a coarse grid, or a dense flow field (averaged down)",
    },
  ],
  settings: {
    scale: { type: "number", default: 6, description: "Arrow length in screen px per px/frame of motion" },
    colorOrder: { type: "string", default: "bgr", enum: ["bgr", "rgb"], description: "Channel order of 3/4-channel images" },
    colorMax: { type: "number", default: 0, description: "Magnitude (px/frame) with the hottest color; 0 = auto" },
    dim: { type: "number", default: 0.25, description: "Darken (or lighten) the image toward the background, 0-1" },
    spacing: { type: "number", default: 24, description: "Arrow spacing in px when a dense field is averaged down" },
  },
  size: { width: 400, height: 260 },
  mount(el, ctx) {
    const canvas = document.createElement("canvas")
    canvas.style.display = "block"
    el.appendChild(canvas)
    const g = canvas.getContext("2d")
    const frame = document.createElement("canvas") // the image at its own resolution
    const fg = frame.getContext("2d")

    let width = 0
    let height = 0
    let dpr = 1
    let theme = ctx.theme
    let scale = 6
    let bgr = true
    let colorMax = 0
    let dim = 0.25
    let spacing = 24

    let pixels = null // ImageData of the last image
    let imageW = 0
    let imageH = 0
    let imageChannels = 1
    let imageDirty = false

    let field = new Float32Array(0) // last vectors row, rows x cols x 2
    let fieldRows = 0
    let fieldCols = 0
    let pooled = new Float32Array(0)
    let poolDirty = true
    let peak = 1 // auto colorMax: decaying max magnitude

    const box = { x: 0, y: 0, w: 0, h: 0 }
    const grid = { rows: 0, cols: 0 }
    let counts = new Int32Array(BUCKETS + 1)
    let order = new Int32Array(0)
    let bucketOf = new Uint8Array(0)

    let ramp = []
    let halo = ""
    let labelFont = ""

    function applyTheme() {
      const low = resolveColor(theme.accent)
      const high = resolveColor(theme.series[1] || theme.text)
      ramp = []
      for (let b = 0; b < BUCKETS; ++b) {
        const t = b / (BUCKETS - 1)
        ramp.push(rgba(low, high, t, 0.6 + 0.4 * t))
      }
      const bg = resolveColor(theme.background)
      halo = rgba(bg, bg, 0, 0.6)
      labelFont = `400 11px ${theme.fontFamily}`
    }

    function readSettings(s) {
      scale = readNumber(s, "scale", 6, 0, 1000)
      colorMax = readNumber(s, "colorMax", 0, 0, 1e6)
      dim = readNumber(s, "dim", 0.25, 0, 1)
      spacing = readNumber(s, "spacing", 24, 6, 400)
      const nextBgr = s.colorOrder !== "rgb"
      if (nextBgr !== bgr && pixels && imageChannels >= 3) {
        // Swap R and B in place: the difference between the two orders.
        const d = pixels.data
        for (let j = 0; j < d.length; j += 4) {
          const t = d[j]
          d[j] = d[j + 2]
          d[j + 2] = t
        }
        imageDirty = true
      }
      bgr = nextBgr
      poolDirty = true
    }

    function setSize(w, h) {
      width = Math.max(0, w)
      height = Math.max(0, h)
      dpr = window.devicePixelRatio || 1
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      canvas.width = Math.max(1, Math.round(width * dpr))
      canvas.height = Math.max(1, Math.round(height * dpr))
      poolDirty = true
    }

    function pool() {
      poolDirty = false
      targetGrid(fieldRows, fieldCols, box.w, box.h, spacing, grid)
      const cells = grid.rows * grid.cols
      if (pooled.length !== cells * 2) {
        pooled = new Float32Array(cells * 2)
        order = new Int32Array(cells)
        bucketOf = new Uint8Array(cells)
      }
      if (grid.rows === fieldRows && grid.cols === fieldCols) pooled.set(field.subarray(0, cells * 2))
      else poolGrid(field, 0, fieldRows, fieldCols, grid.rows, grid.cols, pooled)
      let max = 0
      for (let i = 0; i < cells; ++i) {
        const m = Math.hypot(pooled[2 * i], pooled[2 * i + 1])
        if (m > max) max = m
      }
      peak = Math.max(max, peak * 0.97)
    }

    /** Add the arrow of cell i to the current path. */
    function arrow(i, cellW, cellH, maxLen) {
      const dx = pooled[2 * i]
      const dy = pooled[2 * i + 1]
      const mag = Math.hypot(dx, dy)
      const len = Math.min(maxLen, mag * scale)
      const r = Math.floor(i / grid.cols)
      const c = i - r * grid.cols
      const cx = box.x + (c + 0.5) * cellW
      const cy = box.y + (r + 0.5) * cellH
      const ux = dx / mag
      const uy = dy / mag
      const tx = cx + (ux * len) / 2
      const ty = cy + (uy * len) / 2
      g.moveTo(cx - (ux * len) / 2, cy - (uy * len) / 2)
      g.lineTo(tx, ty)
      const head = Math.min(6, len * 0.4)
      g.moveTo(tx - head * (ux * HEAD_COS - uy * HEAD_SIN), ty - head * (uy * HEAD_COS + ux * HEAD_SIN))
      g.lineTo(tx, ty)
      g.lineTo(tx - head * (ux * HEAD_COS + uy * HEAD_SIN), ty - head * (uy * HEAD_COS - ux * HEAD_SIN))
    }

    function draw() {
      if ((window.devicePixelRatio || 1) !== dpr) setSize(width, height)
      g.setTransform(canvas.width / Math.max(1, width), 0, 0, canvas.height / Math.max(1, height), 0, 0)
      g.fillStyle = theme.background
      g.fillRect(0, 0, width, height)
      const hasImage = pixels !== null
      const hasVectors = fieldRows > 0 && fieldCols > 0
      if (!hasImage && !hasVectors) {
        g.font = labelFont
        g.fillStyle = theme.textMuted
        g.textAlign = "center"
        g.textBaseline = "middle"
        g.fillText("Waiting for image and vectors…", width / 2, height / 2)
        return
      }
      const prevW = box.w
      const prevH = box.h
      if (hasImage) fitRect(imageW, imageH, width, height, box)
      else fitRect(fieldCols, fieldRows, width, height, box)
      if (box.w !== prevW || box.h !== prevH) poolDirty = true

      if (hasImage) {
        if (imageDirty) {
          fg.putImageData(pixels, 0, 0)
          imageDirty = false
        }
        g.imageSmoothingEnabled = true
        g.drawImage(frame, box.x, box.y, box.w, box.h)
        if (dim > 0) {
          g.globalAlpha = dim
          g.fillStyle = theme.background
          g.fillRect(box.x, box.y, box.w, box.h)
          g.globalAlpha = 1
        }
      } else {
        g.fillStyle = theme.surface
        g.fillRect(box.x, box.y, box.w, box.h)
      }
      if (!hasVectors) return

      if (poolDirty) pool()
      const rows = grid.rows
      const cols = grid.cols
      const cells = rows * cols
      const cellW = box.w / cols
      const cellH = box.h / rows
      const maxLen = Math.hypot(cellW, cellH)
      if (!hasImage && cells <= 400) {
        g.strokeStyle = theme.grid
        g.lineWidth = 1
        g.globalAlpha = 0.5
        g.beginPath()
        for (let c = 1; c < cols; ++c) {
          const x = Math.round(box.x + c * cellW) + 0.5
          g.moveTo(x, box.y)
          g.lineTo(x, box.y + box.h)
        }
        for (let r = 1; r < rows; ++r) {
          const y = Math.round(box.y + r * cellH) + 0.5
          g.moveTo(box.x, y)
          g.lineTo(box.x + box.w, y)
        }
        g.stroke()
        g.globalAlpha = 1
      }

      // Bucket cells by color (counting sort); bucket BUCKETS = too short for an arrow.
      const cmax = colorMax > 0 ? colorMax : Math.max(1, peak)
      counts.fill(0)
      for (let i = 0; i < cells; ++i) {
        const mag = Math.hypot(pooled[2 * i], pooled[2 * i + 1])
        const b = mag * scale < 1.5 || mag !== mag ? BUCKETS : Math.min(BUCKETS - 1, Math.floor((mag / cmax) * BUCKETS))
        bucketOf[i] = b
        ++counts[b]
      }
      for (let b = 0, start = 0; b <= BUCKETS; ++b) {
        const n = counts[b]
        counts[b] = start
        start += n
      }
      for (let i = 0; i < cells; ++i) order[counts[bucketOf[i]]++] = i
      // counts[b] is now the end of bucket b.

      g.lineCap = "round"
      g.lineJoin = "round"
      const arrows = counts[BUCKETS - 1]
      if (arrows > 0) {
        g.beginPath()
        for (let k = 0; k < arrows; ++k) arrow(order[k], cellW, cellH, maxLen)
        g.strokeStyle = halo
        g.lineWidth = 3.75
        g.stroke()
        g.lineWidth = 1.75
        for (let b = 0, k = 0; b < BUCKETS; ++b) {
          const end = counts[b]
          if (end === k) continue
          g.beginPath()
          for (; k < end; ++k) arrow(order[k], cellW, cellH, maxLen)
          g.strokeStyle = ramp[b]
          g.stroke()
        }
      }
      // Still cells: a small dot keeps the grid readable.
      if (counts[BUCKETS] > arrows) {
        g.fillStyle = ramp[0]
        g.globalAlpha = 0.5
        g.beginPath()
        for (let k = arrows; k < counts[BUCKETS]; ++k) {
          const i = order[k]
          const r = Math.floor(i / cols)
          const c = i - r * cols
          g.rect(box.x + (c + 0.5) * cellW - 1, box.y + (r + 0.5) * cellH - 1, 2, 2)
        }
        g.fill()
        g.globalAlpha = 1
      }
    }

    const offImage = ctx.onValues("image", values => {
      const v = values[0]
      if (!v || !(v.array instanceof Uint8Array) || v.shape.length < 3) return
      const shape = v.shape
      const h = shape[1]
      const w = shape[2]
      const channels = shape.length > 3 ? shape[3] : 1
      const count = w * h
      if (count <= 0 || (channels !== 1 && channels !== 3 && channels !== 4)) return
      const offset = (shape[0] - 1) * count * channels
      if (offset < 0 || v.array.length < offset + count * channels) return
      if (!pixels || w !== imageW || h !== imageH) {
        imageW = w
        imageH = h
        frame.width = w
        frame.height = h
        pixels = fg.createImageData(w, h)
        const d = pixels.data
        for (let j = 3; j < d.length; j += 4) d[j] = 255
        poolDirty = true
      }
      imageChannels = channels
      convertToRgba(v.array, offset, count, channels, bgr, pixels.data)
      imageDirty = true
      ctx.requestFrame(draw)
    })

    const offVectors = ctx.onValues("vectors", values => {
      const v = values[0]
      if (!v || !(v.array instanceof Float32Array) || v.shape.length < 4 || v.shape[3] !== 2) return
      const rows = v.shape[1]
      const cols = v.shape[2]
      const size = rows * cols * 2
      const offset = (v.shape[0] - 1) * size
      if (size <= 0 || offset < 0 || v.array.length < offset + size) return
      if (field.length !== size) field = new Float32Array(size)
      field.set(v.array.subarray(offset, offset + size))
      fieldRows = rows
      fieldCols = cols
      poolDirty = true
      ctx.requestFrame(draw)
    })

    readSettings(ctx.settings)
    applyTheme()
    setSize(ctx.size.width, ctx.size.height)
    ctx.requestFrame(draw)

    return {
      resize(w, h) {
        setSize(w, h)
        ctx.requestFrame(draw)
      },
      settingsChanged(s) {
        readSettings(s)
        ctx.requestFrame(draw)
      },
      themeChanged(next) {
        theme = next
        applyTheme()
        ctx.requestFrame(draw)
      },
      destroy() {
        offImage()
        offVectors()
        canvas.remove()
      },
    }
  },
}
