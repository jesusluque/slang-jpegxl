# Decisions

What was decided, why, what was measured, and what is not done. One section
per subsystem or milestone, appended to as the work goes. Dates are absolute.

## Scope and rules (2026-10-28)

- A JPEG XL **encoder and decoder** (lossy VarDCT, XYB, one frame first) whose
  numeric work is Slang compute kernels run through slang-rhi. Chosen over the
  two GPU encoders read as references (`cujpegxl`, `gjxl`) because both keep
  entropy coding on the CPU; here it is on the GPU, which is the point of the
  exercise and also the rule the sibling project athenea holds to.
- **Standalone**, on athenea's build recipe (presets, `SLANG_ROOT`, slang-rhi
  pinned to `e17f6d7` with two patches), without depending on athenea: the
  codec is a library another program may embed, and a dependency on a
  renderer is not something an embedder wants.
- **Conformance oracle** is Homebrew `cjxl`/`djxl`. libjxl is not linked, so
  the build has no second copy of the format's logic to disagree with.
- **Encoder uses prefix codes everywhere until M3.** Every token's bit length
  is then a table lookup, section offsets are prefix sums and emission is one
  thread per token with `InterlockedOr`. ANS (serial state per section) is a
  compression gain of a few percent to be added later without changing the
  pipeline. `cujpegxl` chooses prefix over ANS below 100 tokens; libjxl's
  decoder accepts either for any stream.

## Build (M0, 2026-10-28)

- slang-rhi patches kept: `metal-command-buffer-errors` (a Metal command
  buffer that fails is a `SLANG_E_OUT_OF_MEMORY` from the next submit or wait,
  not an assert) and `cuda-driver-symbols` (slang-rhi's dlopen'd `cu*` symbols
  in a namespace). Athenea's render and ray-tracing patches dropped. The
  command-buffer patch applies cleanly without the acceleration-structure
  patch; the one hunk it touches there is independent.
- Metal only on macOS (`SLANG_RHI_ENABLE_VULKAN OFF`); no CPU backend, so
  nothing can fall back to it; no WebGPU (Dawn download).
- Shaders are copied, not embedded, and compiled by the device that opens;
  the disk shader cache (`~/Library/Caches/sjxl/shaders/gen1`) makes the second
  run not pay for it.
- `PrefixSum` is athenea's three-dispatch chunked scan: no groupshared memory,
  no atomics, nothing the backends disagree about. Chunk size 256–4096 chosen
  for ~2048 chunks.

## Roadmap

- **M0 skeleton** (done 2026-10-28): modules core/gpu, tests, CLI `info`.
- **M1 encoder, minimal profile**: XYB, DCT8 only, uniform quant field, no CfL,
  LoopFilter off (gab 0, epf 0), DC Gradient predictor, prefix codes, one AC
  cluster. Verified: `djxl` decodes, GPU PSNR vs source, byte-identical
  re-encode, quant table vs `cujpegxl/src/quant_weights_dct8.h`.
- **M2 decoder, VarDCT**: own output, then `cjxl -d {0.5..4} -e {1..7}` of 8-bit
  sRGB PNGs within max abs diff ≤ 1 of `djxl`. All 8–32 px transforms,
  IDENTITY/DCT2x2/DCT4x4/DCT4x8/AFV, ANS + prefix + LZ77, modular MA trees
  with all predictors, RCT/Palette/Squeeze, Gaborish + EPF, custom orders.
- **M3 encoder quality**: inverse Gaborish + default LoopFilter, EPF sharpness,
  CfL per tile, adaptive quant field, DCT16/32 strategy search, context
  clustering (≤ 32, fixed-point costs as cujpegxl), ANS for AC.
- **M4 decoder completeness**: 64–256 px transforms, upsampling,
  patches/splines/noise, progressive, Modular frames (lossless), alpha, >8-bit.
- **M5 platforms and speed**: Linux CUDA/Vulkan, async readback, batching,
  per-stage timings here.

## M1 encoder, minimal profile (2026-10-28)

Profile: XYB, DCT8 only, uniform quant field from the distance (libjxl's
`ComputeGlobalScaleAndQuant` with a flat field), no chroma-from-luma residual
(ytox = ytob = 0, so X and B are coded against the bias-adjusted
reconstruction of Y as the decoder adds it), LoopFilter off (gab 0, epf 0),
DC with the Gradient predictor, prefix codes everywhere, one AC cluster with
the trivial context map. Dead zones are libjxl's fast-tier thresholds (Y
0.56/0.62, X and B 0.58/0.62).

Pipeline (one `CommandBatch` and two readbacks): `rgbToXyb` → `dctQuant`
(one thread per block, three channels) → item passes in histogram mode →
`buildDcGroupBlobs` / `buildGlobalBlobs` (Huffman codes and serialised modular
and histogram headers, one thread per DC group) → item passes in count mode
→ three prefix sums → `layoutSections` → readback of section bytes → item
passes in emit mode (`InterlockedOr` into the zeroed body) + `blobEmit` →
readback of the body. The CPU writes signature, SizeHeader, ImageMetadata,
FrameHeader and TOC.

Measured on the test images (Apple M5 Pro, Debug build, djxl 0.12 as decoder,
`sjxl compare` for PSNR over all channels):

| image | d | ours bytes | ours PSNR | cjxl -e 1 | cjxl -e 3 | cjxl -e 5 |
|---|---|---|---|---|---|---|
| sveti_stefan_320 (320×180) | 1 | 16323 | 34.45 dB | 16958 B, 35.59 dB | 16385 B, 35.59 dB | 16527 B, 35.59 dB |
| sveti_stefan_320 | 3 | 8090 | 30.31 dB | | | |
| sveti_crop_257x129 | 1 | 9550 | 35.46 dB | | | |
| sveti_crop_257x129 | 3 | 4969 | 30.95 dB | | | |

The largest per-pixel difference (99 at (144,63) channel B) is the same for
cjxl and for us: a property of the image in XYB, not a defect. Test floors:
PSNR ≥ 30 dB at d = 1, ≥ 24 dB at d = 3. Encode of 320×180 takes ~6 ms wall
including readbacks (Debug build).

Not done: chroma-from-luma, adaptive quantisation, larger transforms, ANS,
context clustering, the inverse Gaborish + default loop filter (M3). Item
arrays are padded to full groups (3072 AC items per group, 196608 DC items
and 133120 AcMetadata items per DC group), so a 7×5 image scans 330k items;
cheap on the GPU, but a per-group compaction is the obvious M5 saving. The
AC coefficient buffer is int32 (4K: 100 MB); int16 halves it.

Natural order: `dct8NaturalOrder()` follows `CoeffOrderAndLut` literally; the
first diagonal is memory index 1 then 8 (libjxl swaps x and y on odd
diagonals). Coefficient memory layout is `coef[fx * 8 + fy]` with fx the
horizontal frequency, as cujpegxl checked against libjxl.

## M2 decoder, VarDCT (2026-10-29)

Layout of the work on the device (`shaders/sjxl/dec/`):

- `sections.slang`: one thread per entropy-coded section. `decodeGlobals`
  reads DcGlobal (DC dequant, quantizer, block context map, colour
  correlation, global MA tree) and AcGlobal (dequant matrix encodings,
  `num_histograms`, coefficient orders with Lehmer codes, the AC histograms);
  with one TOC entry it also runs the DC and AC group bodies, since they share
  one bit stream. `decodeDcGroups`: VarDCT DC as a three-channel modular image,
  DequantDC with the DC chroma-from-luma factors and the DC context bucket,
  then AcMetadata (YtoX/YtoB per tile, strategies and quant field, EPF
  sharpness) and the raster placement with the overlap and group-boundary
  checks. `decodeAcGroups`: histogram selector, blocks in raster order,
  channels Y, X, B, non-zero count from the group-local prediction, then
  coefficients in coefficient order. Coefficient k of a varblock lives at the
  slot of its k/64-th covered block, k%64, so the dequantisation kernel finds
  it from the `firstBlock` grid.
- `entropy.slangh` / `modular.slangh` / `frame.slangh`: the libraries behind
  it. Everything decoded sits in one `arena` buffer (code sets, context maps,
  prefix tables with Brotli's two-level layout, ANS alias tables, trees,
  channel storage, LZ77 windows), partitioned by the host: 16 MB for the
  globals thread, 4 MB per DC group, 4 KB per AC group. LZ77 windows (4 MB a
  thread) are allocated only on a retry after a thread reports it needed one.
- Then parallel kernels: `tables.slang` (one thread per dequant matrix entry;
  libjxl's library constants and the DCT/ID/DCT2/DCT4/DCT4X8/AFV formulas),
  `dc_finalize.slang` (AdaptiveDCSmoothing, EPF sigma), `transforms.slang`
  (dequant with the quant bias and chroma-from-luma per coefficient; LLF from
  the DC image, inverse DCT for 8–32 px blocks as a direct separable
  transform, IDENTITY/DCT2X2/DCT4X4/DCT4X8/DCT8X4/AFV per libjxl), `filters.slang`
  (Gaborish, EPF0/1/2 as whole-image passes with mirrored borders at the frame
  edge), `output.slang` (XYB → linear → the declared transfer function:
  sRGB, linear, gamma, DCI).

Facts that cost time, kept here so they are not re-learned:

- **Coefficient memory orientation** (`dct-inl.h ComputeScaledDCT`): a wide
  block (rows < cols) stores `[vertical freq][horizontal freq]`; a tall or
  *square* block stores `[horizontal][vertical]`. The natural order and the
  quant tables are just memory indices, so only the IDCT cares.
- **`ShaderCursor::setData` takes the argument's type literally**: a `double`
  (e.g. `std::pow(float, int)` promotes) written into a `float` field leaves
  garbage — here 0, which silenced the chroma channels.
- **slang-rhi resources are untracked** (Metal untracked hazard mode); every
  encoder waits on and updates one queue fence, so passes are serialised
  with memory visibility. `ComputeKernel::dispatch` still issues
  `globalBarrier()` after each pass for the other backends' contract.
- **A nested context-map stream can itself use LZ77**: it then has two
  contexts and a context map of its own (simple form supported; a third
  level is reported Unsupported).
- **The output transfer function is the declared colour encoding's**: a PNG
  with gAMA 1/2.2 makes cjxl declare gamma 0.4545, and djxl applies it. The
  synthetic PPM tests (no gamma) hid this; the photos showed a uniform ±2.
- `cjxl` at effort 1–5 on these photos used only DCT8 with a uniform quant
  field and zero chroma-from-luma, but custom coefficient orders (bit 0 of
  used_orders) and LZ77 in the AcGlobal context map.

Measured (ours vs djxl 0.12, `sjxl compare`): our own encoder's streams and
cjxl's at effort 1–5 (d 1 and 3) decode within 1 LSB (≈54 dB) on the test
set, including EPF iterations 0–3 and Gaborish. Debug build, 320×180:
≈300 ms a decode, dominated by the single-thread section kernels compiling
and running cold; not measured for speed yet (M5).

Not done: transforms of 64–256 px (`flags` reports them), RAW quant tables,
modular Squeeze, Modular frames, extra channels, ICC, animation, previews,
progressive passes, TOC permutations, a third level of nested context maps.
Effort 7 streams still differ (max 12–73): to be investigated (AFV / DCT4x4
paths or EPF sharpness with custom quant field).
