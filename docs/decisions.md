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
- **M4 decoder completeness**: splines, grey JPEG transcodes, loop filters
  on YCbCr frames, extra channel blending in patches, writing the ICC
  profile into the output file, float samples. (16-bit output, animation
  with blending and partial frames: done.) (64–256 px
  transforms, Modular frames with RCT/palette/Squeeze, lossy modular,
  patches, progressive passes, DC frames, permuted TOCs, upsampling, noise,
  extra channels with alpha output, JPEG recompression: done.)
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
  (dequant with the quant bias and chroma-from-luma per coefficient; the
  lowest frequencies from the DC image, one thread per covered block; the
  inverse DCT of every 8–256 px varblock as two separable passes with one
  thread per pixel, rows into a scratch plane and columns into the XYB
  planes, so no thread holds a block and a 256×256 transform costs 512
  multiply-adds a pixel; IDENTITY/DCT2X2/DCT4X4/DCT4X8/DCT8X4/AFV one thread
  per varblock, per libjxl), `filters.slang`
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
- **The LLF resample scales are the decoder's set.** `dct_scales.h` has two
  families: `DCTResampleScales<8N, N>` (values below 1, the encoder's
  DCFromLowestFrequencies) and `<N, 8N>` (above 1: 1.1089 for 2→16; 1.0258,
  1.1089, 1.2706 for 4→32), and `LowestFrequenciesFromDC` uses the second.
  Using the first gave a smooth low-frequency error (up to 26/255) in every
  block covering more than one 8×8 in both directions, invisible in flat
  regions and in 8×16 blocks.
- **The AFV quant table** (`ComputeQuantTable kQuantModeAFV`) puts
  `afv_weights[0..1]` at raw positions (0,1) and (1,0) — the DC slots of the
  4×8 and 4×4 halves — and `afv_weights[2..4]` at the AFV positions (0,2),
  (2,0), (2,2); the other even/even positions interpolate over `kFreqs`.
  A wrong reading of that mapping left the AFV blocks ±11.
- `cjxl` at effort 1–3 on these photos used only DCT8 with a uniform quant
  field and zero chroma-from-luma, but custom coefficient orders (bit 0 of
  used_orders) and LZ77 in the AcGlobal context map. Effort 5 adds 16×16 to
  32×32 (and 64×64 on the 320 px image), effort 7 adds DCT2X2/4X4/4X8, AFV
  and identity, with chroma-from-luma and a varying quant field.

Measured (ours vs djxl 0.12, `sjxl compare`): our own encoder's streams and
cjxl's at effort 1, 3, 5 and 7 (d 1 and 3) decode within 1 LSB (≈54 dB) on
the test set wherever the stream stays inside the supported feature set,
including EPF iterations 0–3, Gaborish, every 8–32 px transform, AFV and
chroma-from-luma. `cjxl --epf=0 --gaborish=0` streams isolate the transform
path from the filters when a block type misbehaves; `SJXL_DEBUG_ACS=1`
prints the strategy map and `SJXL_DEBUG_TABLES=<file>` dumps the dequant
tables (checked against cujpegxl's extracted DCT8/16/32: ≤ 2e-6 relative).
DCT64X64 (the 320 px photo at effort 5–7, and a 768 px smooth gradient at
effort 7–9, where cjxl uses nothing else) also decodes within 1 LSB; cjxl
never emitted 128/256 px transforms on our inputs, so those share the code
path but are unverified against djxl. Debug build, 320×180:
≈300 ms a decode, dominated by the single-thread section kernels compiling
and running cold; not measured for speed yet (M5).

Not done: RAW quant tables,
modular Squeeze, Modular frames, extra channels, ICC, animation, previews,
progressive passes, TOC permutations, a third level of nested context maps.

## Modular frames (2026-10-29)

Lossless images (`cjxl -d 0`), progressive lossless (`--responsive=1`,
Squeeze) and lossy modular (`-m 1`, XYB with Squeeze) are a single Modular
frame; the same path will carry the reference frames of patches. Design (`shaders/sjxl/dec/modframe.slangh`, entries in
`sections.slang`):

- **Globals thread** (DcGlobal): the DC dequant factors (read for every
  frame), the frame's global MA tree, then the global modular stream:
  GroupHeader, transforms with MetaApply, and every channel no larger than
  the group dimension (`ModularDecode` stops at the first bigger one; meta
  channels, i.e. palettes, never stop it). The full-image channels live in
  the globals arena region, sized by the host as (channels + 1) × pixels.
- **Group threads** decode their rectangle of the remaining channels as
  *views* into the full-image channels (a `stride` field on `Channel`), which
  is libjxl's DecodeGroup + CopyImageTo in one step with no per-group
  storage. Each group reads its own GroupHeader; a group-local transform is
  reported Unsupported (libjxl's encoder never writes one). DC groups take
  the channels with shift ≥ 3, AC groups those with shift 0–2 (single pass).
  Properties use the group-local channel index and coordinates, as in
  libjxl, so a group's prediction never sees its neighbours.
- **Finalise** (one thread, after all groups): inverse RCT and palette over
  the full image, allocating in the AC groups' region (palettes expand into
  full channels). `modular_out.slang` then writes the three float planes:
  samples × 1/(2^bits − 1) for RGB and grey (grey replicated), or Y, X, B ×
  the DC dequant factors with B relative to Y for XYB frames. The output
  stage has a mode that stores those planes as they are (no XYB transform),
  since non-XYB samples are already in the output encoding.
- **Squeeze** (`squeeze.cc`): the parameters (or the defaults a zero count
  stands for, computed from the first non-meta channel at MetaApply),
  MetaSqueeze halving channels and inserting residual channels with
  storage, and the inverse with SmoothTendency, row by row (horizontal) or
  column by column (vertical) in the finalise thread. With in-place
  residuals the channel list ends up small-to-large, so the global stream
  takes the ≤ group-dim ones and the groups the rest; DC groups get those
  with shift ≥ 3. The group rectangle handed to a shifted channel is the
  *unclipped* group dimension shifted, then clipped to the channel (clipping
  first loses a column on odd sizes). `kMaxChannels` is 256 (3 × (1 + 2 ×
  levels) channels after squeezing).
- Group size: `group_size_shift` is honoured everywhere now (`frameDim`
  takes it; the section kernels get `groupDim`), including the VarDCT
  AcMetadata boundary check that assumed 256 px groups.

Measured: `cjxl -d 0` at efforts 1, 3, 7 and 9, with and without
`--responsive=1`, on the four test images, a grey PGM, a six-colour palette
image and a 768 px gradient with `-g 0`, `-g 2` and `-g 3` (128, 512 and
1024 px groups) all decode bit-exact against djxl (`sjxl_decoder_tests
"[modular]"`). Lossy modular (`cjxl -m 1 -d 1`, efforts 3 and 7) is within
1 LSB (≈54 dB).

Not done: YCbCr modular frames, extra channels, floating point and >8-bit
output (16-bit samples decode but the output is 8-bit), group-local
transforms, patches (the reference-frame plumbing: parsing a second frame
and keeping the first), the per-pixel finalise (RCT, palette and Squeeze run
in one thread; fine at these sizes, kernels later: Squeeze is one thread
per row or column).

## Patches and multiple frames (2026-10-29)

cjxl at effort ≥ 7 (and on repeated content at any effort) writes a
reference-only Modular frame holding a dictionary of rectangles, then the
displayed frame with the kPatches flag. Design:

- **Parser**: `Parsed::frames` in codestream order; each frame's header, TOC
  and sections are read, the next frame starts byte-aligned after the last
  section. Reference-only frames must be Modular and saved before the colour
  transform (that is what patches read); the displayed frame must be last.
  `sjxl info in.jxl` prints the headers and the frame list.
- **Reference frames** run through the same `run()` as any frame, at their own
  size (`frameDim` from the frame header's size), and keep their XYB planes
  (after their loop filters) in a `RefFrame` slot instead of producing pixels;
  for a non-XYB image they hold the samples scaled to [0, 1], as libjxl's
  `save_before_color_transform` frame does.
- **Dictionary** (`frame.slangh readPatchDictionary`, first thing in
  DcGlobal, before the DC dequant factors): 10 contexts, reference positions
  (slot, origin, size) and positions (first absolute, then deltas with
  UnpackSigned), one blending (mode, clamp) per position since there are no
  extra channels; capacities 4096 reference positions and 65536 positions
  (Unsupported beyond). The host passes the reference frame sizes for the
  bounds checks.
- **Application** (`patches.slang`, after EPF and before the colour transform,
  as libjxl's render pipeline orders it): libjxl applies the positions
  covering a pixel in dictionary order, and cjxl's additive patches do overlap,
  so one thread per position raced (a lost update showed as a missing glyph,
  diff ≈ 206). The kernel is one thread per *pixel* applying its covering
  positions in order; a single-thread pass first buckets the positions by
  32-row band (in the globals arena above `kGsArenaTop`) so a pixel scans only
  the positions near it. Modes: replace, add, mul (with clamp); the alpha modes
  degrade to replace/add as libjxl does without an alpha channel.

Measured: the two cjxl matrix streams that needed this (effort 7, distance 3)
are within 1 LSB; a generated glyph page (`[patches]` test: 25 reference
rectangles, 808 positions) is within 1 LSB for VarDCT and lossy modular and
bit-exact for lossless.

Not done: extra channels and alpha blending in patches, splines, noise,
animation/blending of several displayed frames, DC frames, upsampling.

## Progressive: passes, DC frames, permuted TOCs (2026-10-29)

`cjxl -p` writes, for VarDCT, a Modular *DC frame* (frame type 1, the image at
1/8, itself in 3 passes with a permuted TOC) followed by the displayed frame
with kUseDcFrame and 3 passes; for Modular images, one frame in 2 passes with
a permuted TOC. Three pieces:

- **Passes** (`frame.slangh decodeAcGlobal`, `acgroup.slangh`,
  `sections.slang`): AcGlobal carries per pass the coefficient orders and
  the AC histograms, kept in a pass table in the arena; the AC group of pass
  p is TOC entry 2 + nDC + p·nAC + g. The AC group thread decodes its passes
  in order, adding `coefficient << shift` to the same slots (libjxl
  interleaves the passes per block, which only matters for its per-pass
  non-zero rows; ours are per call). Modular groups decode per pass the
  channels in the pass's shift bracket (`GetDownsamplingBracket`, computed
  on the host). The order table of every pass is 394k words, added to the
  globals region per pass.
- **Permuted TOC** (`toc.slang`): the permutation is entropy coded and sits
  *before* the section sizes, so the CPU cannot even find the sizes without
  it. `parse()` stops at such a frame (`Parsed::incomplete`), a one-thread
  kernel decodes the permutation and reports the bit after it, the host
  reads both back (bookkeeping) and `finishToc()` reads the sizes, builds the
  logical section table (`permuted_offsets[j] = offsets[permutation[j]]`) and
  parses the frames that follow. `Decoder::parseAll` does the loop; `sjxl
  info` uses it.
- **DC frames**: a DC frame of level L is a Modular XYB frame of size
  ⌈size / 8^L⌉, decoded like any frame and kept as planes in `dcFrames[L]`.
  The frame below it (kUseDcFrame) reads no VarDCT DC in its DC groups (not
  even extra_precision), skips AdaptiveDCSmoothing (FinalizeDC), and takes
  its DC image from the planes (`dc_finalize.slang dcFromFrame`); its DC
  context bucket is 0 (libjxl zero-fills quant_dc).

Measured: `-p` at efforts 5 and 7 (distance 1), `-p -d 0 -e 7` and
`-p -m 1 -d 1 -e 7` on the four test images and a 768 px gradient: within 1
LSB, lossless bit-exact (`[progressive]` test).

## Upsampling and noise (2026-10-29)

- **Upsampling** (`--resampling=2|4|8`; `upsample.slang`): the frame is coded
  at ⌈size / N⌉ (`FrameDimensions::Set`), so `ParsedFrame` carries the coded
  `dim` and the output size `outXsize/outYsize`, and every stage after the
  loop filters (patches, noise, output, reference planes) runs at the output
  size with planes padded to `xsizePadded · N`. Each output pixel is a 5×5
  weighted sum around its source pixel clamped to the window's range; the
  stored 15 / 55 / 210 weights are the triangular half of the kernels of the
  top-left sub-positions and the rest follow by the symmetry in
  `stage_upsampling.cc`, evaluated per tap (`kernelWeight`) rather than
  expanded. Defaults come from a header generated from libjxl's
  `image_metadata.cc` (`UpsamplingWeights.h`); custom weights in the metadata
  replace them. Borders mirror at the coded size.
- **Noise** (`--photon_noise_iso`; `noise.slang`): three random planes at the
  output size, one xorshift128+ generator per `group_dim` tile seeded by the
  visible / nonvisible frame indices and the tile origin (dec_frame.cc counts
  a displayed frame as visible and resets the other), filled row by row in
  batches of 16 words (`bits >> 9 | 0x3F800000`, floats in [1, 2)); the 5×5
  convolution (0.16 × neighbours − 3.84 × centre, mirrored borders); then the
  strength LUT of 8 points over the intensity (Y ± X) / 2 and the mix with the
  colour correlation base ratios, in XYB after patches. The LUT (8 × 10 bits /
  1024) is read in DcGlobal after the patch dictionary.

Measured (`[features]` test): resampling 2, 4 and 8 at distance 1, lossless
with resampling 2, photon noise at ISO 800–6400, and noise with resampling, on
three images: within 1 LSB of djxl.

## Extra channels and alpha (2026-10-29)

Extra channels are modular in every frame (`DecodeGlobalInfo` with
`do_color` false for VarDCT): the globals thread builds the modular image
(colour channels for Modular frames, none for VarDCT, then one channel per
extra channel at ⌈output size / ec_upsampling⌉ with shift log2(ec_upsampling)
− log2(upsampling)) and decodes the global stream; the DC groups decode the
ModularDC group between the VarDCT DC and AcMetadata, the AC groups the
ModularAC group of each pass after the coefficients (ProcessDCGroup /
ProcessACGroup order); the finalise thread undoes the transforms. So the
Modular machinery written for lossless now runs inside VarDCT frames too, and
the AC/DC group arenas take the modular size when a frame has extra channels
(the 4 KB VarDCT budget overflowed in a progressive stream).

Output: `ecToPlanes` scales each extra channel to [0, 1] by 1 / (2^bits − 1)
into its own plane at the output size (upsampled with the same kernel as the
colour when coded smaller); the output stage writes RGBA when an alpha channel
exists (`Rgb8::channels` is 3 or 4; `readImage8` keeps a PNG's alpha, the
compare kernel takes the channel count). Other channel types are decoded and
ignored, as djxl's PNG output does. The patch dictionary parses one blending
per extra channel; a non-trivial one is reported Unsupported.

Measured (`[alpha]` test, berlin_200 with a gradient alpha and a hole): VarDCT
at efforts 5 and 7, lossy modular, resampling 2 and progressive within 1 LSB
in every channel; lossless bit-exact.

Not done: alpha blending in patches, premultiplied alpha (`alpha_associated`:
samples pass through as stored), extra channels of more than 8 bits in the
output, more than 4 extra channels.

## JPEG recompression (2026-10-29)

`cjxl in.jpg` transcodes losslessly into a VarDCT frame in YCbCr (not XYB)
with the JPEG's chroma subsampling and its quantisation tables as RAW
dequant matrices. Decoding it to pixels (not back to a JPEG) needed:

- **Raw quant tables** (`readQuantEncoding` mode 7): a denominator (F16) and
  a 3-channel modular image of 8·size_x × 8·size_y, stream QuantTable(kind)
  with the frame's global tree; the table is den × q in the image's linear
  order (`tables.slang`), the same linear index the formula modes use.
- **Chroma subsampling** (`YCbCrChromaSubsampling`): channel modes map to
  shifts `maxhs − kHShift[mode]`; block counts round up to multiples of
  2^shift (`frameDim`). A subsampled channel codes one block per 2×2 of the
  Y grid; its DC values, non-zero counts, coefficients and dequantised
  values live at the *co-located full-grid block* (sbx << hshift, sby <<
  vshift), which is exactly where libjxl reads the Y coefficients for the
  chroma-from-luma term and the strategy / quant field, so the per-coefficient
  dequantisation needs no change. The AC group keeps a non-zero grid per
  channel at the channel's resolution (replacing the two alternating rows),
  the IDCT locates a chroma pixel's block on the chroma grid, DC smoothing
  runs per channel on its own grid (compressed_dc.cc's single-channel
  variant), and only DCT8 is allowed when not 4:4:4.
- **Chroma upsampling** (`filters.slang chromaUpsample`), the first pipeline
  stage: horizontal then vertical, 0.75 of the sample and 0.25 of the
  neighbour towards the output side, mirrored at the channel's visible size.
- **YCbCr → RGB** in the output stage (mode 2, stage_ycbcr.cc constants,
  +128/255 on Y).

Measured: JPEGs from `sips` (4:2:0) at qualities 75 and 92 of three images,
transcoded by cjxl, decode within 1 LSB of djxl (`[jpeg]` test). The AC
group arena grew to 32 KB for the per-channel grids.

Not done: grey JPEGs (one component), JPEGs with an embedded ICC profile (the
ICC stream is entropy coded and must be decoded to be skipped), loop filters
on YCbCr frames (cjxl writes none for transcodes), reconstruction of the
original JPEG file (jbrd box).

## ICC profiles (2026-10-29)

An image whose colour encoding is an ICC profile (`want_icc`) carries the
profile right after the image metadata, entropy coded (`icc_codec.cc`): a
U64 size, histograms over 41 contexts, one ANS symbol per byte with a
context from the two previous bytes (`ICCANSContext`), the final-state check,
then the frames start at the next byte boundary. The pixels never depend on
the profile's bytes: an XYB image with an ICC profile is output as sRGB (and
djxl without a CMS refuses to decode it at all), a non-XYB one passes its
samples through. So `toc.slang decodeIcc` only decodes the stream to find
its end; `parse()` stops at the ICC (`Parsed::iccPending`), `Decoder::parseAll`
runs the kernel, `finishIcc()` continues with the frames. Reconstructing the
profile (`UnpredictICC`, for embedding it in the PNG) is not done.

Measured (`[icc]` test): a JPEG written by djxl (with its ICC), transcoded,
re-encoded losslessly and as responsive modular: within 1 LSB / bit-exact.

## 16-bit output (2026-10-29)

Images deeper than 8 bits come out as 16-bit samples, as djxl writes them:
the output stage computes each sample on a 65535 scale and packs two
little-endian bytes (`bytesPerSample`); `image::Rgb8` (the name stayed)
carries `bitDepth`, `readImage` keeps a file's depth and alpha, and the
module writes 16-bit PNGs itself (stb_image_write only does 8: filter-0 rows,
big-endian samples, a stored-deflate zlib stream with Adler-32 and CRC-32).
The compare kernel works on samples of either depth in two passes (sums and
the largest difference, then the first sample reaching it; the old packing of
difference and index in one word had no room for 16-bit differences) and
reports PSNR on the images' own scale.

Measured (`[depth16]` test, berlin_200 as 16-bit with informative low bytes):
lossless and responsive lossless bit-exact; VarDCT at distance 1 within 3 /
65535 (108 dB), lossy modular within 1 / 65535.

## Animation, blending and partial frames (2026-10-29)

Every displayed frame now composes onto a *canvas* of the image's size
(`blend.slang blendCanvas`, one thread per canvas pixel): the frame, converted
to the output encoding (`output.slang toOutputPlanes`), is placed at its
origin over the canvas saved in its blending source slot (zeros when none)
with the frame's blending mode (replace, add, multiply, alpha blend,
alpha-weighted add; extra channels have their own modes and are blended first,
so the colour uses the pre-blend alpha; premultiplied alpha follows
alpha.cc). Outside the frame's rectangle the background shows through. The
canvas is then packed to bytes, and kept in a reference slot when the frame
`CanBeReferenced()` (not last, and duration 0 or an explicit slot) with
`inXyb = false`: patches use slots saved before the colour transform,
blending slots saved after, and a mismatch is an error, as in libjxl. A frame
is emitted when it is last or has a duration; `Decoder::decodeAll` returns
them with durations, `decode()` the first, and `sjxl decode` writes
`out_000.png, out_001.png, …` for animations.

Two facts this exposed: libjxl keeps out-of-range values through the transfer
function (the sRGB curve applies to the magnitude with the sign restored,
gamma gives 0 below zero) and clamps only when packing bytes — clamping before
blending gave a blue of 20 instead of 0 where a lossy frame's ringing went
negative under a translucent edge; and the alpha blend is computed in floats
on both sides, so blended frames of a *lossless* animation can still differ
by one 8-bit step from djxl (236 of 18432 pixels did).

Measured (`[animation]` test, 4-frame RGBA APNGs made by
`tests/support/apng_make.py`, djxl's APNG split by `apng_split.py`): full
frames lossless bit-exact, VarDCT and lossy modular within 1; partial frames
blended over with alpha within 1 in all three encodings.
