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
- **Encoder entropy coding.** Every stream is ANS with clustered contexts
  (M3): the AC groups in libjxl's coefficient contexts, the DC and
  AcMetadata streams in the leaves of libjxl's fixed trees. The chain is
  serial per stream, one thread per group, run twice (size, then placement
  backwards from the stream's end); histograms are counted in parallel first.
  Prefix codes remain for the headers' own small codes (the MA trees). ANS
  alone gains little; the context modelling it makes affordable gains 12 % on
  the AC and 16–41 % on the DC streams.

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
- **M3 encoder quality**: (done 2026-10-29: inverse Gaborish + LoopFilter as
  libjxl chooses it, EPF sharpness 4, CfL per tile, adaptive quant field,
  AdjustQuantBlockAC; ANS for the AC groups with clustered coefficient
  contexts; lossless Modular encoding of RGB and RGBA at 8 and 16 bits;
  transform selection DCT8..64x64 with libjxl's entropy model and the 8x8
  specials; PNG colour chunks; DC and AcMetadata with the weighted predictor,
  libjxl's fixed trees and ANS; lossless with the weighted predictor and
  the per-channel predictor choice; alpha and 16-bit in the lossy path;
  animations of full frames; progressive AC passes and DC frames; a learned
  MA tree and libjxl's histogram precision search for lossless.)
  The RCT is chosen per group among libjxl's four hare-tier candidates;
  palettes, run-length LZ77 and per-cluster hybrid-uint configurations as
  libjxl's hare tier; its modular group sizes.)
  Next: Squeeze (lossy modular), three-channel strategy costs, frames that
  reference the previous one in animations, encoder speed (the serial ANS and
  WP passes).
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

## M3 encoder: adaptive quantisation, chroma-from-luma, loop filter (2026-10-29)

The first M3 slice ports libjxl's hare-tier (`-e 5`) perceptual heuristics,
each as a kernel, so the encoder's choices at a given distance are close to
cjxl's:

- **Quant field** (`enc/aq.slang`): `aqDiff` per 4×4 cell computes libjxl's
  masking measure of Y (the gamma-corrected 4-neighbour Laplacian, squared,
  limited to 0.2, `MaskingSqrt`, summed ×0.25: the `pre_erosion` image);
  `aqMap` per block runs `FuzzyErosion` (the four smallest of each cell's 3×3
  neighbourhood with the distance-dependent weights, four cells per block)
  then `PerBlockModulations` (`ComputeMask`, `GammaModulation`,
  `HfModulation`, min with `BlueModulation`, `exp` with the dampening ramp
  for distances 2..14), and `aqRawField` turns the float field into 1..256
  under the global scale. The field uses the planes *before* the inverse
  Gaborish, as libjxl does, and distance × 0.62 when the Gaborish is off.
- **Quantizer**: libjxl's speeds up to squirrel do *not* derive the global
  scale from the field's median; they call
  `ComputeGlobalScaleAndQuant(quant_dc, 0.39 / distance, 0)` and only the
  butteraugli iterations of kitten and slower use the median. So the
  quantizer is a function of the distance, computed on the host
  (`quantParamsFor`) and uploaded; a first version that took the median on
  the device through a log histogram gave a 5-level field (raw qf 2..6) and
  worse quality. With the uniform field (`--no-aq`) the scale and the field
  are 0.79 / distance, libjxl's fast tiers.
- **Inverse Gaborish** (`enc/gaborish.slang`): the 5×5 `WeightsSymmetric5`
  kernel from `kGaborish[5]` with mul 1, mirrored borders, applied after the
  field is computed. LoopFilter as `LoopFilterFromParams`: Gaborish when
  distance > 0.5, EPF iterations by thresholds 0.7 / 1.5 / 4.0; the
  AcMetadata sharpness channel is 4 everywhere through the MA leaf's offset
  (so it still costs no bits). `x_qm_scale` is 3 up to distance 2.5 then
  4, 5, 6 (`b_qm_scale` 2); the encoder multiplies X's quant by
  1.25^(x_qm_scale − 2) as the decoder divides its steps.
- **Chroma-from-luma** (`enc/cfl.slang`): `dct` keeps the float DCT8
  coefficients of every block for the two consumers; `cflTile` per 64 px tile
  regresses X and B on Y over the tile's blocks with the coefficients scaled
  by `Scale() · 128 · qf / dequant` (libjxl's "strange multiplier"), the
  fast closed-form `FindBestMultiplier` with the 2.6 towards-zero bias,
  clamped to −128..127 in 1/84 units. On the test images ytox is always 0
  (also in cjxl's files) and ytob ranges about −28..13.
- **Quantise** (`enc/quantize.slang`): `QuantizeRoundtripYBlockAC` at the
  hare tier: `AdjustQuantBlockAC` for Y, X, B from the field's value (finer
  for flat blocks and for blocks whose high frequencies would all round to
  zero, coarser in busy blocks, Y dead zones tuned from the rounding errors),
  the largest quant wins and is written back to the field; Y quantised and
  reconstructed with the bias; X and B as residuals against it with the
  tile's factors (ytox/84, 1 + ytob/84). The chroma DC uses the DC factors
  (0, 1) that the decoder's `DequantDC` adds, not the tile factors as
  libjxl's encoder does.

Measured with `djxl` as decoder and SSIMULACRA2 against the source, at the
file size cjxl `-e 5` produces for the distance (our distance found by
bisection), Debug build:

| image | d | cjxl bytes / SSIM2 | ours, same d | ours at cjxl's size | Δ at size |
|---|---|---|---|---|---|
| berlin_200 | 1 | 5762 / 59.77 | 6051 / 59.84 | 5748 / 59.53 | −0.24 |
| berlin_200 | 3 | 2654 / 48.84 | 2966 / 48.56 | 2653 / 45.16 | −3.68 |
| sveti_crop_257x129 | 1 | 9780 / 88.55 | 10233 / 86.20 | 9772 / 85.77 | −2.78 |
| sveti_crop_257x129 | 3 | 5079 / 76.95 | 5092 / 74.96 | 5085 / 74.87 | −2.09 |
| sveti_stefan_320 | 1 | 16527 / 86.79 | 18164 / 84.40 | 16522 / 83.28 | −3.51 |
| sveti_stefan_320 | 3 | 7641 / 72.75 | 8406 / 69.96 | 7625 / 66.71 | −6.04 |

Before this slice (M1 profile) the gap at matched size was about −3.8 on
sveti_stefan_320 at d = 1 and the quality at the same distance was 3–4 points
below cjxl's; now the quality at the same distance matches cjxl within
0.3–2.8 points and the files are 0–12 % larger. Ablation at matched size
(mean Δ over the six rows, before AdjustQuantBlockAC): all stages −3.05,
without the field −3.4, without CfL −4.0, without the loop filter −2.7 (cjxl
itself scores +0.3 higher with `--epf=0 --gaborish=0` under SSIMULACRA2, so
the filters are kept for parity and for block boundaries, which the metric
underweights). AdjustQuantBlockAC moved the mean by nothing but brought the
same-distance quality level with cjxl. The remaining gap is coding
efficiency, not quantisation: one prefix-coded AC cluster against cjxl's
ANS with clustered contexts and custom orders, and DCT8 only where cjxl's
strategy map uses 16×16 to 32×16 transforms in the smooth areas. Those are
the next slices. Debug: `SJXL_DEBUG=1 sjxl encode` prints the quantizer, the
field range and the first tiles' cmap; `SJXL_DEBUG=1 SJXL_DEBUG_QF=1 sjxl
decode` dumps any file's quant field per block (used to check the field's
shape against cjxl's: same pattern, cjxl's one step higher where its
AdjustQuantBlockAC raised it).

## M3 encoder: AC entropy coding with contexts, clustering and ANS (2026-10-29)

With the quantisation level with cjxl's, the remaining gap was coding
efficiency: one prefix code for every AC token against cjxl's ANS over
clustered coefficient contexts. Measured on sveti_stefan_320 at d = 1 (same
tokens throughout):

| AC coding | AcGlobal | AC sections | file |
|---|---|---|---|
| one prefix code, trivial context map | 21 B | 15 771 B | 18 164 B |
| prefix codes, 9 clusters, coded context map (prefix) | 1 941 B | 13 922 B | 18 236 B |
| ANS, one cluster | 31 B | 15 713 B | 18 116 B |
| ANS, 9 clusters, MTF + ANS context map | 567 B | 12 963 B | 15 902 B |

So the gain is the context modelling (−12 % on the sections), and ANS is
what makes its context map affordable: a prefix code costs at least one bit
per entry, 7425 entries, 1.9 KB on a 16 KB file; MTF + ANS brings it to 567
bytes. The pieces, all kernels:

- **Contexts** (`common/ac_context.slangh`): libjxl's `NonZeroContext`
  (predicted count from the top and left blocks of the group, default 32 at
  the corner, times the 15 block contexts of the default BlockCtxMap: DCT8
  blocks of Y in context 0, X and B in 7) and `ZeroDensityContext` (non-zeros
  left, position, previous coefficient non-zero), 7425 contexts, 64-symbol
  alphabet (tokens of values below 2^16). `acItems` in mode 0 counts tokens
  per context with atomics into a 7425 × 64 histogram.
- **Clustering** (`enc/cluster.slang`): libjxl's `FastClusterHistograms` as
  one thread-group of 256: per-context entropy and totals, farthest-first
  seeding by `ClusterDistance` (the bits a merge costs, with the prefix-code
  correction for single-symbol histograms) until the farthest context is
  within 48 bits of a seed or there are 32 seeds, then every context to its
  nearest seed (libjxl assigns sequentially against the growing clusters; the
  parallel form assigns against the fixed seeds), clusters renumbered by
  first appearance and contexts without tokens taking their predecessor's
  cluster. `--clusters n` caps the count (1 reproduces the trivial map).
- **ANS histograms** (`enc/ans.slangh`): counts normalised to 4096 (every
  present symbol ≥ 1, the largest absorbing the rounding), serialised as
  libjxl's `ANSEncodingHistogram::Encode` with the small form for one or two
  symbols and the general form with shift 13 (every count exact, no RLE);
  the decoder's alias table built the same way and inverted into a reverse
  map (slot per symbol offset), 4096 entries per cluster. The AC alphabet
  uses log_alpha 6, the context map 5.
- **Sections** (`enc/ac_sections.slang`): one thread per AC group walks the
  group's tokens backwards (blocks in reverse raster, channels B, X, Y, each
  block-channel's token list reversed) through `ANSCoder::PutSymbol`, in mode
  1 to size the section (32 bits of state, raw bits, 16 per
  renormalisation), in mode 2 to place the bits from the end down: in the
  stream a token is [renormalisation bits][raw bits], tokens forward after
  the state, as libjxl's `WriteTokens` produces. The layout reads the AC
  sizes from this kernel; DC and AcMetadata keep the prefix-sum path.
- **Context map** (`codes.slang writeContextMapStream`): the globals thread
  applies the move-to-front transform, builds the one-context ANS histogram
  of the result, writes the nested histogram set and the 7425 tokens
  backwards with the same two-pass trick inside the blob.

Matched-size comparison against cjxl `-e 5` (as in the previous section):

| image | d | cjxl bytes / SSIM2 | ours, same d | ours at cjxl's size | Δ at size |
|---|---|---|---|---|---|
| berlin_200 | 1 | 5762 / 59.77 | 5624 / 59.84 | 5774 / 61.85 | +2.08 |
| berlin_200 | 3 | 2654 / 48.84 | 2780 / 48.56 | 2649 / 47.46 | −1.39 |
| sveti_crop_257x129 | 1 | 9780 / 88.55 | 9173 / 86.20 | 9785 / 86.98 | −1.57 |
| sveti_crop_257x129 | 3 | 5079 / 76.95 | 4721 / 74.96 | 5086 / 77.04 | +0.09 |
| sveti_stefan_320 | 1 | 16527 / 86.79 | 15902 / 84.40 | 16549 / 85.02 | −1.77 |
| sveti_stefan_320 | 3 | 7641 / 72.75 | 7629 / 69.96 | 7638 / 69.98 | −2.77 |

Mean Δ at matched size −0.9 (was −3.1 before this slice, −3.8 with the M1
profile). What remains: the DC section (2348 B of the 15.9 KB here: a single
Gradient leaf, prefix coded) against cjxl's modular DC, the AcMetadata
stream, and the larger transforms cjxl uses in smooth areas. Encode time
rose from 17 to 44 ms (Debug build) on 320×180: the ANS chain is serial per
group and runs twice; a 2-lane interleave or the count pass reused for the
placement are M5 options.

## Transform selection: varblocks up to 32x32 (2026-10-29)

The VarDCT pipeline now codes varblocks: every kernel after the quant field
works on the strategy map (`acs[block] = strategy << 1 | first`,
`firstBlock[block]`), with the geometry helpers of `enc/ac_tokens.slangh`
and the tables of `common/transform_basis.slangh` (the decoder's basis
functions, memory layout, covered-block and order-bucket tables, the
dequant table layout, the natural order generator). The stages, each a kernel:

- `selectStrategies` (`enc/strategy.slang`): one thread-group of 256 per
  32×32 superblock. It loads the Y tile and the 8/16/32-point scaled DCT
  bases into threadgroup memory and, for seven families (32×32, two 32×16,
  two 16×32, four 16×16, eight 16×8, eight 8×16, sixteen 8×8; every family is
  1024 coefficients, four per thread) computes the separable DCT, quantises
  each coefficient with the varblock's field value (AdjustQuantField: the max
  of its blocks', mixed with the mean above distance 1.54) and the kind's
  dequant table, skips the lowest frequencies (coded as DC) and accumulates a
  cost: a token estimate (0.3 bits for a zero, 3 + 2 log2 |q| otherwise) plus
  λ times the squared rounding error in quant steps. Thread 0 then picks per
  quadrant the cheapest of 16×16, two 16×8, two 8×16 or four 8×8, and for the
  superblock the cheapest of 32×32, two 32×16, two 16×32 or the quadrants.
  Candidates outside the block grid are invalid; a block beyond the grid
  costs nothing, so partial superblocks fall back to what fits (the first
  version let an all-infinite comparison pick a 32×32 that overhung the frame,
  which both decoders rejected). `--no-acs` keeps DCT8; `--lambda` and
  `SJXL_ACS_MASK` (a bit per family) are for experiments.
- `adjustQuantField`: the varblock's field at its first block.
- `dctRows` / `dctCols`: one thread per pixel and channel, the separable
  forward scaled DCT of any varblock (the exact inverse of the decoder's
  idctBasis), written to libjxl's coefficient memory (covered block m >> 6,
  slot m & 63). The 8×8 butterfly is gone; the direct form changed the DCT8
  output only in float rounding (same size, SSIMULACRA2 84.403 vs 84.401).
- `dcFromLlf`: a covered block's DC image value from its varblock's lowest
  frequencies, dc = idct(LLF / (rs rs)), the inverse of the decoder's
  LowestFrequenciesFromDC; the DC of X and B is quantised against Y's as
  before.
- `quantize` per varblock: AdjustQuantBlockAC generalised (libjxl's rows ×
  columns view of the coefficient memory, the partial-block exclusions, the
  kMul1/kMul2 step for the large transforms, dead zones widened for chroma on
  blocks of four or more), the non-zero count over the non-LLF coefficients,
  Y's reconstruction kept in a scratch buffer for the chroma residuals.
- `acTokenize` per varblock and channel: the non-zero count's context from
  the group-local grid (a cell holds the normalised count of the varblock
  covering it, read through firstBlock, so no serial state is needed), the
  coefficient tokens in the bucket's natural order with
  ZeroDensityContext(nz, k, covered, log2 covered, prev), each stored as
  `ctx << 18 | value` at the coefficient index's slot, and the context
  histograms. `acSections` walks these backwards per group. The AcMetadata
  stream carries one strategy and one qf per varblock (`count` is the number
  of first blocks); chroma-from-luma loops the tile's varblocks.
- The dequant tables of every kind come from the decoder's table kernel over
  a stub globals region at `Encoder::create` (every encoding "library"), and
  the natural orders of the 13 buckets from `buildOrders`.

λ was tuned on the three test images at matched size (mean Δ SSIMULACRA2
against cjxl -e 5 over d = 1 and 3): λ = 1 −0.98, 2 −1.12, 4 −1.08, 8 −0.79
(DCT8 only −0.9). The small images hardly move; the full-size photos
(cujpegxl/data, 3840×2160) do:

| image | d | cjxl bytes / SSIM2 | DCT8 only at cjxl's size | with selection |
|---|---|---|---|---|
| berlin_central | 1 | 1 217 920 / 84.61 | 82.99 (−1.62) | 83.15 (−1.46) |
| berlin_central | 3 | 558 710 / 66.69 | 59.54 (−7.15) | 63.36 (−3.33) |
| sveti_stefan | 1 | 1 885 027 / 84.46 | 83.20 (−1.26) | 82.52 (−1.94) |
| sveti_stefan | 3 | 751 307 / 56.71 | 44.44 (−12.26) | 54.09 (−2.62) |

At the same distance the selected layouts are 10 % smaller than DCT8 with a
point less quality; at d = 3 on 4K images DCT8 alone was 23–34 % larger than
cjxl at the same distance. The cost model is the obvious lever left (it
prefers the rectangular shapes and rarely picks 16×16 or 32×32 on the small
images; cjxl's maps use them in smooth areas), together with checking our
quant field and strategy maps against cjxl's on the same image.

## Lossy frames with alpha and 16-bit input (2026-10-29)

The VarDCT path takes the same inputs as the lossless one. 16-bit samples
go through `sampleAt` in the XYB kernel (normalised by 65535) and the
ImageMetadata carries the bit depth, so `djxl` writes a 16-bit PNG. An alpha
channel becomes the frame's extra channel, coded as libjxl codes extra
channels of a VarDCT frame: a modular image of one channel whose global part
sits at the end of DcGlobal (has_tree, the tree with its clustered ANS
histograms, the GroupHeader with the global tree and no transforms, and the
channel itself when the image fits a group) and whose groups follow the AC
stream inside every AC group section (a 4-bit GroupHeader, then the ANS
stream over the group's pixels). The channel is tokenised with the lossless
machinery (`extractAlpha`, then `modTokenize` twice for the weighted and
gradient predictors, `channelEntropy`, the host's choice, `remapTokens` into
the alpha histogram set, which is numbered after the DC groups' sets) and is
exact. `acSections` keeps the AC stream's own bit count (`acOnlyBits`) to
place the second stream; the DcGlobal tail and the section writer are the only
format changes, plus the ImageMetadata extra channel, `ec_upsampling` and the
per-channel BlendingInfo in the frame header, shared with the lossless
writer. djxl and our decoder agree within 1–2 LSB on the RGBA, 16-bit and
RGBA 16-bit variants (colour PSNR ≈ 35 dB at d = 1, alpha exact).

## DC and AcMetadata streams: weighted predictor, fixed trees, ANS (2026-10-29)

With the AC groups level with cjxl's, the DC group sections were 17–22 %
larger than cjxl's on the 4K image (DC 177 KB + AcMetadata 65 KB against
198 KB in all at d = 1): a single Gradient leaf and prefix codes against
cjxl's weighted predictor, fixed trees and ANS. The two modular streams of
every DC group now follow libjxl's hare tier:

- **DC** (`enc/dc_tokens.slang dcTokenize`, one thread per DC group): the
  quantised DC of Y, X, B in raster order through the weighted predictor
  (`enc/wp.slangh`, the decoder's port with its state in a per-thread region
  of a scratch buffer: the error rows propagate, so this pass is serial per
  group and channel) and libjxl's `kWPFixedDC` tree, a fixed binary search
  over the WP property (the largest neighbouring error) with 33 cutoffs,
  shallower for small images, built on the host like the lossless tree. The
  token and its leaf context are kept per block; a second kernel walks them
  backwards for ANS.
- **AcMetadata** (`metaTokenize`, one thread per item): libjxl's `kACMeta`
  tree, re-serialised in DecodeTree's breadth-first order on the host (the
  table in enc_encoding.cc is not in that order): Gradient for the
  chroma-from-luma tiles, Left for the qf with splits on the left value, Zero
  for the strategies, and Zero with offset 4 for the EPF sharpness. The ACS/QF
  channel is a two-row image of the varblocks in raster order, so an item's
  "left" is the previous varblock's value (found by scanning back through the
  covered blocks) and the qf row's top is its own strategy. **Every token has
  to be in an ANS stream**, the zero-cost EPF ones included: the decoder's
  state walks through each symbol, and the first version, which skipped them
  as the prefix code could, decoded as corrupt. They cost no bits because
  their contexts form a single-symbol cluster.
- **Histogram sets**: the clustering and the histogram-set writer now work on
  numbered sets (set 0 the AC contexts, 1 + 2g the DC stream of group g,
  2 + 2g its AcMetadata), each with its own context range, cluster count and
  code regions; the group's "pre" and "mid" headers carry the local
  GroupHeader, the tree and the clustered ANS histograms; `dcGroupSections`
  (one thread per group) measures and places the two ANS streams. The
  prefix-code item kernels are gone.

On berlin_central (3840×2160) the DC tokens went from 177 KB to 148 KB and
the AcMetadata from 65 KB to 38 KB at d = 1 (2.9 KB of headers): the DC
group sections are 189 KB against cjxl's 198 KB, and the whole file
1 212 760 bytes against cjxl's 1 217 920 at a higher SSIMULACRA2 (84.81 vs
84.61); at d = 3, 526 909 against 558 710 bytes. At matched size the 4K
photos now come out ahead of cjxl -e 5:

| image | d | cjxl bytes / SSIM2 | ours at cjxl's size | Δ |
|---|---|---|---|---|
| berlin_central | 1 | 1 217 920 / 84.61 | 1 215 696 / 84.84 | +0.23 |
| berlin_central | 3 | 558 710 / 66.69 | 558 467 / 67.26 | +0.57 |
| sveti_stefan | 1 | 1 885 027 / 84.46 | 1 888 580 / 84.55 | +0.08 |
| sveti_stefan | 3 | 751 307 / 56.71 | 748 760 / 57.31 | +0.60 |

The small images average −0.77 (berlin_200 −1.95 / −2.35, sveti_crop +0.35 /
+0.58, sveti_stefan_320 +0.20 / −1.43 at d = 1 / 3): on a 200×112 image the
per-group trees and histogram headers weigh, and the three-channel strategy
cost is still missing.

## The 64-pixel families and libjxl's entropy model (2026-10-29)

cjxl's map of berlin_central at d = 3 is 38 % DCT64X64, and our selection
stopped at 32×32 with a cost model that charged 0.3 bits per zero
coefficient, which no coder pays (coefficients after the last non-zero are
not coded at all) and which punished exactly the large smooth blocks. Two
changes:

- **The 64-pixel families** (`enc/large.slang`): DCT64X64, two DCT64X32 (32
  wide, 64 tall) and two DCT32X64 per 64×64 region. Too large for one
  thread-group's memory, they run through the plane-sized rowPass buffer:
  `largeQf` per candidate (validity, AdjustQuantField), `largeRows` and
  `largeCols` per pixel (the separable DCT, quantisation with the kind's
  table, the cost by atomics into per-candidate accumulators), `mergeLarge`
  per region against the sum of the four superblocks' chosen costs, which
  `selectStrategies` now exports (`sbCost`); a win rewrites the region's map.
  The rest of the pipeline already handled any covered size (the oracle run
  with cjxl's 4K map decoded correctly before this).
- **libjxl's `EstimateEntropy` as the proxy**: `cost_delta · sqrt|q|` per
  non-zero coefficient and nothing for zeros, `zeros_mul · (bits of the
  non-zero count and bits of that)` per block, the per-transform
  `entropy_mul` (DCT8 1, 16×8 1.21, 16×16 1.34, 16×32 1.49, 32×32 1.48, the
  64-pixel ones 2.25, DCT4X4 1.35, DCT2X2 1.19, IDENTITY 1.30, DCT4X8 1.07,
  AFV 1.02, with the distance adjustments that favour DCT2X2/IDENTITY at
  high quality and avoid the other specials above distance 4), plus λ times
  the squared rounding error in quant steps (libjxl uses an L8 norm of the
  pixel-domain error under its masking field; the L2 in coefficient space is
  the cheap stand-in). λ re-tuned at matched size: 4K photos λ = 128 mean Δ
  −0.78 (λ = 256 −1.0), small images λ = 128 −0.43 (λ = 256 −0.96).

| image | d | cjxl bytes / SSIM2 | before (32×32 cap, bits proxy) | now (λ = 128) |
|---|---|---|---|---|
| berlin_central | 1 | 1 217 920 / 84.61 | 83.07 (−1.54) | 83.84 (−0.77) |
| berlin_central | 3 | 558 710 / 66.69 | 63.54 (−3.14) | 66.02 (−0.67) |
| sveti_stefan | 1 | 1 885 027 / 84.46 | 82.53 (−1.94) | 83.18 (−1.28) |
| sveti_stefan | 3 | 751 307 / 56.71 | 54.15 (−2.55) | 56.32 (−0.39) |

Our map of berlin_central at d = 3 now has 37 % of its blocks in 64×64
varblocks. The small images: berlin_200 −1.56 / −1.67, sveti_crop +0.35 /
+1.87, sveti_stefan_320 −0.11 / −1.43 at d = 1 / 3. The oracle gap (cjxl's
maps in our coder: +7 % bytes at its quality) is what the DC and AcMetadata
streams and the three-channel cost (we score Y only) still leave.

## Colour chunks, the 8x8 special transforms and the oracle maps (2026-10-29)

**Where the gap really was.** An oracle experiment settles what the
heuristics are worth: `SJXL_DEBUG=1 SJXL_DEBUG_MAPS=f sjxl decode` dumps any
file's quant field and strategy map, and `SJXL_ENC_MAPS=f` makes the encoder
use them instead of its own (`SJXL_ENC_NO_ADJUST=1` also skips
AdjustQuantBlockAC, since an imported field already went through cjxl's).
With cjxl's own maps our encoder of sveti_stefan_320 at d = 1 reached 85.07
SSIMULACRA2 at 16 448 bytes against cjxl's 86.79 at 16 527; our own maps give
the same quality at matched size, so the quant field and transform
selection were not the gap. Comparing the quantised coefficients of the same
blocks showed ours a few percent smaller at low frequencies and the DC values
off by up to 7 % in X: the *input* differed. The sveti test images carry a
PNG gAMA chunk (gamma 1/2.2) and cHRM, which cjxl honours (a gamma transfer,
sRGB primaries after its chromaticity match) while we read every PNG as sRGB
and wrote sRGB; SSIMULACRA2 then compared our output under the wrong transfer.
On a copy with the chunks stripped the same bytes scored 85.77 instead of
83.61. berlin_200, which has no chunks, was the image where we matched or beat
cjxl all along.

**Colour chunks** (`image::Rgb8::gamma`, `Image.cpp scanPngColor`): the
PNG's chunks before IDAT are read with libjxl's precedence (cICP, iCCP, sRGB,
then gAMA with cHRM): an ICC profile or cICP is not honoured (a note on
stderr, sRGB assumed), sRGB means sRGB, gAMA gives the encoding gamma and
cHRM is checked against sRGB primaries and D65 (a mismatch is noted and
ignored). The XYB kernel linearises with `v ^ (1 / gamma)` for a gamma
transfer, and both header writers emit the ColorEncoding (RGB, D65, sRGB
primaries, have_gamma with the 24-bit gamma × 10^7, relative intent) for
lossy and lossless frames alike, so `djxl` writes the matching profile. The
decoder already rendered gamma transfers (that is how cjxl's files of these
images decoded bit-exactly).

**Special 8x8 transforms** (`enc/special.slangh`): the forward IDENTITY,
DCT2X2, DCT4X4, DCT4X8, DCT8X4 and AFV0..3 as `TransformFromPixels` computes
them (the small DCTs in direct form with the decoder's coefficient memory, the
AFV basis transposed from the decoder's inverse, the 2×2 or 2×1 combination
of the sub-blocks' DCs into positions 0, 1, 8, 9). `specialTransform`
replaces the separable DCT for those blocks; `dcFromLlf`, the orders (bucket
1), the quant tables (kinds 1, 2, 3, 9, 10) and AdjustQuantBlockAC's
exclusion of partial kinds were already generic. In `selectStrategies` one
thread per 8×8 block scores the nine specials against DCT8 with the same
proxy and the quadrant search uses the block's best (`--no-specials` turns
them off). cjxl's map for the 320×180 image uses DCT2X2 on 13 % of the
blocks; ours picks DCT4X8/8X4 and a few IDENTITY and AFV.

At matched size against cjxl -e 5 (Δ SSIMULACRA2), the small images now
average −0.37 (was −0.79): berlin_200 −0.33 / −1.94, sveti_crop +0.56 /
+0.94, sveti_stefan_320 −0.83 / −0.62 at d = 1 / 3. The 4K photos (sRGB
chunk, so no colour effect) stay at −1.5 / −3.1 (berlin_central) and −1.9 /
−2.6 (sveti_stefan) at d = 1 / 3. The oracle on berlin_central at d = 3
explains them: cjxl's map is 38 % DCT64X64, 17 % DCT32X32 and only 13 % DCT8;
with that map our encoder reaches cjxl's quality (66.80 vs 66.69) at 7 %
more bytes, and the same size at −1.4 without the quant adjustment, while
our own selection (capped at 32×32) gives 64.58 at +3 %. So the next levers
are the 64-pixel families in the selection and the remaining coding
overhead (DC and AcMetadata streams).

## Lossless Modular encoder (2026-10-29)

`sjxl encode --lossless` writes a Modular frame: ImageMetadata not XYB
(8-bit sRGB), FrameHeader encoding Modular with no colour transform, 256 px
groups, LoopFilter off. The frame's global stream applies the YCoCg RCT
(type 6) over the three channels; every pixel is predicted with the Gradient
predictor and coded in the context of libjxl's `kGradientFixedDC` fixed tree
(a balanced binary search over the gradient property W + N − NW with 33
cutoffs, shallower for small images), with ANS histograms clustered as the AC
ones. Everything numeric is a kernel (`enc/modular.slang`,
`codes.slang buildModularGlobalBlob`):

- `rctYCoCg` per pixel (enc_rct.cc `FwdRctRow<6>`): Y = tmp + (Cg >> 1), Co =
  R − B, Cg = G − tmp with tmp = B + (Co >> 1).
- `modToken` (`enc/modular.slangh`): a pixel's context and PackSigned
  residual, with the decoder's edge rules inside the rectangle it is coded
  in (a group, or the whole image when it fits a group): at the left edge the
  "left" is the pixel above, at the top the "top" is the left. Lossless means
  the predictor sees the original neighbours, so the residuals and contexts
  of every pixel are independent: `modHist` counts them in parallel with
  atomics, and only the ANS chain is serial.
- The host builds the tree (`fixedGradientTree`, a port of `MakeFixedTree`:
  the shape depends on the pixel count and bit depth, not on the pixels),
  uploads it, and `buildModularGlobalBlob` serialises it as `DecodeTree`
  reads it (breadth-first node order, one prefix code over the six token
  contexts), then the clustered histogram set (`writeHistogramSet`, now
  shared with the AC path: the context map is written with plain bits when
  there are at most 8 clusters over at most 64 contexts, else as the MTF +
  ANS stream), then the global GroupHeader with the RCT, and, when the image
  fits a group, the whole image's tokens as one ANS stream placed backwards
  inside the blob.
- Larger images: each AC group is one thread (`modGroupSections`) coding its
  rectangle of the three channels after a 4-bit GroupHeader (global tree,
  default weighted header, no transforms), measured then placed as the AC
  sections are. The DC groups and AcGlobal are empty sections; the layout is
  host bookkeeping from the bit counts.

Results (bit-exact through `djxl` and through our decoder; `cjxl -d 0 -e 5`
for comparison, which learns a tree per image and uses the weighted predictor):

| image | ours | cjxl -d 0 -e 5 |
|---|---|---|
| berlin_7x5 | 109 B | 108 B |
| sveti_64 (64×36) | 3988 B | 3707 B |
| berlin_200 (200×112) | 30014 B | 27684 B |
| sveti_crop_257x129 | 46141 B | 40975 B |
| sveti_stefan_320 | 84703 B | 78385 B |

8–13 % larger than cjxl with the Gradient predictor and the gradient tree.

**Weighted predictor** (later the same day): with the DC streams' machinery
in place, the lossless channels went through the weighted predictor and the
fixed WP-property tree (`modTokenize`, one thread per group: the error state
is serial, so the residuals and contexts come out of one pass per group and
channel, kept per sample for the ANS pass; the global stream of a small image
reads the same tokens). Sizes: sveti_64 3643 B (cjxl 3707), berlin_200
28 206 (27 684), sveti_crop 41 426 (40 975), sveti_stefan_320 78 928 (78 385):
within 2 % of cjxl -d 0 -e 5, which at this effort picks per channel between
Gradient and the weighted predictor (Predictor::Best) and learns its tree.

**Best** (the per-channel choice): `modTokenize` runs twice, once with the WP
tree and once with the gradient tree, into separate token and per-channel
histogram buffers; `channelEntropy` estimates each channel's bits under an
ideal code per context (n log2(N / n)); the host picks the cheaper predictor
per channel (bookkeeping on eight numbers) and builds the frame's tree: the
chosen tree when the channels agree, else the per-channel subtrees joined
under a chain of splits on the channel property, re-serialised breadth-first
with a leaf map; `remapTokens` renumbers the chosen pass's contexts into it
and counts the final histograms. On the 8-bit test images every channel keeps
the weighted predictor; the synthetic 16-bit variant (a deterministic
low-byte ramp) switches all channels to Gradient and goes from 98 754 (WP) to
88 150 bytes (cjxl 74 386 with its learned tree; the per-channel subtrees,
when tried on it with the same predictor everywhere, gave 85 615, so
channel-separated contexts help wide alphabets while costing 0.3 % on the
8-bit images, which is why the unanimous case keeps one tree). Remaining
headroom: a learned tree, Squeeze/palette for flat or few-colour images, and
the LZ77 RLE. Encode time is dominated by the serial WP and ANS passes per group (two
tokenising passes now; 300 ms Debug for 320×180).

**Alpha and 16 bits.** The same path takes RGBA and 16-bit input
(`image::readImage` keeps both): the alpha channel is the modular image's
fourth channel, declared as an unassociated alpha extra channel
(`ExtraChannelInfo` all_default at 8 bits, explicit type/depth at 16) with
its `ec_upsampling` and `BlendingInfo` fields in the frame header (the
missing per-channel BlendingInfo shifted every later field, which showed as
a 128 px group size in `sjxl info`); the RCT covers channels 0..2 only.
16-bit samples make the residual tokens reach 71, so the modular ANS
alphabet is 128 (log_alpha 7) at that depth and the context histograms have
128 bins throughout; the clustering kernel keeps its distances over those
bins but no longer accumulates the cluster histograms in threadgroup memory
(two 32 × 128 tables would exceed Metal's 32 KB): `accumulateClusters` adds
every context's histogram into its cluster's code region with atomics.
`modular_16_bit_buffer_sufficient` is false at 16 bits (the RCT of 16-bit
samples exceeds int16). Bit-exact through djxl and our decoder; on
berlin_200 variants: alpha 32 767 B (cjxl 27 797), 16-bit 88 150 B (cjxl
74 386), both 93 953 B (cjxl 74 741): cjxl's learned tree and transforms
matter more as the alphabet grows. The lossy path still takes 8-bit RGB only
(alpha and 16-bit input return Unsupported there).

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

Group-local transforms (2026-10-29): with 128 px groups cjxl picks an RCT
per group. A group whose GroupHeader declares transforms decodes into local
channels of the views' shapes (meta channels MetaApply inserted are local
already), undoes the transforms there and copies the result into the image
(libjxl's DecodeGroup + CopyImageTo); groups without transforms keep writing
through the views. `[groups]` test: `-g 0` lossless at efforts 7 and 9
bit-exact, lossy modular within 1.

Not done: YCbCr modular frames, floating point output, the per-pixel finalise
(RCT, palette and Squeeze run in one thread; fine at these sizes, kernels
later: Squeeze is one thread per row or column).

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
- **DC frames**: a DC frame of level L is a frame of size ⌈size / 8^L⌉
  (Modular, or VarDCT itself taking its DC from level L+1 with
  `--progressive_dc=2`), decoded like any frame and kept as planes in
  `dcFrames[L]`.
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

## Encoding animations (2026-10-29)

`Encoder::encodeAnimation(frames, durations, ticksPerSecond, numLoops, options)`
writes one codestream header with the animation fields (ImageMetadata
`extra_fields`: orientation 1, have_animation, tps numerator/denominator,
num_loops, no timecodes; ToneMapping all_default) and then one frame per
image, each lossy (VarDCT) or lossless (Modular) exactly as a still would be
encoded: `encode` and `encodeAnimation` both go through `encodeFrame`, which
returns the frame header, TOC and body of one frame byte-aligned. The frame
header gets `duration` (U32 0/1/8/32 bits), `is_last` and, for a frame that
is not last, `save_as_reference 0`; every frame is a full canvas with blend
mode kReplace, so the decoder's reference slot bookkeeping is trivial and no
frame depends on the previous one. All frames share the first one's size,
channels and depth (checked on the host; the per-frame pipelines are
unchanged). `sjxl animate out.jxl [--fps n] [--loops n] [--lossless] [-d d]
f1.png f2.png …` is the CLI; every frame lasts one tick at `fps` ticks per
second.

Measured: the round-trip test (`[animation]` in `sjxl_decoder_tests`) encodes
three 200×200 frames lossy and lossless; djxl and jxlinfo read the files
(three frames of the given durations, looping), our decoder returns three
frames with the durations and the lossless frames are bit-exact.

Not done: frames that code only what changed (a sub-rectangle blended over
the previous frame, or a frame referencing a saved one through patches), which
is where animation compression comes from; per-frame distances; APNG/GIF
input (the CLI takes one PNG per frame).

## Progressive AC passes in the encoder (2026-10-29)

`Options::progressive` splits every varblock's quantised AC coefficients over
several passes as libjxl's `ProgressiveSplitter::SplitACCoefficients` does:
mode 1 is libjxl's qprogressive (two passes: every coefficient shifted right
by one with rounding towards zero, then the remainder), mode 2 its progressive
(three passes by frequency: the 2x2 lowest frequencies of each 8x8 block,
scaled by the covered blocks, then 3x3, then everything). `passValue`
(`transform.slang`) is the split: zero outside the pass's frequency square or
inside what earlier exact passes covered, otherwise what the previous pass's
shift left, at this pass's shift. The decoder adds `value << shift` per pass,
so the sum reconstructs the single-pass coefficients exactly: the round-trip
test checks our decode of a progressive file against our decode of the
single-pass file at max diff 0, and against djxl at 1.

Per pass the pipeline runs `passNonZeroCount` (the pass's non-zero count per
varblock, the single-pass count is quantize's) and `acTokenize` (tokens and
the non-zero context from the pass's own group-local grid, libjxl keeps one
`num_nzeroes` grid per pass) into pass-indexed token buffers, and each pass
has its own histogram set with its own clustering: the sets are now
numbered passes first (`numAcSets` of 7425 contexts), then the DC groups' DC
and AcMetadata sets and the alpha set (`setCtxBase` in `codes.slang`,
`dc_tokens.slang` and the host agree). AcGlobal writes per pass
`used_orders` (natural) and the pass's clustered histograms; the frame header
carries `Passes` (shifts, the downsampling stops libjxl's `InitPasses`
derives from each pass's `suitable_for_downsampling_of_at_least`); the TOC has
one AC section per group and pass (`FrameDim::numSections(numPasses)`), and
`acSections` runs one thread per (pass, group). An alpha channel's modular
group goes with the last pass, whose downsampling bracket holds shift 0.

Measured on sveti_stefan_320 at d 1: single pass 16488 B, three frequency
passes 17397 B (+5.5 %), two precision passes 17562 B (+6.5 %); cjxl -e 5
pays +4.4 % and +5.9 % for the same modes. The debugging hook `SJXL_PASSES=
"ncoeffs:shift:downsample,..."` overrides the pass list, and the decoder's AC
group failures now say which check failed (non-zero count, leftover
non-zeros, ANS final state) with `SJXL_DEBUG=1` naming the thread.

Not done: group reordering (`centerfirst` permuted TOC); choosing the passes
per image.

## DC frames in the encoder (2026-10-29)

`Options::progressiveDc` (`--progressive-dc`; cjxl's `-p` is this plus the
three frequency passes) writes libjxl's progressive_dc 1: a DC frame before
the VarDCT frame, which then carries flag kUseDcFrame, no upsampling fields,
and DC group sections holding only the AcMetadata (no extra_precision, DC
tree, histograms or DC stream: `buildDcGroupBlobs` and `dcGroupSections`
skip them and `dcTokenize` is not run). The decoder takes the frame's DC from
the DC frame's planes and skips the adaptive DC smoothing, as libjxl does.

The DC frame is a Modular frame in XYB at 1/8 size (frame type kDCFrame,
dc_level 1, 256 px groups, one pass, no filters, not displayed; libjxl's
ImageBundle "lie": the pixels are already XYB). A modular XYB frame holds the
channels Y, X and B − Y as integers that the decoder scales by its own
dc_quant per channel (`DequantMatrices::DecodeDC`, F16 of 128 × the value),
so `dcFramePlanes` rounds each channel of the DC image (`dcValue`, what
`dcFromLlf` measured) to its own grid and makes B relative to Y, and the DC
frame's DcGlobal writes the frame's three DC steps (kDCQuant[c] / (global
scale × quant_dc), the same steps the VarDCT DC would have used) as those
dc_quant values; the kernel applies the same F16 rounding the decoder will.
B is thus quantised on its own grid rather than relative to Y's reconstruction
(the DC CfL of the VarDCT path), and libjxl's lossy modular DC frame (distance
× 0.02) is replaced by one at exactly the frame's DC precision. The modular
machinery is the lossless path's: `encodeLossless` now only runs the RCT and
calls `encodeModularFrame(spec, planes)`, which the DC frame calls with the
RCT off and the custom dc_quant (`CodeParams rct, dcQuantCustom,
dcQuantX/Y/B`). No DC frame with an alpha channel (libjxl disables
progressive_dc with extra channels too; the option is ignored).

Measured at d 1: sveti_stefan_320 16786 B with the DC frame (2039 B of it)
against 16488 B single pass, PSNR 35.59 vs 35.60 dB; the 4K photo (as RGB)
1240668 B against 1212760 B (+2.3 %) with a 480×270 DC frame of four groups
(28 KB, more than the DC stream it replaces: the DC frame's fixed trees and
per-group headers, and B on its own grid instead of relative to Y). Our decode matches djxl within
1 on every test image, with and without the frequency passes (`[dcframe]`
test). Not done: deeper DC levels (progressive_dc 2), DC frames with extra
channels.

## Learned MA tree for lossless (2026-10-29)

`Options::learnedTree` (default; `--fixed-tree` keeps the per-channel WP /
gradient trees) learns the Modular frame's MA tree from the image as libjxl's
`ComputeBestTree` does at its hare tier, as level-synchronous kernels in
`enc/tree.slang`:

- `propStats` samples pixel values and horizontal differences (clamped to
  ±512) and `propThresholds` turns them into the per-property quantisation
  (`PreQuantizeProperties`: quantile thresholds for the pixel, absolute and
  difference properties, the fixed table for the WP property, one bin per
  channel, stream ids by pixel count, 48 values at most).
- `modSamples` (modular.slang, one thread per group, the WP state serial as
  in the tokeniser) keeps a quarter of the pixels (`nb_repeats` 0.5 × 0.5)
  with the residual token of each candidate predictor (Weighted, Gradient:
  `Predictor::Best`, tokenised with the (4, 1, 2) split libjxl's
  `AddSample` uses) and the quantised values of the four candidate
  properties: libjxl's hare order {0, 1, 15, 9, 10, …} with the stream
  property dropped under 30 streams, so {channel, WP, gradient, W − NW} on
  the test images and {channel, stream, WP, gradient} on 4K.
- Per level: `treeHistograms` counts every open node's tokens per
  (property, value, predictor) with atomics; `treeSplit` (one thread-group
  per node) sweeps the thresholds of each property with libjxl's cost
  (`EstimateBits` with the 1/4096 probability floor on each side plus the raw
  bits, the predictor-change penalty 800 / (100 + threshold), the
  static / static-constant / non-static / no-WP categories and
  `fast_decode_multiplier` 1.001) and splits when the saving beats the node
  threshold 75 + 14 × 5 scaled by `LearnTree`'s 0.9 f + 0.1 for the sample
  fraction f; `treeAssign` moves the samples to the children. The host reads
  back two counters per level and stops when a level splits nothing (at most
  4096 nodes, 384 nodes per level).
- The host lays the tree out breadth first with consecutive children (the
  decoder's `TokenizeTree` order; leaf ids in that order) from the learner's
  node records (bookkeeping, no pixel data), and `modTokenizeTree` tokenises
  the frame under it with all sixteen properties computed per pixel.

Histograms: `ansWriteBestHistogram` (ans.slangh) replaces the exact
normalisation: libjxl's `ANSEncodingHistogram::ComputeBest` fast strategy
tries the count precisions 0, 6 and 11 (`GetPopulationCountPrecision`,
counts rounded to representable values, the most frequent symbol taking the
remainder) and the flat histogram, measures the description with a dry run
plus the data bits, and writes the cheapest, with libjxl's run-length code
for five or more identical counts. The decoder read all of this already.
Private arrays of 256 entries in that single-thread kernel made Metal's
compiler hang for minutes; they are groupshared now.

Measured lossless (bytes, this encoder vs `cjxl -d 0 -e 5`): sveti_stefan_320
78527 vs 78385 (+0.18 %; fixed trees were 78928), berlin_200 27919 vs 27684
(+0.85 %; was 28206), sveti_crop_257x129 40947 vs 40975 (−0.07 %),
sveti_64 3556 vs 3707 (−4 %), berlin_7x5 93 vs 108. The trees have 70–135
nodes where libjxl's have 129 on the two-group file (389 single-group);
`SJXL_TREE_THRESHOLD` overrides the node threshold for experiments and the
decoder prints the global tree's node count under `SJXL_DEBUG`. The
remaining gap on berlin_200 is where libjxl tries four RCTs per group.

## The RCT per group (2026-10-29)

libjxl's hare tier tries four reversible colour transforms per modular
stream (none, YCoCg, type 5 = (R, G − (R + B) / 2, B − R), type 10 =
(G, B − G, R − G)) and keeps the one whose gradient residuals estimate
cheapest (`EstimateCost`: the Shannon entropy of the ClampedGradient residual
tokens in 18 contexts by the spread of the three neighbours, plus the raw
bits, per channel). `rctCost` (modular.slang, one thread-group per group and
candidate, the transform applied on the fly to the pixel and its neighbours)
computes that estimate and `rctApply` writes every group's pixels under its
cheapest candidate (ties to libjxl's order) into the planes, recording the
choice in `groupRct`. The transform goes into the stream that holds the
pixels: the global stream's GroupHeader when the image fits a group, each
group's own header otherwise (the global header then has no transforms, since
its transforms would apply to the whole image on top of the groups'; the
first version wrote both and decoded wrongly on two-group images). `rct_type`
is written as its U32 (Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10)).

Measured: the test images keep YCoCg nearly everywhere (sizes within a byte
or two), the 4K photo as RGB encodes to 5338472 B against cjxl's 5579965
(−4.3 %), bit-exact in djxl and our decoder, in 2.0 s on the M5 Pro (cjxl
-e 5: about 1 s).

## Palettes, run-length LZ77 and hybrid-uint configurations (2026-10-29)

The modular encoder now works over a channel table (`chanTable`: width,
height, offset in the planes, grouped): meta channels of any size come first
and belong to the global stream, image channels larger than a group are split
into group rectangles; a group stream numbers its channels from the first
non-meta one (the channel property), the global stream from 0. The global
header carries the global transforms (palettes) and, when the image is one
stream, the RCT; group headers carry each group's RCT.

Palettes (`enc/palette.slang`, libjxl's `try_palettes` at the hare tier,
which applies them without a cost check): `paletteCount` hashes every colour
(up to four 16-bit channels) into a table with counts, claiming a slot with
one compare-exchange on a 32-bit fingerprint and publishing the full key after
it: a spin-wait on another thread's slot never ends inside a SIMD group (the
first version did that and the 200-colour image lost its palette), so
`paletteVerify` checks each pixel's full key afterwards and a fingerprint
collision drops the palette. Within libjxl's colour limit (min(cost × 0.0005
+ pixels / 128 + 128, 1024) with the flat 0.2 bits × depth × channels
estimate) `paletteBuild` orders the colours as libjxl does (frequent ones by
descending luma, then the rest ascending) and `paletteIndex` writes the index
channel; the frame is then [palette (colours × channels), indices] with the
Palette transform (nb_deltas 0, predictor Zero) and no RCT. Otherwise each
channel using fewer than 95 % of its range (and under pixels / 16 values) gets
a compaction palette (`compactStats` / `compactBuild` / `compactApply`: the
used values in order, the samples their ranks), inserted at the front in
application order as the decoder does, and the RCT candidates run over the
compacted channels. Implicit (cube) palette entries are not used.

Run-length LZ77 (`enc/lz77.slang rleScan`, libjxl's `ApplyLZ77_RLE`): one
thread per stream (the groups', then the global one) walks its tokens in
order into a compact list; a run of tokens repeating the previous value
becomes a length token (the run's first context, value run − 3) plus a
distance token (value 1, the special distance (1, 0) of modular streams) in
an extra context when the symbols' cost under the pre-LZ77 statistics beats
CeilLog2Nonzero(len + 1) + 1 bits; the frame takes LZ77 when the total saving
beats 0.2 × symbols + 16 bits, else the lists are rewritten plain. The section
writers read the lists backwards; the header gets min_symbol 224, min_length
3 and the (4, 2, 0) length configuration at alphabet 8. On the test images
the rule never fires (residual runs are already cheap), as with cjxl.

Hybrid-uint configurations (`uintConfigHist` / `uintConfigChoose`, libjxl's
`ChooseUintConfigs` with its fast candidates (4, 2, 0), (4, 1, 2), (0, 0, 0),
(2, 0, 1)): per cluster, the histogram of its tokens under each candidate; the
cheapest by `ANSPopulationCost` (the best histogram form's description plus
data, `ansBestHistogramCost`) plus raw bits plus the configuration's bits
wins, its counts become the cluster's code region and its configuration is
written per cluster; the set's alphabet size follows the largest symbol. The
ANS histogram search now tries libjxl's approximate set of precisions (0, 2,
…, 10, 11). Loop trip counts around the histogram writer are runtime values:
with constants the GPU backend compiler unrolls the writer several times and
never finishes.

Modular frames take libjxl's group size (`GetGroupSizeShift`: 128 px groups up
to 128 px, 256 up to 256, 512 up to 400, 256 beyond), DC frames included.

Measured lossless (bytes, this encoder vs `cjxl -d 0 -e 5`): sveti_stefan_320
78411 vs 78385, sveti_crop 40662 vs 40975, berlin_200 27805 vs 27684, sveti_64
3531 vs 3707, the 4K photo 5334390 vs 5579965 (−4.4 %); synthetic palette
images: 200 colours 100×80 2243 vs 2253, 16 colours 300×200 11650 vs 10764
(+8 %: libjxl's tree has 179 nodes there to our 133). All bit-exact in djxl
and our decoder, 16-bit and RGBA included.

## JPEG recompression in the encoder (2026-10-29)

`sjxl encode in.jpg out.jxl` transcodes a baseline JPEG as cjxl does: a VarDCT
frame in YCbCr (RGB when the JPEG says so through Adobe/JFIF ids), its
quantised DCT coefficients copied, not requantised. The CPU (`image::readJpeg`)
parses the markers only: SOF/DQT/DHT/DRI/SOS and the byte offsets of the
restart intervals; it refuses progressive, arithmetic and 12-bit JPEGs with
`Unsupported`. Everything else is `enc/jpeg.slang`:

- **Huffman decoding on the GPU**: one thread per restart interval (the whole
  scan when there is no DRI), a bit reader with 0xFF00 unstuffing that stops at
  a marker, canonical decode by code length (limit / valoffset tables built from
  the DHT counts on the host — bookkeeping), DC prediction and the run/size AC
  loop into natural order. Interleaved scans walk MCUs; single-component scans
  walk the component's unpadded block grid.
- **jpegToCoeffs** moves each component's blocks to libjxl's layout: the frame
  grid is padded to the subsampling multiple (`frameDim(..., maxHShift,
  maxVShift)`), a subsampled channel keeps its block at the co-located
  full-grid block (as the decoder expects), coefficients are transposed into
  `[horizontal][vertical]`, DC is the JPEG's DC (plus 1024/q0 for RGB JPEGs:
  libjxl's DCzero rule), every block is DCT8 with quant field 1 and no
  chroma-from-luma. Channels X and B of a grey JPEG are empty (zero).
- **Headers**: quantizer global_scale 65536 and quant_dc 1 (inverse scale 1),
  DC dequant F16s of q0/(8·255), the DCT8 table in kQuantModeRAW (denominator
  1/2040 and the table as a 3-channel modular stream, `buildQuantTableBlob`
  in its own blob region between the two halves of AcGlobal), the other 16
  kinds from the library, kSkipAdaptiveDCSmoothing as cjxl writes it on every
  transcode, no loop filter, `x_qm`/`b_qm` absent (not XYB), chroma modes per
  channel (`kH = {0,1,1,0}`, `kV = {0,1,0,1}`).
- **Block context map** as cjxl's (`jpegBlockCtx`): the luma DC histogram is
  cut into equal buckets (1..7 thresholds from the block count and the first
  chroma table's low frequencies, capped so the map has at most 8 contexts
  and fits the simple context-map form), Y gets one AC context per bucket, X
  and B one per two buckets (one for both when grey). `acTokenize` takes the
  block context from the map and the block's luma DC; the AC histogram set
  then has `contexts × 495` entries instead of 7425 (the buffer stride stays).
  The context count is read back as one counter for the cluster and code
  stages.
- **DC and AcMetadata** as any VarDCT frame (weighted predictor, fixed trees,
  ANS), on the per-channel grids.

Two defects found on the way, both general: `jpegToCoeffs` wrote the
chroma-from-luma factors per block into buffers sized per 64-pixel tile, and
the overrun landed in `dcQuant` (five chroma DC values of one row vanished);
and the context clustering seeded from the context with most tokens, which for
a grey transcode is the all-zero non-zero-count context of X and B, whose
distance to everything is its whole count, so the farthest-first seeding
stopped at two clusters (the AC stream was 24 % bigger). Seeds now rank a
one-symbol context last. F16 fields are written as libjxl writes them
(mantissa truncated, not rounded), which made the 4:2:0 transcodes
pixel-identical to cjxl's through djxl.

Measured against `cjxl -e 5` on six JPEGs (PIL and sips, qualities 75..92):

| JPEG | JPEG bytes | sjxl | cjxl | djxl(sjxl) vs djxl(cjxl) |
|---|---|---|---|---|
| 320×180 4:4:4 | 24767 | 21211 | 21259 | cjxl uses CfL (see below) |
| 320×180 4:2:0 | 15029 | 12952 | 13310 | identical |
| 320×180 4:2:2 | 16523 | 14272 | 14631 | identical |
| 200×112 4:2:0 | 4637 | 4060 | 4514 | identical |
| 320×180 grey | 12449 | 11108 | 11330 | max diff 1 |
| 320×213 4:2:0, DRI 20 | 26582 | 22175 | 22618 | identical |

Every file is smaller than cjxl's. For 4:4:4 cjxl applies its JPEG-domain
chroma-from-luma (an integer, exactly reversible residual); djxl's *pixel*
decode of such a file scales the CfL term by the luma table instead of the
chroma one and lands up to 25 LSB from the JPEG's own decode, where ours
(no CfL, 50 dB PSNR against the JPEG) is 4 LSB off. We leave CfL off: the
file is already smaller and decodes closer. The `[jpeg]` test makes 4:2:0
and grey JPEGs with `sips` at two qualities, requires our file smaller than
the JPEG, our decoder within 1 LSB of djxl and 40 dB against the JPEG's
decode.

Not done: cjxl's custom coefficient order (`used_orders` 1), a jbrd box to
reconstruct the JPEG byte-exact, progressive and arithmetic JPEGs, Huffman
decoding in parallel without restart markers (one thread per scan: 80 ms for
a 320-pixel JPEG, mostly fixed cost).
