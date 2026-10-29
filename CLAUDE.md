# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

sjxl is a JPEG XL encoder and decoder whose numeric work runs as Slang compute
kernels through slang-rhi (Metal on macOS; CUDA/Vulkan on Linux later). The
CPU parses and writes headers, the TOC and files. Everything else, entropy
coding included, is a kernel. `docs/decisions.md` has the design, the
measurements and the roadmap (M0–M5).

## Commands

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
ctest --preset macos-arm64-debug                            # all tests, serial (they share the GPU)
ctest --test-dir build/macos-arm64-debug -R prefix          # tests matching a name
build/macos-arm64-debug/bin/sjxl_gpu_tests "prefix sum*"    # one Catch2 case by name (or a [tag])
cmake --build build/macos-arm64-debug --target sjxl_gpu_tests   # one binary
build/macos-arm64-debug/bin/sjxl info [in.jxl]              # device info, or a file's headers and frame list
build/macos-arm64-debug/bin/sjxl encode in.png out.jxl --distance 1.0 [--no-aq] [--no-cfl] [--no-filters] [--no-acs] [--lambda x] [--clusters n] [--progressive | --qprogressive] [--progressive-dc] [--fixed-tree]
build/macos-arm64-debug/bin/sjxl encode in.png out.jxl --lossless   # Modular frame, bit-exact; RGB/RGBA, 8/16-bit
build/macos-arm64-debug/bin/sjxl animate out.jxl --fps 10 [--loops n] [--lossless] [-d d] f1.png f2.png ...   # animation, one tick per frame
build/macos-arm64-debug/bin/sjxl decode in.jxl out.png                  # every stage on the GPU; animations write out_000.png, ...
build/macos-arm64-debug/bin/sjxl compare a.png b.png                    # PSNR + largest diff, on the GPU
build/macos-arm64-debug/bin/sjxl pixel a.png x y ; sjxl convert in.png out.png   # one pixel; re-save as RGB8
SJXL_DEBUG=1 SJXL_DEBUG_BLOCK=57 build/macos-arm64-debug/bin/sjxl decode ...    # dumps one block's coefficients, DC, orders
SJXL_DEBUG=1 SJXL_DEBUG_ACS=1 SJXL_DEBUG_TABLES=/tmp/t.bin ... decode ...      # strategy map per block; dequant tables to a file
cjxl in.png x.jxl -e 7 -d 1 --epf=0 --gaborish=0   # isolates the transform path from the filters
~/tools/slang/bin/slangc shaders/sjxl/<dir>/<file>.slang -I shaders -target metal -entry <entry> -stage compute -o /dev/null
cjxl in.png ref.jxl -d 1 -e 5 ; djxl out.jxl check.png      # Homebrew oracle (/opt/homebrew/bin)
```

- **Shaders** under `shaders/sjxl/` are copied to `build/<preset>/shaders` by the
  build and compiled at run time by slang-rhi. A shader edit needs any build
  (the copy step), no C++ rebuild; a new `.slang` file needs a re-configure.
  Module names are paths: `ComputeKernel::create(lib, "sjxl/algo/prefix_local", "prefixLocal")`.
- **Tests** are `sjxl_<area>_tests` (Catch2, `tests/CMakeLists.txt`, `sjxl_test`).
  They SKIP when no device opens and never fall back. `tests/support/TestDevice.h`
  gives one device and shader library per process.
- **Environment**: `SJXL_BACKEND=metal,cuda,vulkan` (order to try),
  `SJXL_GPU_VALIDATION=1` (debug layers), `SJXL_SHADER_DIR`, `SJXL_SHADER_CACHE`
  (default `~/Library/Caches/sjxl/shaders`).
- **Toolchain**: Slang 2026.14.1 at `~/tools/slang` (`SLANG_ROOT`), slang-rhi
  fetched by `cmake/Dependencies.cmake` pinned to commit `e17f6d7` with the two
  patches in `cmake/patches/` (same recipe as the sibling project athenea).

## Rules this codebase holds to

- **No CPU arithmetic on image or coded data.** The CPU reads and writes
  files, parses and emits headers, TOC and container boxes, and does
  bookkeeping (sizes, offsets, section tables). Colour transforms, DCTs,
  quantisation, tokens, histograms, code tables, bit emission and symbol
  decoding are kernels. Tests check results with kernels too (mismatch
  counters, PSNR/max-diff metrics) and read back only counters, section
  sizes or the final bytes/pixels to write a file.
- **Shader parameters are bound by name** through reflection
  (`cursor["coeffs"].setBinding(...)`, uniforms as `ConstantBuffer<Params> params`),
  never by slot. Dispatch sizes are in threads; `[numthreads]` is read from
  reflection; kernels bounds-check. `setData` writes the argument's own type:
  pass a `float` (cast `std::pow` results), never a double, into a float field.
- **Coefficient memory is libjxl's**: square and tall blocks store
  `[horizontal freq][vertical freq]`, wide blocks `[vertical][horizontal]`
  (`transforms.slang memIndex`). Orders and quant tables are memory indices.
- **Decoder scratch is one `arena` buffer** partitioned by the host; a thread's
  code sets, trees and channels are arena offsets, so AC threads read the code
  set the globals thread decoded. A thread reports errors through `status`
  (code | stage << 8), never silently.
- **Errors are `Result<T>`** with `SJXL_TRY` / `SJXL_ASSIGN`; no exceptions.
  A codestream feature not implemented fails with `ErrorCode::Unsupported`
  naming the feature. Never a silent CPU path.
- **A buffer made without data is zero** on every backend (`Buffer::create`
  clears on non-Metal). Histograms and counters rely on it.
- **Conformance oracle is Homebrew `cjxl`/`djxl`.** libjxl is not linked.
  Numeric tables (quant weights, opsin matrices, EPF/Gaborish constants) are
  computed from libjxl's formulas at init and checked by tests against the
  extracted copies in `cujpegxl/src/*.h`.
- **Encoder entropy coding**: every stream is ANS with contexts clustered on
  the device (`enc/cluster.slang`, ≤ 32 clusters per histogram set): the AC
  groups in libjxl's coefficient contexts (`common/ac_context.slangh`,
  `enc/ac_sections.slang`), the DC groups' DC stream in the leaves of the
  kWPFixedDC tree over the weighted predictor (`enc/wp.slangh`, serial per
  group) and their AcMetadata in the kACMeta tree (`enc/dc_tokens.slang`).
  Tokens are counted in parallel, then one thread per group runs each chain
  twice (size, then placement backwards from the stream's end). Every token
  must be in the stream, zero-cost ones included. ANS histograms go through
  `ansWriteBestHistogram` (precision 0/6/11 or flat, whichever is cheapest,
  run-length coded widths); single-thread kernels keep 256-entry scratch in
  groupshared arrays, never private ones (Metal's compiler hangs). Context maps over many
  contexts are MTF + ANS streams; the headers' MA trees use prefix codes.
- **One Slang and one slang-rhi** per process, the pinned ones.
- **`docs/decisions.md`** records each subsystem's design, what was measured
  and what is not done. Update it with the change.
- **The three untracked clones** at the repo root are reading material and
  gitignored: `cujpegxl/` (CUDA encoder: GPU DC/AC token pipeline in
  `src/frame_encoder.cu`, `entropy.cu`, `prefix_build.cu`; minimal codestream
  profile in `src/bitstream/`; extracted libjxl tables), `gjxl/` (Metal/CUDA
  encoder; `docs/` on tokenisation and entropy coding; `src/gpu/metal/kernels`),
  `libjxl/` (ground truth: `lib/jxl/dec_*.cc`, `quant_weights.cc`,
  `dct-inl.h`, `dec_ans.cc`, `modular/`). Do not build them here.

## Architecture

Modules are static libraries `sjxl_<name>` aliased `sjxl::<name>`
(`cmake/Warnings.cmake: sjxl_add_module`), listed in dependency order in
`modules/CMakeLists.txt`; each links only the ones above it.

| Module | What it holds |
|---|---|
| core | `Result`/`Error`, `log::*`, `platform::{env,cacheDirectory,executableDir}` |
| gpu | `Device` (backend choice, Slang session, disk shader cache), `ShaderLibrary`, `ComputeKernel`, `CommandBatch` (one encoder per stage; `clear`, `copy`, `submit`), `Buffer` (Device/Upload/Readback), `algo::PrefixSum` (exclusive uint32 scan, 3 dispatches, no atomics) |
| codestream | CPU-side syntax only: `BitWriter`, U32/U64/F16 field coders, SizeHeader, ImageMetadata, FrameHeader, LoopFilter, TOC (`Headers.h`); `Tables.h`: quantizer params from a distance, DCT8 dequant weights, natural order, computed from libjxl formulas |
| image | file I/O only: PNG/PPM/JPEG via stb (`readRgb8` 8-bit RGB, `readImage8` keeps alpha, `readImage` keeps alpha and 16-bit depth; `Rgb8` has `channels` 3/4 and `bitDepth` 8/16; `writePng` writes 16-bit PNGs itself; `readFile`, `writeFile`) |
| metrics | `metrics::compare`: PSNR, largest difference and where, measured by `shaders/sjxl/metrics/compare.slang`; what tests and `sjxl compare` use |
| encoder | `enc::Encoder::encode(Rgb8, Options{lossless, distance, adaptiveQuant, chromaFromLuma, loopFilter, acClusters, acStrategy, acsLambda, acsSpecials})`: RGB/RGBA, 8/16-bit; lossless → `encodeLossless` (rctYCoCg, then `encodeModularFrame`, also the DC frame's writer: modHist → cluster → buildModularGlobalBlob (tree, histograms, RCT header, the stream when the image fits a group) → modGroupSections (ANS per group)); lossy → the VarDCT pipeline (`shaders/sjxl/enc/`): xyb → aq (quant field) → gaborish (inverse) → cfl (dct, per-tile least squares) → quantize (AdjustQuantBlockAC, dead zones) → strategy + large (transform selection per 32×32 superblock, then the 64-pixel families per 64×64 region) → transform (adjustQuantField, dctRows/dctCols for any varblock, dcFromLlf, acTokenize) → cfl → quantize (per varblock) → dc_tokens (dcTokenize: WP + kWPFixedDC per DC group; metaTokenize per item with kACMeta) → cluster per set → codes (trees, ANS histograms and context maps per set) → ac_sections + dcGroupSections (ANS per group, count then emit) → layout → emit |
| decoder | `dec::Decoder::decode(bytes)`: `codestream::parse` on the CPU, then `shaders/sjxl/dec/`: sections (one thread per section: globals, DC groups, AC groups; libraries entropy/modular/frame/modframe .slangh over one `arena` buffer) → VarDCT: tables → dc_finalize → transforms; Modular: finalise (inverse transforms) → modular_out (channels to planes) → filters → output |
| decoder (M2) | `dec::Decoder`: the GPU decode pipeline |

Granularity glossary: block 8×8 px; varblock = one transform (8×8 to 256×256);
tile 64×64 px (chroma-from-luma); AC group 256×256 px (one entropy-coded
section per pass); DC group 2048×2048 px (one section: DC + AcMetadata).

Encoder stages (each a kernel, one `CommandBatch` per stage): sRGB8→XYB per
pixel → adaptive quant field (libjxl's masking model per 4×4 cell and block,
`aq.slang`; the quantizer's global scale is 0.39/distance as libjxl's hare
tier) → inverse Gaborish → transform selection per 32×32 superblock
(`strategy.slang`: DCT8 and the 8×8 specials IDENTITY/DCT2x2/DCT4x4/DCT4x8/
DCT8x4/AFV, 8×16, 16×8, 16×16, 16×32, 32×16, 32×32; `large.slang`: 64×64,
64×32, 32×64 per 64×64 region through the rowPass buffer; libjxl's
EstimateEntropy model, sqrt|q| per non-zero and entropy_mul per transform,
plus λ = 128 times the squared rounding error) → separable forward DCT of every varblock
(`transform.slang`, libjxl's coefficient memory; `common/transform_basis.slangh`
holds the bases and strategy tables) → DC from the lowest frequencies →
chroma-from-luma per 64 px tile (least squares on the scaled coefficients) →
quantise per varblock (AdjustQuantBlockAC per channel, Y dead zones, CfL
residuals, x_qm_scale) → AC tokens per varblock and channel (contexts from
the firstBlock grid, `ctx << 18 | value` per coefficient slot) → DC gradient
residual and AcMetadata tokens (YtoX, YtoB, strategy and qf per varblock, sharpness 4) as tokens in the leaves of
libjxl's fixed trees (DC: weighted predictor + kWPFixedDC, one thread per DC
group; AcMetadata: kACMeta per item) → histograms (atomics, per context) →
clustering per histogram set (one thread-group each) → ANS histograms,
alias/reverse tables, context maps and the per-group headers with their
trees → stream sizes (one ANS chain per AC group and two per DC group) →
layout (section bit offsets and byte sizes) → bit emission by
`InterlockedOr` into a zeroed body buffer (every chain placed backwards by
its thread) → readback of sizes and body; CPU writes headers (LoopFilter:
Gaborish above d 0.5, EPF iterations by distance) + TOC and appends the body.
`sjxl encode --no-aq --no-cfl --no-filters --no-acs --lambda x --clusters n`
switch the stages off or tune them for A/B runs (`--no-specials` drops the 8×8
special transforms, `SJXL_ACS_MASK` limits the families); `SJXL_DEBUG=1`
prints the quantizer, cluster count, strategy census and map, blob bits and
section bytes. Oracle runs: `SJXL_DEBUG=1 SJXL_DEBUG_MAPS=f sjxl decode x.jxl`
dumps a file's quant field and strategy map, `SJXL_ENC_MAPS=f` (plus
`SJXL_ENC_NO_ADJUST=1`) encodes with them. PNG gAMA/cHRM/sRGB chunks are
honoured as libjxl does (`image::Rgb8::gamma`; ICC profiles are not).

Progressive (`--progressive`: three frequency passes; `--qprogressive`: two
precision passes, libjxl's pass lists): `passValue` splits the quantised
coefficients per pass, every pass has its own token buffers, non-zero grid and
histogram set (sets are numbered passes first, then DC/AcMetadata per DC
group, then alpha), AcGlobal carries one orders+histograms block per pass and
the TOC one AC section per (pass, group); `SJXL_PASSES` overrides the list.
`--progressive-dc` writes a DC frame first (a Modular XYB frame at 1/8 size
from `dcFramePlanes`, its DcGlobal carrying the frame's DC steps as dc_quant,
through `encodeModularFrame`) and the VarDCT frame with kUseDcFrame and no DC
stream; cjxl's `-p` is `--progressive --progressive-dc`.
`Encoder::encodeAnimation` writes the animation metadata and one frame per
image through the same `encodeFrame` (full canvas, kReplace, duration in
ticks, `is_last` on the final frame).
Both paths take RGB or RGBA at 8 or 16 bits (`image::readImage`). In a lossy
frame the alpha channel is the extra channel: a one-channel modular image
tokenised like the lossless one, its global part at the end of DcGlobal
(`codes.slang writeExtraChannelGlobal`) and its groups after the AC stream in
every AC group section (`ac_sections.slang`), in its own histogram set after
the DC groups' sets. Lossless (`--lossless`): a Modular frame, YCoCg RCT over the colour channels (alpha is the
fourth modular channel and an extra channel in the headers), an MA tree
learned as libjxl's ComputeBestTree (`enc/tree.slang`: property thresholds,
`modSamples` keeps a quarter of the pixels with both predictors' tokens, then
per level histograms → split decision per node → sample assignment; the host
orders the nodes breadth first) and `modTokenizeTree` over all 16 properties
(`--fixed-tree`: per channel libjxl's fixed WP or gradient tree, whichever
`channelEntropy` estimates cheaper, joined under channel splits),
residual tokens kept per sample (`enc/modular.slangh` `modTreeLeaf`, the
decoder's group-local edge rules), clustered ANS histograms,
the global section by one thread (`codes.slang buildModularGlobalBlob`; it
carries the whole image when it fits a 256 px group), one thread per group
otherwise; DC groups and AcGlobal are empty sections.

Decoder stages: CPU parses signature/headers/TOC into a section table →
one-thread kernels decode DcGlobal and AcGlobal (quantizer, block context
map, dequant matrices, coefficient orders, histograms → prefix/ANS tables) →
one thread per DC group (modular DC + AcMetadata, MA tree, predictors) → per
block DequantDC, AdaptiveDCSmoothing, EPF sigma → one thread per AC group
(ANS/prefix tokens → quantised coefficients) → per coefficient dequant and
CfL, per covered block LLF, per pixel separable IDCT (rows, then columns; any
size 8–256), per varblock the 8×8 special transforms → whole-image Gaborish,
EPF passes, XYB→sRGB8 → readback; CPU writes the file.

Modular frames (lossless; `modframe.slangh`): the globals thread decodes the
global tree and the channels that fit a group; group threads decode their
rectangle of the big channels through strided views into the full-image
channels; one thread undoes RCT/palette; a per-pixel kernel scales the
integers into the planes; the output stage copies them (mode 1) or, for XYB
modular, runs the usual colour transform. RCT, palette and Squeeze are undone
in the finalise thread. Patches: reference-only Modular frames come first in
the codestream and keep their planes; the displayed frame's globals thread
decodes the dictionary, and `patches.slang` blends the rectangles per pixel
after the filters (`docs/decisions.md`, "Patches and multiple frames").
JPEG recompression: YCbCr frames with chroma subsampling keep a subsampled
channel's data at the co-located full-grid block; raw quant tables are modular
streams; `chromaUpsample` runs first, the output stage converts YCbCr (mode 2).
Extra channels are modular in every frame (the globals thread builds the
modular image, DC/AC groups decode their modular groups, `ecToPlanes` makes
planes, the output stage writes RGBA when there is alpha).
Upsampling (`upsample.slang`) runs after the filters at the output size, then
patches, then noise (`noise.slang`: per-tile xorshift128+, 5×5 convolution,
strength LUT), then the colour transform.
Displayed frames compose onto a canvas (`output.slang toOutputPlanes` then
`blend.slang`, blending modes incl. alpha) kept in a reference slot when the
frame can be referenced; `Decoder::decodeAll` returns the emitted frames.
Progressive: passes add shifted coefficients per AC group thread; a permuted
TOC is decoded by `toc.slang` and the parser resumes (`Decoder::parseAll`); a
DC frame's planes become the next frame's DC (`dcFromFrame`).
