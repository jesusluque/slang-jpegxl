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
build/macos-arm64-debug/bin/sjxl info                       # open the device, print backend/adapter
build/macos-arm64-debug/bin/sjxl encode in.png out.jxl --distance 1.0   # M1
build/macos-arm64-debug/bin/sjxl decode in.jxl out.png                  # VarDCT, every stage on the GPU
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
- **Prefix codes for every encoder stream until M3**; the decoder supports
  prefix and ANS. Keeping token lengths table-driven is what makes bit
  emission a prefix sum instead of a serial chain.
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
| image | file I/O only: PNG/PPM/JPEG via stb (`readRgb8`, `writePng`, `readFile`, `writeFile`) |
| metrics | `metrics::compare`: PSNR, largest difference and where, measured by `shaders/sjxl/metrics/compare.slang`; what tests and `sjxl compare` use |
| encoder | `enc::Encoder::encode(Rgb8, Options{distance})`: the GPU encode pipeline (`shaders/sjxl/enc/`): xyb → dct_quant → items (mode 0 histogram, 1 count, 2 emit) → codes (Huffman + headers per DC group, globals) → prefix sums → layout → emit |
| decoder | `dec::Decoder::decode(bytes)`: `codestream::parse` on the CPU, then `shaders/sjxl/dec/`: sections (one thread per section: globals, DC groups, AC groups; libraries entropy/modular/frame/modframe .slangh over one `arena` buffer) → VarDCT: tables → dc_finalize → transforms; Modular: finalise (inverse transforms) → modular_out (channels to planes) → filters → output |
| decoder (M2) | `dec::Decoder`: the GPU decode pipeline |

Granularity glossary: block 8×8 px; varblock = one transform (8×8 to 256×256);
tile 64×64 px (chroma-from-luma); AC group 256×256 px (one entropy-coded
section per pass); DC group 2048×2048 px (one section: DC + AcMetadata).

Encoder stages (M1 profile; each a kernel, one `CommandBatch` per stage):
sRGB8→XYB per pixel → DCT8 + quantise per block → DC gradient residual and
AcMetadata tokens → histograms (atomics) → 15-bit length-limited prefix codes
+ Brotli-style code-length serialisation per histogram → token bit lengths →
prefix sums (section bit offsets, section byte sizes) → bit emission by
`InterlockedOr` into a zeroed body buffer → readback of sizes and body; CPU
writes headers + TOC and appends the body.

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
in the finalise thread.
