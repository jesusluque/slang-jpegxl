# sjxl — JPEG XL on Slang

A JPEG XL encoder and decoder whose every numeric step is a Slang compute
kernel, run through [slang-rhi](https://github.com/shader-slang/slang-rhi):
Metal on Apple silicon, CUDA or Vulkan on Linux. The CPU parses and writes
headers, the table of contents and files; transforms, quantisation, tokens,
histograms, code tables, bit emission and symbol decoding run on the GPU,
entropy coding included.

Status: the decoder reproduces djxl's output on everything `cjxl` emits on
the test images, lossy and lossless: VarDCT with every transform size, Modular
with RCT, palettes and Squeeze, chroma-from-luma, adaptive quantisation,
Gaborish and EPF, noise, upsampling, patches, progressive passes and DC frames,
permuted TOCs, extra channels with alpha, 16-bit output, ICC-tagged images,
JPEG recompression and animation with blending. Pixels match within one 8-bit
step (lossless bit-exact); the conformance tests in `tests/decoder` are the
record. The encoder is VarDCT with transforms from 8×8 to 64×64 chosen by
libjxl's entropy model, libjxl's adaptive quant field, chroma-from-luma,
Gaborish + EPF, the weighted predictor and fixed trees for the DC, and ANS
with clustered contexts throughout: at the file size `cjxl -e 5` produces it
scores slightly above it on 4K photos (+0.1 to +0.6 SSIMULACRA2) and within
a point or two on 200-pixel images; alpha (coded losslessly as an extra
channel) and 16-bit input are accepted, `sjxl animate` writes an animation
(lossy or lossless) whose frames code only the rectangle that changed, `--progressive`/`--qprogressive` split the AC
coefficients over frequency or precision passes and `--progressive-dc` puts a
1/8-size DC frame first. `--lossless` writes a Modular frame
(palettes or per-group RCTs, a learned MA tree over the weighted and gradient
predictors, run-length LZ77 and per-cluster hybrid-uint configurations, ANS
with libjxl's histogram precision search; RGB or RGBA, 8 or 16 bits) within
0.5 % of `cjxl -d 0 -e 5` on the test photos, smaller on most of them and
4 % smaller on a 4K photo. A JPEG input is transcoded losslessly (Huffman
decoding on the GPU too), 2 to 10 % smaller than `cjxl` makes it. M3
continues with DC coding, the strategy and quant-field heuristics, and better
lossless trees.
`docs/decisions.md`
holds the design, what each milestone measured and the roadmap; `CLAUDE.md`
the short version a build needs.

## Build

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
ctest --preset macos-arm64-debug
build/macos-arm64-debug/bin/sjxl info
```

Needs CMake ≥ 3.24, Ninja, a Slang release unpacked at `~/tools/slang`
(or `SLANG_ROOT`), and for the conformance tests the Homebrew `cjxl`/`djxl`.

## License

BSD-3-Clause (see LICENSE). A derivative work of libjxl; JPEG XL is a trademark
of its owners, and this project is not affiliated with the JPEG committee.
