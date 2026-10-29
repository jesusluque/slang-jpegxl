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
record. The encoder is VarDCT with DCT8, libjxl's adaptive quant field,
chroma-from-luma, Gaborish + EPF, and ANS with clustered coefficient
contexts for the AC groups: at the file size `cjxl -e 5` produces it scores
within about one SSIMULACRA2 point of it. `--lossless` writes a Modular frame
(YCoCg, Gradient predictor, fixed tree, ANS) 8–13 % larger than `cjxl -d 0`.
M3 continues with alpha and 16-bit input, larger transforms and DC coding.
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
