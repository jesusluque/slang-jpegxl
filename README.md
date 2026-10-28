# sjxl — JPEG XL on Slang

A JPEG XL encoder and decoder whose every numeric step is a Slang compute
kernel, run through [slang-rhi](https://github.com/shader-slang/slang-rhi):
Metal on Apple silicon, CUDA or Vulkan on Linux. The CPU parses and writes
headers, the table of contents and files; transforms, quantisation, tokens,
histograms, code tables, bit emission and symbol decoding run on the GPU,
entropy coding included.

Status: early. `docs/decisions.md` holds the design, what each milestone
measured and the roadmap; `CLAUDE.md` the short version a build needs.

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
