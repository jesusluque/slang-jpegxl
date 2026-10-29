// Copyright (c) 2026 jesus luque.
//
// Prefix codes and the section headers that carry them, built on the device:
// one thread per histogram, with the tree arrays in groupshared memory. A
// straight port of libjxl's enc_huffman_tree.cc / enc_huffman.cc as cujpegxl
// adapted them for a single thread (serialize_impl.h): length-limited (15 bit)
// Huffman codes, Brotli-style code-length coding, the hybrid-uint config, and
// the modular GroupHeader + MA tree + histogram description of the DC and
// AcMetadata streams.
//
// Regions: histograms and codes share one indexing (256 entries each): the AC
// clusters are regions 0..31, DC group g has its DC code at 32 + 2g and its
// AcMetadata code at 33 + 2g. Blobs (serialised headers) are 8192-word
// regions: DcGlobal 0, AcGlobal 1 (the clustered AC histogram set with its
// coded context map), DC group g "pre" (extra_precision + DC modular header)
// at 2 + 2g and "mid" (count + AcMetadata modular header) at 3 + 2g.
module enc_codes;

#include "sjxl/common/hybrid_uint.slangh"
#include "sjxl/common/bitio.slangh"
#include "sjxl/common/ac_context.slangh"
#include "sjxl/enc/ans.slangh"
#include "sjxl/enc/modular.slangh"

static const uint kMaxAlphabet = 256;
static const uint kCodeLengthCodes = 18;
static const uint kNumTreeContexts = 6;
static const uint kPrefixMaxBits = 15;
static const uint kCodeStride = 256;
static const uint kBlobStrideWords = 8192;
static const uint kAcClusters = 32;        // clusters per histogram set, and code regions per set
static const uint kSetContexts = 64;       // contexts of the DC group sets (tree leaves)
/// Histogram sets: 0 the AC contexts, 1 + 2g the DC group g's DC stream, 2 + 2g its AcMetadata.
/// Histogram sets: the AC passes first (numAcSets of numAcContexts each), then
/// the DC groups' DC and AcMetadata sets and the alpha set (kSetContexts each).
uint setCtxBase(uint set) {
    return set < params.numAcSets ? set * params.numAcContexts : params.numAcSets * params.numAcContexts + (set - params.numAcSets) * kSetContexts;
}
uint setRegionBase(uint set) { return set * kAcClusters; }
uint mapRegion() { return (params.numAcSets + 2 * params.numDcGroups + params.numEcSets) * kAcClusters; }   // the context-map stream's own code
uint ecSet() { return params.numAcSets + 2 * params.numDcGroups; }
uint dcSetOf(uint g) { return params.numAcSets + 2 * g; }
uint metaSetOf(uint g) { return params.numAcSets + 1 + 2 * g; }
static const int kPredictorZero = 0;
static const int kPredictorGradient = 5;
static const int kEpfSharpness = 4;   // what libjxl writes at every speed up to wombat

struct CodeParams {
    uint numDcGroups;
    uint numGroups;
    uint blocksX;
    uint blocksY;
    // Modular (lossless) frames:
    uint width;
    uint height;
    uint groupDim;
    uint numTreeNodes;
    uint numLeaves;
    uint numChannels;
    uint bitDepth;
    uint dcTreeNodes;
    uint dcLeaves;
    uint metaTreeNodes;
    uint metaLeaves;
    uint numAcContexts;   // the AC set's contexts (sets after it start at this index, kSetContexts each)
    uint numEcSets;       // 1 when a VarDCT frame carries an alpha channel (set 1 + 2 numDcGroups)
    uint ecInGlobal;      // 1: the alpha channel fits a group and its stream goes into DcGlobal
    uint ecLogAlpha;      // the alpha stream's ANS alphabet (6 at 8 bits, 7 at 16)
    uint numAcSets;       // AC passes (one histogram set each); 1 for a lossless frame's layout
    uint rct;             // modular frames: 1 when the frame has an RCT (groupRct; in the global header when the image is one stream)
    uint rctBeginC;       // the RCT's first channel
    uint numTransforms;   // modular frames: global transforms in `transforms`
    uint totalSamples;    // modular frames: samples over every channel
    uint lz77Modular;     // modular frames: the set carries run-length tokens (an extra distance context)
    uint modLogAlpha;     // modular frames: the set's log_alpha_size
    uint histogramTries;  // ANS histogram precisions tried (7: libjxl's approximate strategy); runtime so no unrolling
    uint prepRegions;     // prepareHistograms: code regions to prepare (sets x kAcClusters)
    uint prepModular;     // 1: one set (the modular frame's) at modLogAlpha
    uint jpeg;            // JPEG recompression: custom DC quant, zero DC chroma-from-luma, the raw DCT table
    uint qtRegion;        // the code region of the quant table stream
    uint qtCount;         // samples of the raw table (192); runtime so the writer's loops are not unrolled
    uint numQuantKinds;   // 17
    uint qtBlobRegion;    // blob region of the quant table stream (between AcGlobal's two parts)
    uint acPostRegion;    // blob region of AcGlobal after the dequant matrices
    float dcQuantJpegX;   // DequantMatrices::DecodeDC values (q0 / (8 x 255)) per channel X, Y, B
    float dcQuantJpegY;
    float dcQuantJpegB;
    uint dcQuantCustom;   // modular frames: 1 writes dc_quant (DequantMatrices::DecodeDC) as the three F16s below
    uint useDcFrame;      // VarDCT frames: 1 leaves the DC groups' DC stream out (kUseDcFrame)
    float dcQuantX;       // the custom dc_quant per channel (X, Y, B)
    float dcQuantY;
    float dcQuantB;
    uint numAcCtxsUsed;   // contexts of each AC set actually coded (block contexts x 495; the stride stays numAcContexts)
};

/// dc_quant as the decoder will hold it: F16 of 128 x the value, back to float.
float dcQuantF16(float v) { return f16tof32(f32tof16Trunc(v * 128.0f)) / 128.0f; }

StructuredBuffer<uint4>    dcGroups;   // bx0, by0, w, h in blocks
StructuredBuffer<uint>     acs;        // per block: strategy << 1 | first
StructuredBuffer<uint>     hist;
StructuredBuffer<uint>     quantizer;  // [0] global_scale, [1] quant_dc
StructuredBuffer<uint>     ctxMap;     // AC context -> cluster
StructuredBuffer<uint>     groupRct;   // modular frames: the RCT type of each group (rct != 0)
StructuredBuffer<uint4>    chanTable;  // modular frames: per channel w, h, offset, grouped
StructuredBuffer<uint4>    transforms; // modular frames: the global transforms (palettes): id, begin_c, num_c, nb_colors
StructuredBuffer<uint>     streamTokens;  // modular frames: the compact per-stream token lists
StructuredBuffer<uint>     streamStart;   // per stream (groups, then the global one)
StructuredBuffer<uint>     streamLen;
StructuredBuffer<uint>     clusterCfg;    // per code region: hybrid-uint config split | msb << 8 | lsb << 16
StructuredBuffer<int>      rawQuant;      // JPEG recompression: the DCT8 table, 3 x 64 in libjxl's layout (kQuantModeRAW)
StructuredBuffer<uint>     qtTree;        // JPEG recompression: the quant table stream's tree (one leaf, predictor Zero)
StructuredBuffer<uint>     clusterInfo;   // [0] number of AC clusters
StructuredBuffer<uint>     bctxMap;       // JPEG recompression: jpegBlockCtx's BlockCtxMap ([0] DC contexts, [1] block contexts, [2..9) thresholds, [16..) the map)
RWStructuredBuffer<uint>   ctxMapMtf;  // scratch: the context map after the move-to-front transform
RWStructuredBuffer<uint>   ansInfo;    // ANS symbol info per code region (the context map's code is region kAcClusters)
RWStructuredBuffer<uint>   ansReverse; // ANS reverse maps per code region
RWStructuredBuffer<uint>   histPrep;      // prepareHistograms: per code region kPrepStride words: normalised counts, length, shift, flat
StructuredBuffer<int>      planes;     // modular: 3 * width * height channel values
StructuredBuffer<uint>     treeNodes;  // modular: the fixed tree, kModTreeWords per node
StructuredBuffer<uint>     modTokens;  // modular: ctx << 18 | value per channel sample
StructuredBuffer<uint>     dcTree;     // VarDCT DC groups: the kWPFixedDC tree
StructuredBuffer<uint>     metaTree;   // VarDCT DC groups: the kACMeta tree
RWStructuredBuffer<uint>   codes;      // depth << 16 | bits
RWStructuredBuffer<uint>   blobs;
RWStructuredBuffer<uint>   blobBits;
ConstantBuffer<CodeParams> params;

groupshared uint gHist[kMaxAlphabet];
groupshared uint gDepth[kMaxAlphabet];
groupshared uint gBits[kMaxAlphabet];
groupshared uint gDataDepth[kMaxAlphabet];   // the data code, kept while its description is built
groupshared uint gDataBits[kMaxAlphabet];
groupshared uint gTreeCount[2 * kMaxAlphabet + 4];
groupshared int  gTreeLeft[2 * kMaxAlphabet + 4];
groupshared int  gTreeRight[2 * kMaxAlphabet + 4];
groupshared uint gStackNode[2 * kMaxAlphabet + 4];
groupshared uint gStackLevel[2 * kMaxAlphabet + 4];
groupshared uint gHuffTree[2 * kMaxAlphabet];
groupshared uint gHuffExtra[2 * kMaxAlphabet];

// --- Huffman tree construction (enc_huffman_tree.cc) --------------------------

void setDepth(uint root) {
    uint sp = 0;
    gStackNode[sp] = root;
    gStackLevel[sp] = 0;
    ++sp;
    while (sp != 0) {
        --sp;
        const uint node = gStackNode[sp];
        const uint level = gStackLevel[sp];
        if (gTreeLeft[node] >= 0) {
            gStackNode[sp] = uint(gTreeLeft[node]);
            gStackLevel[sp] = level + 1;
            ++sp;
            gStackNode[sp] = uint(gTreeRight[node]);
            gStackLevel[sp] = level + 1;
            ++sp;
        } else {
            gDepth[uint(gTreeRight[node])] = level;
        }
    }
}

void setTreeNode(uint at, uint count, int left, int right) {
    gTreeCount[at] = count;
    gTreeLeft[at] = left;
    gTreeRight[at] = right;
}

/// gDepth[i] for gHist[0..length), limited to treeLimit bits. gDepth must be
/// zero on entry for the symbols that are absent.
void createHuffmanTree(uint length, uint treeLimit) {
    for (uint countLimit = 1;; countLimit *= 2) {
        uint size = 0;
        for (int i = int(length) - 1; i >= 0; --i) {
            if (gHist[i] != 0) {
                setTreeNode(size, max(gHist[i], countLimit - 1), -1, i);
                ++size;
            }
        }
        const uint n = size;
        if (n == 1) {
            gDepth[uint(gTreeRight[0])] = 1;
            break;
        }
        // Stable insertion sort, ascending by count.
        for (uint i = 1; i < n; ++i) {
            const uint kc = gTreeCount[i];
            const int kl = gTreeLeft[i];
            const int kr = gTreeRight[i];
            uint j = i;
            while (j > 0 && gTreeCount[j - 1] > kc) {
                gTreeCount[j] = gTreeCount[j - 1];
                gTreeLeft[j] = gTreeLeft[j - 1];
                gTreeRight[j] = gTreeRight[j - 1];
                --j;
            }
            gTreeCount[j] = kc;
            gTreeLeft[j] = kl;
            gTreeRight[j] = kr;
        }
        setTreeNode(size++, 0xFFFFFFFFu, -1, -1);
        setTreeNode(size++, 0xFFFFFFFFu, -1, -1);
        uint i = 0;
        uint j = n + 1;
        for (uint k = n - 1; k != 0; --k) {
            uint left, right;
            if (gTreeCount[i] <= gTreeCount[j]) { left = i; ++i; } else { left = j; ++j; }
            if (gTreeCount[i] <= gTreeCount[j]) { right = i; ++i; } else { right = j; ++j; }
            const uint jEnd = size - 1;
            setTreeNode(jEnd, gTreeCount[left] + gTreeCount[right], int(left), int(right));
            setTreeNode(size++, 0xFFFFFFFFu, -1, -1);
        }
        setDepth(2 * n - 1);
        uint maxDepth = 0;
        for (uint d = 0; d < length; ++d) {
            maxDepth = max(maxDepth, gDepth[d]);
        }
        if (maxDepth <= treeLimit) {
            break;
        }
    }
}

uint reverseBits(uint numBits, uint bits) {
    const uint lut[16] = {0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
    uint retval = lut[bits & 0xf];
    for (uint i = 4; i < numBits; i += 4) {
        retval <<= 4;
        bits >>= 4;
        retval |= lut[bits & 0xf];
    }
    retval >>= ((0u - numBits) & 0x3);
    return retval;
}

/// gBits from gDepth: canonical codes, bit-reversed for LSB-first writing.
void convertBitDepthsToSymbols(uint len) {
    uint blCount[16];
    uint nextCode[16];
    for (uint i = 0; i < 16; ++i) blCount[i] = 0;
    for (uint i = 0; i < len; ++i) ++blCount[gDepth[i]];
    blCount[0] = 0;
    uint code = 0;
    nextCode[0] = 0;
    for (uint i = 1; i < 16; ++i) {
        code = (code + blCount[i - 1]) << 1;
        nextCode[i] = code;
    }
    for (uint i = 0; i < len; ++i) {
        if (gDepth[i] != 0) {
            gBits[i] = reverseBits(gDepth[i], nextCode[gDepth[i]]++);
        }
    }
}

// --- storing a code (enc_huffman.cc) ------------------------------------------

/// Code-length symbols of the data code in gDataDepth, one per symbol up to the
/// last non-zero depth (no run-length codes: valid, a few bits longer).
uint writeHuffmanTree(uint length) {
    uint newLength = length;
    while (newLength > 0 && gDataDepth[newLength - 1] == 0) {
        --newLength;
    }
    uint size = 0;
    for (uint i = 0; i < newLength; ++i) {
        gHuffTree[size] = gDataDepth[i];
        gHuffExtra[size] = 0;
        ++size;
    }
    return size;
}

void storeHuffmanTreeOfHuffmanTree(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint numCodes) {
    const uint storageOrder[kCodeLengthCodes] = {1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const uint symbols[6] = {0, 7, 3, 2, 1, 15};
    const uint bitLengths[6] = {2, 4, 3, 2, 2, 4};
    uint codesToStore = kCodeLengthCodes;
    if (numCodes > 1) {
        for (; codesToStore > 0; --codesToStore) {
            if (gDepth[storageOrder[codesToStore - 1]] != 0) {
                break;
            }
        }
    }
    uint skipSome = 0;
    if (gDepth[storageOrder[0]] == 0 && gDepth[storageOrder[1]] == 0) {
        skipSome = 2;
        if (gDepth[storageOrder[2]] == 0) {
            skipSome = 3;
        }
    }
    serialWrite(buf, w, 2, skipSome);
    for (uint i = skipSome; i < codesToStore; ++i) {
        const uint l = gDepth[storageOrder[i]];
        serialWrite(buf, w, bitLengths[l], symbols[l]);
    }
}

/// The full form: code-length code, then every symbol's length.
void storeHuffmanTree(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint length) {
    const uint treeSize = writeHuffmanTree(length);
    for (uint i = 0; i < kCodeLengthCodes; ++i) {
        gHist[i] = 0;
        gDepth[i] = 0;
        gBits[i] = 0;
    }
    for (uint i = 0; i < treeSize; ++i) {
        ++gHist[gHuffTree[i]];
    }
    uint numCodes = 0;
    uint code = 0;
    for (uint i = 0; i < kCodeLengthCodes; ++i) {
        if (gHist[i] != 0) {
            if (numCodes == 0) {
                code = i;
                numCodes = 1;
            } else if (numCodes == 1) {
                numCodes = 2;
                break;
            }
        }
    }
    createHuffmanTree(kCodeLengthCodes, 5);
    convertBitDepthsToSymbols(kCodeLengthCodes);
    storeHuffmanTreeOfHuffmanTree(buf, w, numCodes);
    if (numCodes == 1) {
        gDepth[code] = 0;
    }
    for (uint i = 0; i < treeSize; ++i) {
        const uint ix = gHuffTree[i];
        serialWrite(buf, w, gDepth[ix], gBits[ix]);
        if (ix == 16) {
            serialWrite(buf, w, 2, gHuffExtra[i]);
        } else if (ix == 17) {
            serialWrite(buf, w, 3, gHuffExtra[i]);
        }
    }
}

struct Scan {
    uint count;      // used symbols, capped at 5
    uint s4[4];      // the first four
    uint maxBits;    // bits to name a symbol of this alphabet
};

Scan scanHistogram(uint length) {
    Scan s;
    s.count = 0;
    s.s4[0] = s.s4[1] = s.s4[2] = s.s4[3] = 0;
    for (uint i = 0; i < length; ++i) {
        if (gHist[i] != 0) {
            if (s.count < 4) {
                s.s4[s.count] = i;
            } else if (s.count > 4) {
                break;
            }
            ++s.count;
        }
    }
    uint counter = length - 1;
    s.maxBits = 0;
    while (counter != 0) {
        counter >>= 1;
        ++s.maxBits;
    }
    return s;
}

void storeSimpleHuffmanTree(RWStructuredBuffer<uint> buf, inout SerialWriter w, Scan s) {
    serialWrite(buf, w, 2, 1);
    serialWrite(buf, w, 2, s.count - 1);
    uint symbols[4] = {s.s4[0], s.s4[1], s.s4[2], s.s4[3]};
    for (uint i = 0; i < s.count; ++i) {
        for (uint j = i + 1; j < s.count; ++j) {
            if (gDataDepth[symbols[j]] < gDataDepth[symbols[i]]) {
                const uint t = symbols[j];
                symbols[j] = symbols[i];
                symbols[i] = t;
            }
        }
    }
    for (uint i = 0; i < s.count; ++i) {
        serialWrite(buf, w, s.maxBits, symbols[i]);
    }
    if (s.count == 4) {
        serialWrite(buf, w, 1, gDataDepth[symbols[0]] == 1 ? 1 : 0);
    }
}

/// Builds the code of gHist[0..length) into gDataDepth/gDataBits (and, when
/// codeBase != 0xFFFFFFFF, the codes buffer region at codeBase) and stores its
/// description.
void buildAndStoreCode(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint length, uint codeBase) {
    for (uint i = 0; i < length; ++i) {
        gDepth[i] = 0;
        gBits[i] = 0;
    }
    Scan s = scanHistogram(length);
    if (s.count > 1) {
        createHuffmanTree(length, kPrefixMaxBits);
        convertBitDepthsToSymbols(length);
    }
    for (uint i = 0; i < length; ++i) {
        gDataDepth[i] = gDepth[i];
        gDataBits[i] = gBits[i];
        if (codeBase != 0xFFFFFFFFu) {
            codes[codeBase + i] = (gDepth[i] << 16) | gBits[i];
        }
    }
    if (s.count <= 1) {
        serialWrite(buf, w, 4, 1);
        serialWrite(buf, w, s.maxBits, s.s4[0]);
    } else if (s.count <= 4) {
        storeSimpleHuffmanTree(buf, w, s);
    } else {
        storeHuffmanTree(buf, w, length);
    }
}

// --- histogram descriptions (dec_ans.cc DecodeHistograms, prefix path) ---------

void encodeUintConfigAt(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint logAlphaSize, uint split, uint msb, uint lsb) {
    serialWrite(buf, w, ceilLog2Nonzero(logAlphaSize + 1), split);
    if (split == logAlphaSize) {
        return;
    }
    serialWrite(buf, w, ceilLog2Nonzero(split + 1), msb);
    serialWrite(buf, w, ceilLog2Nonzero(split - msb + 1), lsb);
}

void encodeUintConfig(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint logAlphaSize) {
    encodeUintConfigAt(buf, w, logAlphaSize, kHybridSplitExponent, kHybridMsbInToken, kHybridLsbInToken);
}

void storeVarLenUint16(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint n) {
    if (n == 0) {
        serialWrite(buf, w, 1, 0);
        return;
    }
    serialWrite(buf, w, 1, 1);
    const uint nbits = firstbithigh(n);
    serialWrite(buf, w, 4, nbits);
    serialWrite(buf, w, nbits, n - (1u << nbits));
}

/// lz77 off, the trivial context map if there are several contexts, prefix
/// codes, one hybrid-uint config, the alphabet size.
void writePrefixHistogramHead(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint length, uint numContexts) {
    serialWrite(buf, w, 1, 0);   // lz77.enabled
    if (numContexts > 1) {
        serialWrite(buf, w, 1, 1);   // is_simple
        serialWrite(buf, w, 2, 0);   // bits_per_entry = 0: every context is histogram 0
    }
    serialWrite(buf, w, 1, 1);       // use_prefix_code
    encodeUintConfig(buf, w, kPrefixMaxBits);
    storeVarLenUint16(buf, w, length - 1);
}

/// Head + the code of gHist[0..length) (zero-length code for one symbol).
void writePrefixHistograms(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint length, uint numContexts, uint codeBase) {
    writePrefixHistogramHead(buf, w, length, numContexts);
    if (length > 1) {
        buildAndStoreCode(buf, w, length, codeBase);
    } else {
        for (uint i = 0; i < kMaxAlphabet; ++i) {
            gDataDepth[i] = 0;
            gDataBits[i] = 0;
        }
        if (codeBase != 0xFFFFFFFFu) {
            codes[codeBase] = 0;
        }
    }
}

static const uint kAcLogAlpha = 6;    // 64-symbol ANS alphabet for the AC clusters
static const uint kMapLogAlpha = 5;   // the context map's values are below 32

/// The context map as a one-context ANS stream after the move-to-front
/// transform (EncodeContextMap with use_mtf): its histogram set, then the
/// tokens coded backwards and placed from the end as WriteTokens does.
[noinline]
void writeContextMapStream(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint numContexts, uint numClusters, uint ctxBase) {
    uint mtf[kAcClusters];
    for (uint i = 0; i < kAcClusters; ++i) mtf[i] = i;
    for (uint i = 0; i < kMaxAlphabet; ++i) gAnsCounts[i] = 0;
    uint length = 1;
    const uint kMapRegion = mapRegion();
    for (uint ctx = 0; ctx < numContexts; ++ctx) {
        const uint v = ctxMap[ctxBase + ctx];
        uint index = 0;
        while (mtf[index] != v) ++index;
        for (uint j = index; j > 0; --j) mtf[j] = mtf[j - 1];
        mtf[0] = v;
        ctxMapMtf[ctxBase + ctx] = index;
        uint tok, nbits, raw;
        hybridEncode(index, tok, nbits, raw);
        ++gAnsCounts[tok];
        length = max(length, tok + 1);
    }
    // The nested histogram set: LZ77 off, one context, ANS, one config, the histogram.
    serialWrite(buf, w, 1, 0);
    serialWrite(buf, w, 1, 0);                     // use_prefix_code = 0
    serialWrite(buf, w, 2, kMapLogAlpha - 5);
    encodeUintConfig(buf, w, kMapLogAlpha);
    ansWriteBestHistogram(buf, w, length, params.histogramTries);
    ansBuildTables(ansInfo, ansReverse, kMapRegion, length, kMapLogAlpha);
    // Two passes over the tokens backwards: the size, then the placement.
    uint total = 32;
    for (uint pass = 1; pass <= 2; ++pass) {
        uint state = kAnsSignature << 16;
        uint pos = w.pos + total;
        for (int ctx = int(numContexts) - 1; ctx >= 0; --ctx) {
            uint tok, nbits, raw;
            hybridEncode(ctxMapMtf[ctxBase + uint(ctx)], tok, nbits, raw);
            const uint e = ansInfo[kMapRegion * kAnsInfoStride + tok];
            if (pass == 2) {
                pos -= nbits;
                serialWriteAt(buf, w.base, pos, nbits, raw);
            }
            uint ansBits, ansN;
            ansPut(state, e & 0xFFFFu, e >> 16, ansReverse, kMapRegion * kAnsReverseStride, ansN, ansBits);
            if (pass == 2) {
                pos -= ansN;
                serialWriteAt(buf, w.base, pos, ansN, ansBits);
            } else {
                total += nbits + ansN;
            }
        }
        if (pass == 2) {
            serialWriteAt(buf, w.base, w.pos, 32, state);
        }
    }
    w.pos += total;
}

/// A clustered histogram set (DecodeHistograms): LZ77 off, the context map
/// (0 bits per entry for one cluster, plain bits for a few clusters over few
/// contexts, else coded through writeContextMapStream), ANS with a 1 <<
/// logAlpha alphabet, then every cluster's hybrid-uint config and its
/// normalised histogram, whose alias table is inverted for the section kernels.
/// `lz77`: the set's streams carry run-length tokens: LZ77Params min_symbol 224,
/// min_length 3, the length config (4, 2, 0) at alphabet 8; numContexts then
/// includes the distance context.
[noinline]
void writeHistogramSetLz77(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint numContexts, uint numClusters, uint logAlpha, uint set, bool lz77) {
    const uint ctxBase = setCtxBase(set);
    const uint regionBase = setRegionBase(set);
    serialWrite(buf, w, 1, lz77 ? 1 : 0);   // lz77.enabled
    if (lz77) {
        serialWrite(buf, w, 2, 0);   // min_symbol: Val(224)
        serialWrite(buf, w, 2, 0);   // min_length: Val(3)
        encodeUintConfigAt(buf, w, 8, 4, 2, 0);
    }
    if (numContexts > 1) {
        const uint entryBits = ceilLog2Nonzero(numClusters);
        if (numClusters == 1) {
            serialWrite(buf, w, 1, 1);   // is_simple
            serialWrite(buf, w, 2, 0);   // bits_per_entry = 0
        } else if (entryBits <= 3 && numContexts <= 64) {
            serialWrite(buf, w, 1, 1);   // is_simple
            serialWrite(buf, w, 2, entryBits);
            for (uint ctx = 0; ctx < numContexts; ++ctx) serialWrite(buf, w, entryBits, ctxMap[ctxBase + ctx]);
        } else {
            serialWrite(buf, w, 1, 0);   // is_simple
            serialWrite(buf, w, 1, 1);   // use_mtf
            writeContextMapStream(buf, w, numContexts, numClusters, ctxBase);
        }
    }
    serialWrite(buf, w, 1, 0);       // use_prefix_code = 0
    serialWrite(buf, w, 2, logAlpha - 5);
    for (uint c = 0; c < numClusters; ++c) {
        const uint cfg = clusterCfg[regionBase + c];
        encodeUintConfigAt(buf, w, logAlpha, cfg & 0xFFu, (cfg >> 8) & 0xFFu, (cfg >> 16) & 0xFFu);
    }
    for (uint c = 0; c < numClusters; ++c) {
        // The histogram as prepareHistograms normalised it (its tables are built there too).
        const uint prep = (regionBase + c) * kPrepStride;
        for (uint i = 0; i < kMaxAlphabet; ++i) gAnsCounts[i] = histPrep[prep + i];
        ansWriteHistogramAt(buf, w, histPrep[prep + 256], histPrep[prep + 257], histPrep[prep + 258] != 0, false);
    }
}

static const uint kPrepStride = 260;

/// The ANS alphabet of a histogram set: 64 symbols for the AC and AcMetadata sets,
/// 128 for the DC sets, the frame's for the extra channel and modular sets.
uint setLogAlpha(uint set) {
    if (params.prepModular != 0) return params.modLogAlpha;
    if (set < params.numAcSets) return kAcLogAlpha;
    if (set >= ecSet()) return params.ecLogAlpha;
    return ((set - params.numAcSets) & 1) == 0 ? 7 : 6;
}

/// One thread group per code region: the cluster's histogram normalised in its
/// cheapest form (ansWriteBestHistogram's choice) into histPrep, and its alias and
/// reverse tables. What the blob writers used to do in sequence for every cluster.
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareHistograms(uint3 gid: SV_GroupID) {
    const uint r = gid.x;
    if (r >= params.prepRegions) return;
    const uint set = r / kAcClusters, c = r % kAcClusters;
    if (c >= clusterInfo[set]) return;
    const uint logAlpha = setLogAlpha(set);
    const uint length = loadHistogram(r * kCodeStride);
    uint numSymbols = 0;
    for (uint s = 0; s < 256; ++s) {
        gAnsRaw[s] = s < length ? gHist[s] : 0;
        if (gAnsRaw[s] != 0) ++numSymbols;
    }
    uint shift;
    bool flat;
    ansBestHistogramChoice(blobs, length, numSymbols, params.histogramTries, shift, flat);
    if (numSymbols <= 2) {
        shift = kAnsLogTabSize - 1;
        flat = false;
        ansNormalizeShift(length, shift);
    } else if (flat) {
        const uint base = kAnsTabSize / length, rem = kAnsTabSize % length;
        for (uint s = 0; s < 256; ++s) gAnsCounts[s] = s < length ? base + (s < rem ? 1 : 0) : 0;
        shift = 0;
    } else {
        ansNormalizeShift(length, shift);
    }
    const uint prep = r * kPrepStride;
    for (uint i = 0; i < 256; ++i) histPrep[prep + i] = i < length ? gAnsCounts[i] : 0;
    histPrep[prep + 256] = length;
    histPrep[prep + 257] = shift;
    histPrep[prep + 258] = flat ? 1u : 0u;
    ansBuildTables(ansInfo, ansReverse, r, length, logAlpha);
}

void writeHistogramSet(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint numContexts, uint numClusters, uint logAlpha, uint set) {
    writeHistogramSetLz77(buf, w, numContexts, numClusters, logAlpha, set, false);
}

/// Last used symbol + 1 of the histogram region at histBase (at least 1), and
/// the histogram copied into gHist.
uint loadHistogram(uint histBase) {
    uint last = 0;
    for (uint s = 0; s < kMaxAlphabet; ++s) {
        gHist[s] = hist[histBase + s];
        if (gHist[s] != 0) {
            last = s + 1;
        }
    }
    return last == 0 ? 1 : last;
}

// --- modular headers (GroupHeader + MA tree) ------------------------------------

/// A leaf: property 0, `predictor`, offset 0, multiplier 1 (log 0, bits 0).
/// A leaf: predictor, offset, multiplier 1 (mul_log 0, mul_bits 0).
void appendLeaf(inout uint tokens[12], inout uint count, int predictor, int offset) {
    tokens[count++] = 0;
    tokens[count++] = uint(predictor);
    tokens[count++] = packSigned(offset);
    tokens[count++] = 0;
    tokens[count++] = 0;
}

/// GroupHeader (no global tree, default weighted header, no transforms) then
/// the tree: its own prefix histogram over 6 contexts, then its tokens.
void writeModularTree(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint tokens[12], uint count) {
    serialWrite(buf, w, 1, 0);   // use_global_tree
    serialWrite(buf, w, 1, 1);   // weighted::Header all_default
    serialWrite(buf, w, 2, 0);   // num_transforms = 0
    for (uint i = 0; i < kMaxAlphabet; ++i) gHist[i] = 0;
    uint treeLen = 1;
    for (uint i = 0; i < count; ++i) {
        ++gHist[tokens[i]];
        treeLen = max(treeLen, tokens[i] + 1);
    }
    writePrefixHistograms(buf, w, treeLen, kNumTreeContexts, 0xFFFFFFFFu);
    for (uint i = 0; i < count; ++i) {
        serialWrite(buf, w, gDataDepth[tokens[i]], gDataBits[tokens[i]]);
    }
}

/// GroupHeader of a stream with its own tree: use_global_tree 0, default weighted header, no transforms.
void writeLocalGroupHeader(RWStructuredBuffer<uint> buf, inout SerialWriter w) {
    serialWrite(buf, w, 1, 0);
    serialWrite(buf, w, 1, 1);
    serialWrite(buf, w, 2, 0);
}

// The two DC group blobs are separate kernels: each inlines one ANS histogram
// writer (two in one kernel made Metal's backend compiler hang intermittently).

/// "pre" of DC group g: extra_precision, the DC stream's header: the kWPFixedDC
/// tree and its clustered ANS histograms (the group's DC set). Nothing with a DC frame.
[shader("compute")]
[numthreads(1, 1, 1)]
void buildDcGroupPre(uint3 gid: SV_GroupID) {
    const uint g = gid.x;
    if (g >= params.numDcGroups) {
        return;
    }
    SerialWriter pre;
    pre.base = (2 + 2 * g) * kBlobStrideWords;
    pre.pos = 0;
    if (params.useDcFrame == 0) {
        serialWrite(blobs, pre, 2, 0);
        writeLocalGroupHeader(blobs, pre);
        writeTreeFromNodes(blobs, pre, dcTree, params.dcTreeNodes);
        writeHistogramSet(blobs, pre, params.dcLeaves, clusterInfo[dcSetOf(g)], 7, dcSetOf(g));
    }
    blobBits[2 + 2 * g] = pre.pos;
}

/// "mid" of DC group g: count - 1 (one AcMetadata entry per varblock), the
/// AcMetadata stream's header: the kACMeta tree and its histograms (set 2 + 2g).
[shader("compute")]
[numthreads(1, 1, 1)]
void buildDcGroupMid(uint3 gid: SV_GroupID) {
    const uint g = gid.x;
    if (g >= params.numDcGroups) {
        return;
    }
    const uint4 rect = dcGroups[g];
    SerialWriter mid;
    mid.base = (3 + 2 * g) * kBlobStrideWords;
    mid.pos = 0;
    const uint totalBlocks = rect.z * rect.w;
    uint numVarblocks = 0;
    for (uint by = 0; by < rect.w; ++by) {
        for (uint bx = 0; bx < rect.z; ++bx) {
            numVarblocks += acs[(rect.y + by) * params.blocksX + rect.x + bx] & 1;
        }
    }
    serialWrite(blobs, mid, ceilLog2Nonzero(totalBlocks), numVarblocks - 1);
    writeLocalGroupHeader(blobs, mid);
    writeTreeFromNodes(blobs, mid, metaTree, params.metaTreeNodes);
    writeHistogramSet(blobs, mid, params.metaLeaves, clusterInfo[metaSetOf(g)], 6, metaSetOf(g));
    blobBits[3 + 2 * g] = mid.pos;
}

// --- Modular (lossless) frames -------------------------------------------------------

/// The modular alphabet: 8-bit residual tokens stay below 64, 16-bit ones below 128.
uint modLogAlpha() { return params.bitDepth > 8 ? 7 : 6; }

/// The fixed tree's tokens as DecodeTree reads them, node by node: property + 1
/// and the split value for a split, 0, predictor, offset, mul_log, mul_bits for
/// a leaf. One prefix code over the six contexts (trivial context map).
/// The tokens of node n of the tree (TokenizeTree): a split is property + 1 and
/// the split value; a leaf is 0, predictor, offset, mul_log 0, mul_bits 0.
uint treeNodeTokens(StructuredBuffer<uint> tree, uint n, out uint t[5]) {
    const uint base = n * kModTreeWords;
    const int prop = int(tree[base + 0]);
    if (prop >= 0) {
        t[0] = uint(prop) + 1;
        t[1] = packSigned(int(tree[base + 1]));
        t[2] = t[3] = t[4] = 0;
        return 2;
    }
    t[0] = 0;
    t[1] = tree[base + 5];                   // predictor
    t[2] = packSigned(int(tree[base + 6]));  // offset
    t[3] = 0;                                // mul_log
    t[4] = 0;                                // mul_bits
    return 5;
}

[noinline]
void writeTreeFromNodes(RWStructuredBuffer<uint> buf, inout SerialWriter w, StructuredBuffer<uint> tree, uint numNodes) {
    for (uint i = 0; i < kMaxAlphabet; ++i) gHist[i] = 0;
    uint treeLen = 1;
    for (uint n = 0; n < numNodes; ++n) {
        uint t[5];
        const uint count = treeNodeTokens(tree, n, t);
        for (uint i = 0; i < count; ++i) {
            uint tok, nbits, raw;
            hybridEncode(t[i], tok, nbits, raw);
            ++gHist[tok];
            treeLen = max(treeLen, tok + 1);
        }
    }
    writePrefixHistograms(buf, w, treeLen, kNumTreeContexts, 0xFFFFFFFFu);
    for (uint n = 0; n < numNodes; ++n) {
        uint t[5];
        const uint count = treeNodeTokens(tree, n, t);
        for (uint i = 0; i < count; ++i) {
            uint tok, nbits, raw;
            hybridEncode(t[i], tok, nbits, raw);
            serialWrite(buf, w, gDataDepth[tok], gDataBits[tok]);
            serialWrite(buf, w, nbits, raw);
        }
    }
}

/// The whole image's three channels as one ANS stream inside the blob (the
/// global modular stream of an image no larger than a group): two passes, the
/// size then the placement from the end.
/// The global stream's tokens (its compact list is the last stream, index
/// numGroups) as one ANS stream inside the blob: two passes, the size then the
/// placement from the end. Length tokens take kMinSymbol + their (4, 2, 0)
/// token; the others their cluster's configuration.
static const uint kLengthFlag = 0x80000000u;
static const uint kLz77MinSymbol = 224;

void writeModularImageStream(RWStructuredBuffer<uint> buf, inout SerialWriter w, uint set) {
    const uint ctxBase = setCtxBase(set);
    const uint regionBase = setRegionBase(set);
    const uint stream = params.numGroups;
    const uint base = streamStart[stream];
    const uint n = streamLen[stream];
    uint total = 32;
    for (uint pass = 1; pass <= 2; ++pass) {
        uint state = kAnsSignature << 16;
        uint pos = w.pos + total;
        for (int i = int(n) - 1; i >= 0; --i) {
            const uint t = streamTokens[base + uint(i)];
            const uint ctx = (t & ~kLengthFlag) >> 18, value = t & 0x3FFFFu;
            const uint cluster = regionBase + ctxMap[ctxBase + ctx];
            uint tok, nbits, raw;
            if ((t & kLengthFlag) != 0) {
                hybridEncode(value, tok, nbits, raw);
                tok += kLz77MinSymbol;
            } else {
                const uint cfg = clusterCfg[cluster];
                hybridEncodeCfg(value, cfg & 0xFFu, (cfg >> 8) & 0xFFu, (cfg >> 16) & 0xFFu, tok, nbits, raw);
            }
            const uint e = ansInfo[cluster * kAnsInfoStride + tok];
            if (pass == 2) {
                pos -= nbits;
                serialWriteAt(buf, w.base, pos, nbits, raw);
            }
            uint ansBits, ansN;
            ansPut(state, e & 0xFFFFu, e >> 16, ansReverse, cluster * kAnsReverseStride, ansN, ansBits);
            if (pass == 2) {
                pos -= ansN;
                serialWriteAt(buf, w.base, pos, ansN, ansBits);
            } else {
                total += nbits + ansN;
            }
        }
        if (pass == 2) serialWriteAt(buf, w.base, w.pos, 32, state);
    }
    w.pos += total;
}

/// The global modular part of a VarDCT frame with an alpha channel: the global
/// tree (treeNodes) with its clustered ANS histograms, the GroupHeader (global
/// tree, default WP header, no transforms), and the channel itself when the
/// image fits a group (else the AC groups carry it).
void writeExtraChannelGlobal(RWStructuredBuffer<uint> buf, inout SerialWriter w) {
    serialWrite(buf, w, 1, 1);   // has_tree
    writeTreeFromNodes(buf, w, treeNodes, params.numTreeNodes);
    writeHistogramSet(buf, w, params.numLeaves, clusterInfo[ecSet()], params.ecLogAlpha, ecSet());
    serialWrite(buf, w, 1, 1);   // use_global_tree
    serialWrite(buf, w, 1, 1);   // weighted header all_default
    serialWrite(buf, w, 2, 0);   // num_transforms = 0
    if (params.ecInGlobal != 0) writeModularImageStream(buf, w, ecSet());
}

/// The raw DCT quant table of a JPEG recompression (ModularFrameEncoder::
/// EncodeQuantTable): a modular stream of three 8x8 channels under a local
/// one-leaf tree (predictor Zero), one histogram, then the tokens.
void writeQuantTableStream(RWStructuredBuffer<uint> buf, inout SerialWriter w) {
    writeLocalGroupHeader(buf, w);
    writeTreeFromNodes(buf, w, qtTree, 1);
    // One context: lz77 off, no context map, ANS, alphabet 7 bits, the default uint config.
    serialWrite(buf, w, 1, 0);
    serialWrite(buf, w, 1, 0);
    serialWrite(buf, w, 2, 7 - 5);
    encodeUintConfigAt(buf, w, 7, 4, 2, 0);
    for (uint i = 0; i < kMaxAlphabet; ++i) gHist[i] = 0;
    uint length = 1;
    for (uint i = 0; i < params.qtCount; ++i) {
        uint tok, nbits, raw;
        hybridEncode(packSigned(rawQuant[i]), tok, nbits, raw);
        ++gHist[tok];
        length = max(length, tok + 1);
    }
    for (uint i = 0; i < kMaxAlphabet; ++i) gAnsCounts[i] = gHist[i];
    ansWriteBestHistogram(buf, w, length, params.histogramTries);
    ansBuildTables(ansInfo, ansReverse, params.qtRegion, length, 7);
    // The tokens (channel-major, raster) as one ANS stream: size, then placement from the end.
    uint total = 32;
    for (uint pass = 1; pass <= 2; ++pass) {
        uint state = kAnsSignature << 16;
        uint pos = w.pos + total;
        for (int i = int(params.qtCount) - 1; i >= 0; --i) {
            uint tok, nbits, raw;
            hybridEncode(packSigned(rawQuant[uint(i)]), tok, nbits, raw);
            const uint e = ansInfo[params.qtRegion * kAnsInfoStride + tok];
            if (pass == 2) {
                pos -= nbits;
                serialWriteAt(buf, w.base, pos, nbits, raw);
            }
            uint ansBits, ansN;
            ansPut(state, e & 0xFFFFu, e >> 16, ansReverse, params.qtRegion * kAnsReverseStride, ansN, ansBits);
            if (pass == 2) {
                pos -= ansN;
                serialWriteAt(buf, w.base, pos, ansN, ansBits);
            } else {
                total += nbits + ansN;
            }
        }
        if (pass == 2) serialWriteAt(buf, w.base, w.pos, 32, state);
    }
    w.pos += total;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void buildGlobalBlobs(uint3 gid: SV_GroupID) {
    if (gid.x != 0) {
        return;
    }
    // DcGlobal: DC dequant default, quantizer, default block context map,
    // default DC chroma-from-luma, no global modular tree.
    SerialWriter dc;
    dc.base = 0;
    dc.pos = 0;
    if (params.jpeg != 0) {
        // DequantMatrices::DecodeDC: the JPEG DC quant steps q0 / (8 x 255) as F16 of 128 x the value.
        serialWrite(blobs, dc, 1, 0);
        serialWrite(blobs, dc, 16, f32tof16Trunc(params.dcQuantJpegX * 128.0f));
        serialWrite(blobs, dc, 16, f32tof16Trunc(params.dcQuantJpegY * 128.0f));
        serialWrite(blobs, dc, 16, f32tof16Trunc(params.dcQuantJpegB * 128.0f));
    } else {
        serialWrite(blobs, dc, 1, 1);   // DequantMatrices DC all_default
    }
    serialWriteU32(blobs, dc, distrBitsOffset(11, 1), distrBitsOffset(11, 2049), distrBitsOffset(12, 4097),
                   distrBitsOffset(16, 8193), quantizer[0]);
    serialWriteU32(blobs, dc, distrVal(16), distrBitsOffset(5, 1), distrBitsOffset(8, 1), distrBitsOffset(16, 1),
                   quantizer[1]);
    if (params.jpeg != 0) {
        // BlockCtxMap, not default: luma DC thresholds (jpegBlockCtx), no quant field
        // thresholds, the map in the simple context-map form (at most 8 block contexts).
        const uint numDc = bctxMap[0];
        const uint numCtxs = bctxMap[1];
        serialWrite(blobs, dc, 1, 0);
        for (uint j = 0; j < 3; ++j) {
            const uint n = j == 1 ? numDc - 1 : 0;
            serialWrite(blobs, dc, 4, n);
            for (uint t = 0; t < n; ++t) {
                serialWriteU32(blobs, dc, distrBitsOffset(4, 0), distrBitsOffset(8, 16), distrBitsOffset(16, 272), distrBitsOffset(32, 65808),
                               packSigned(int(bctxMap[2 + t])));
            }
        }
        serialWrite(blobs, dc, 4, 0);   // no quant field thresholds
        const uint entries = 3 * 13 * numDc;   // kNumOrders orders per channel
        const uint entryBits = ceilLog2Nonzero(numCtxs);
        serialWrite(blobs, dc, 1, 1);   // is_simple
        serialWrite(blobs, dc, 2, entryBits);
        for (uint i = 0; i < entries; ++i) serialWrite(blobs, dc, entryBits, bctxMap[16 + i]);
    } else {
        serialWrite(blobs, dc, 1, 1);   // default block context map
    }
    if (params.jpeg != 0) {
        // ColorCorrelation::DecodeDC with libjxl's non-XYB defaults: colour factor 84, bases 0, no DC offsets.
        serialWrite(blobs, dc, 1, 0);
        serialWrite(blobs, dc, 2, 0);
        serialWrite(blobs, dc, 16, f32tof16Trunc(0.0f));
        serialWrite(blobs, dc, 16, f32tof16Trunc(0.0f));
        serialWrite(blobs, dc, 8, 128);
        serialWrite(blobs, dc, 8, 128);
    } else {
        serialWrite(blobs, dc, 1, 1);   // default DC chroma-from-luma
    }
    if (params.numEcSets != 0) {
        writeExtraChannelGlobal(blobs, dc);
    } else {
        serialWrite(blobs, dc, 1, 0);   // no global modular tree (and no modular image)
    }
    blobBits[0] = dc.pos;

    // AcGlobal: dequant matrices default, one histogram per group set, then per
    // pass natural orders and the pass's clustered AC histogram set.
    SerialWriter ac;
    ac.base = kBlobStrideWords;
    ac.pos = 0;
    if (params.jpeg != 0) {
        // DequantMatrices::Decode: the DCT8 table raw (kQuantModeRAW: denominator 1 / (8 x 255),
        // the table as a modular stream), every other transform's table from the library.
        serialWrite(blobs, ac, 1, 0);
        serialWrite(blobs, ac, 3, 7);   // kind 0 (DCT8): raw: the denominator, then the table stream (its own blob region, buildQuantTableBlob)
        serialWrite(blobs, ac, 16, f32tof16Trunc(1.0f / (8.0f * 255.0f)));
    } else {
        serialWrite(blobs, ac, 1, 1);
    }
    blobBits[1] = ac.pos;
    // The rest of AcGlobal in its own region (after the quant table stream when there is one).
    SerialWriter post;
    post.base = params.acPostRegion * kBlobStrideWords;
    post.pos = 0;
    if (params.jpeg != 0) {
        for (uint kind = 1; kind < params.numQuantKinds; ++kind) serialWrite(blobs, post, 3, 0);   // library tables (runtime count: no unrolling)
    }
    ac = post;
    serialWrite(blobs, ac, ceilLog2Nonzero(params.numGroups), 0);
    for (uint p = 0; p < params.numAcSets; ++p) {
        serialWrite(blobs, ac, 2, 2);   // used_orders: Val(0) is the third distribution
        writeHistogramSet(blobs, ac, params.numAcCtxsUsed, clusterInfo[p], kAcLogAlpha, p);
    }
    blobBits[params.acPostRegion] = ac.pos;
}

/// The raw DCT quant table's modular stream (JPEG recompression) in its blob region.
[shader("compute")]
[numthreads(1, 1, 1)]
void buildQuantTableBlob(uint3 gid: SV_GroupID) {
    if (gid.x != 0) return;
    SerialWriter w;
    w.base = params.qtBlobRegion * kBlobStrideWords;
    w.pos = 0;
    writeQuantTableStream(blobs, w);
    blobBits[params.qtBlobRegion] = w.pos;
}

/// DcGlobal of a lossless Modular frame: the DC dequant defaults, the global
/// tree with its clustered ANS histograms, the global stream's GroupHeader
/// (the YCoCg RCT over channels 0..2; an alpha channel follows untouched) and,
/// when the image fits a group, the channels themselves.
[shader("compute")]
[numthreads(1, 1, 1)]
void buildModularGlobalBlob(uint3 gid: SV_GroupID) {
    if (gid.x != 0) return;
    SerialWriter dc;
    dc.base = 0;
    dc.pos = 0;
    if (params.dcQuantCustom != 0) {
        // DequantMatrices::DecodeDC: dc_quant per channel as F16 of 128 x the value
        // (a DC frame's integers scale by these into XYB).
        serialWrite(blobs, dc, 1, 0);
        serialWrite(blobs, dc, 16, f32tof16Trunc(params.dcQuantX * 128.0f));
        serialWrite(blobs, dc, 16, f32tof16Trunc(params.dcQuantY * 128.0f));
        serialWrite(blobs, dc, 16, f32tof16Trunc(params.dcQuantB * 128.0f));
    } else {
        serialWrite(blobs, dc, 1, 1);    // DequantMatrices DC all_default
    }
    serialWrite(blobs, dc, 1, 1);        // has_tree
    writeTreeFromNodes(blobs, dc, treeNodes, params.numTreeNodes);
    writeHistogramSetLz77(blobs, dc, params.numLeaves + (params.lz77Modular != 0 ? 1 : 0), clusterInfo[0], params.modLogAlpha, 0, params.lz77Modular != 0);
    // GroupHeader of the global stream.
    serialWrite(blobs, dc, 1, 1);        // use_global_tree
    serialWrite(blobs, dc, 1, 1);        // weighted::Header all_default
    // The global stream's transforms apply to the whole image: the palettes, and
    // the RCT when the image is the global stream (it fits a group); otherwise
    // every group's own header carries its RCT.
    const bool imageInGlobal = params.width <= params.groupDim && params.height <= params.groupDim;
    const uint rctType = (params.rct != 0 && imageInGlobal) ? groupRct[0] : 0;
    const uint numTransforms = params.numTransforms + (rctType != 0 ? 1 : 0);
    serialWriteU32(blobs, dc, distrVal(0), distrVal(1), distrBitsOffset(4, 2), distrBitsOffset(8, 18), numTransforms);
    for (uint t = 0; t < params.numTransforms; ++t) {
        const uint4 tr = transforms[t];   // id, begin_c, num_c, nb_colors
        serialWriteU32(blobs, dc, distrVal(0), distrVal(1), distrVal(2), distrVal(3), tr.x);
        serialWriteU32(blobs, dc, distrBitsOffset(3, 0), distrBitsOffset(6, 8), distrBitsOffset(10, 72), distrBitsOffset(13, 1096), tr.y);
        if (tr.x == 1) {   // palette: num_c, nb_colors, nb_deltas 0, predictor 0
            serialWriteU32(blobs, dc, distrVal(1), distrVal(3), distrVal(4), distrBitsOffset(13, 1), tr.z);
            serialWriteU32(blobs, dc, distrBitsOffset(8, 0), distrBitsOffset(10, 256), distrBitsOffset(12, 1280), distrBitsOffset(16, 5376), tr.w);
            serialWriteU32(blobs, dc, distrVal(0), distrBitsOffset(8, 1), distrBitsOffset(10, 257), distrBitsOffset(16, 1281), 0);
            serialWrite(blobs, dc, 4, 0);
        }
    }
    if (rctType != 0) {
        serialWrite(blobs, dc, 2, 0);    // Transform id: RCT
        serialWriteU32(blobs, dc, distrBitsOffset(3, 0), distrBitsOffset(6, 8), distrBitsOffset(10, 72), distrBitsOffset(13, 1096), params.rctBeginC);
        serialWriteU32(blobs, dc, distrVal(6), distrBitsOffset(2, 0), distrBitsOffset(4, 2), distrBitsOffset(6, 10), rctType);
    }
    // The channels of the global stream (all of them when the image fits a group, the meta channels otherwise).
    bool anyGlobal = false;
    for (uint c = 0; c < params.numChannels; ++c) anyGlobal = anyGlobal || chanTable[c].w == 0;
    if (anyGlobal) {
        writeModularImageStream(blobs, dc, 0);
    }
    blobBits[0] = dc.pos;
}
