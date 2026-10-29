// Copyright (c) 2026 jesus luque.
//
// Splines (libjxl splines.cc): the globals thread left the quantised splines
// in the arena (frame.slangh readSplines). One thread per spline dequantises
// its control points and DCTs, walks the centripetal Catmull-Rom curve through
// them at unit arc length and turns every point into a Gaussian segment with
// the colour and sigma of the continuous IDCTs at that arc position
// (counted, then written after a prefix sum). The segments are bucketed by
// row (a difference array over the rows, a serial scan, a scatter), and one
// thread per pixel adds the segments of its row that reach it, as
// DrawSegment does, to the frame's planes before upsampling.
module dec_splines;

RWStructuredBuffer<uint>  arena;
RWStructuredBuffer<float> xyb;        // the frame's 3 planes, paddedWidth * paddedHeight
RWStructuredBuffer<float> ctrl;       // scratch: every spline's absolute control points (x, y)
RWStructuredBuffer<uint>  segCount;   // per spline: its segments (mode 0), then where they start (splineOffsets)
RWStructuredBuffer<float> segments;   // kSegWords per segment
RWStructuredBuffer<uint>  rowDiff;    // height + 2: segments starting / ending per row, then the coverage
RWStructuredBuffer<uint>  rowStart;   // height + 1: first row entry of each row
RWStructuredBuffer<uint>  rowFill;    // height + 1: scatter counters
RWStructuredBuffer<uint>  segIdx;     // the row entries: segment indices
RWStructuredBuffer<uint>  status;     // [0] segments, [1] row entries, [2] 1 when a capacity was exceeded

struct SplineParams {
    uint width;
    uint height;
    uint paddedWidth;
    uint paddedHeight;
    uint mode;           // splineSegments: 0 count, 1 write
    uint maxSegments;
    uint maxRowEntries;
    uint pad0;
};
ConstantBuffer<SplineParams> params;

static const uint kGsCflBaseX = 13, kGsCflBaseB = 14, kGsSplinesOff = 63;
static const uint kSplineWords = 4 + 128;
static const uint kSegWords = 10;   // cx, cy, maximum distance, 1/sigma, sigma/4 x intensity, colour x3, y0, y1
static const float kPi = 3.14159265358979323846f;
static const float kSqrt2 = 1.4142135623730951f;
static const float kSqrt0_5 = 0.7071067811865476f;
static const float kChannelWeight[4] = {0.0042f, 0.075f, 0.07f, 0.3333f};
static const float kDistanceExp = 5.0f;   // JXL_HIGH_PRECISION

/// FastCosf (fast_math-inl.h): range reduction, a Taylor-like polynomial on x/4 and two angle duplications.
float fastCos(float x) {
    const float pi2 = kPi * 2.0f;
    const float xm = x - floor(x * (0.5f / kPi)) * pi2;
    const float xpi = min(xm, pi2 - xm);
    const bool above = xpi >= kPi / 2.0f;
    const float xh = above ? kPi - xpi : xpi;
    const float xs = xh * 0.25f;
    const float x2 = xs * xs;
    const float x4 = x2 * x2;
    const float pre = x4 * 0.06960438f + (x2 * -0.84087373f + 1.68179268f);
    const float s1 = pre * pre - 1.414213562f;
    const float s2 = s1 * s1 - 1.0f;
    return above ? -s2 : s2;
}

/// FastErff: 1 - 1 / p(|x|)^8 with libjxl's constants.
float fastErf(float x) {
    const float a = abs(x);
    const float d1 = a * 7.77394369e-02f + 2.05260015e-04f;
    const float d2 = d1 * a + 2.32120216e-01f;
    const float d3 = d2 * a + 2.77820801e-01f;
    const float d4 = d3 * a + 1.0f;
    const float d5 = d4 * d4;
    const float inv = 1.0f / d5;
    const float r = 1.0f - inv * inv;
    return x <= 0.0f ? -r : r;
}

/// ContinuousIDCT of one of the spline's 32-point DCTs (arena ints at `off`,
/// scaled by `weight`) at position t in [0, 31].
float continuousIdct(uint off, float weight, float t) {
    float result = 0.0f;
    const float tandhalf = t + 0.5f;
    for (uint i = 0; i < 32; ++i) {
        const float dct = float(int(arena[off + i])) * (i == 0 ? kSqrt0_5 : 1.0f) * weight;
        result += kSqrt2 * dct * fastCos(kPi / 32.0f * float(i) * tandhalf);
    }
    return result;
}

float2 ctrlPoint(uint base, uint i) { return float2(ctrl[2 * (base + i)], ctrl[2 * (base + i) + 1]); }

/// The extended control point list of DrawCentripetalCatmullRomSpline: a mirrored point at each end.
float2 extPoint(uint base, uint n, uint j) {
    if (j == 0) return 2.0f * ctrlPoint(base, 0) - ctrlPoint(base, 1);
    if (j == n + 1) return 2.0f * ctrlPoint(base, n - 1) - ctrlPoint(base, n - 2);
    return ctrlPoint(base, j - 1);
}

/// Intermediate point k of the Catmull-Rom curve: 16 per control point interval, then the last point.
float2 curvePoint(uint base, uint n, uint k) {
    if (n == 1) return ctrlPoint(base, 0);
    const uint numInter = (n - 1) * 16 + 1;
    if (k + 1 >= numInter) return ctrlPoint(base, n - 1);
    const uint seg = k / 16, i = k % 16;
    if (i == 0) return extPoint(base, n, seg + 1);
    float2 p[4];
    for (uint j = 0; j < 4; ++j) p[j] = extPoint(base, n, seg + j);
    float d[3], t[4];
    t[0] = 0.0f;
    for (uint j = 0; j < 3; ++j) {
        d[j] = sqrt(max(length(p[j + 1] - p[j]), 1e-30f));
        t[j + 1] = t[j] + d[j];
    }
    const float tt = d[0] + (float(i) / 16.0f) * d[1];
    float2 a[3];
    for (uint j = 0; j < 3; ++j) a[j] = p[j] + ((tt - t[j]) / d[j]) * (p[j + 1] - p[j]);
    float2 b[2];
    for (uint j = 0; j < 2; ++j) b[j] = a[j] + ((tt - t[j]) / (d[j] + d[j + 1])) * (a[j + 1] - a[j]);
    return b[0] + ((tt - t[1]) / d[1]) * (b[1] - b[0]);
}

/// ForEachEquallySpacedPoint: point k of the walk at unit arc length and its
/// multiplier (1, or the remaining arc at the end). Returns false after the last.
struct Walk {
    uint   next;         // index of the next curve point
    uint   numInter;
    float2 current;
    float2 previous;
    float  arcFromPrevious;
    bool   started;
    bool   done;
};

Walk walkBegin(uint base, uint n) {
    Walk w;
    w.numInter = n == 1 ? 1 : (n - 1) * 16 + 1;
    w.next = 0;
    w.current = curvePoint(base, n, 0);
    w.previous = w.current;
    w.arcFromPrevious = 0.0f;
    w.started = false;
    w.done = false;
    return w;
}

/// The next point to draw; false when the walk has ended.
bool walkNext(inout Walk w, uint base, uint n, out float2 point, out float multiplier) {
    point = w.current;
    multiplier = 1.0f;
    if (w.done) return false;
    if (!w.started) {   // the first point
        w.started = true;
        w.previous = w.current;
        w.arcFromPrevious = 0.0f;
        return true;
    }
    for (;;) {
        if (w.next >= w.numInter) {
            point = w.previous;
            multiplier = w.arcFromPrevious;
            w.done = true;
            return true;
        }
        const float2 np = curvePoint(base, n, w.next);
        const float arcToNext = length(np - w.previous);
        if (w.arcFromPrevious + arcToNext >= 1.0f) {
            w.current = w.previous + ((1.0f - w.arcFromPrevious) / arcToNext) * (np - w.previous);
            point = w.current;
            multiplier = 1.0f;
            w.previous = w.current;
            w.arcFromPrevious = 0.0f;
            return true;
        }
        w.arcFromPrevious += arcToNext;
        w.previous = np;
        ++w.next;
    }
}

/// One thread per spline: mode 0 counts its segments, mode 1 writes them from segCount[spline].
[shader("compute")]
[numthreads(64, 1, 1)]
void splineSegments(uint3 tid: SV_DispatchThreadID) {
    const uint table = arena[kGsSplinesOff];
    if (table == 0) return;
    const uint numSplines = arena[table];
    const uint spline = tid.x;
    if (spline >= numSplines) return;
    const int quantAdj = int(arena[table + 1]);
    const float invQuant = quantAdj >= 0 ? 1.0f / (1.0f + 0.125f * float(quantAdj)) : (1.0f - 0.125f * float(quantAdj));
    const float yToX = asfloat(arena[kGsCflBaseX]);
    const float yToB = asfloat(arena[kGsCflBaseB]);
    const uint sp = table + 2 + spline * kSplineWords;
    // Control points: absolute, into the scratch (this spline's offset: the earlier splines' counts).
    uint base = 0;
    for (uint i = 0; i < spline; ++i) base += arena[table + 2 + i * kSplineWords + 2] + 1;
    const uint n = arena[sp + 2] + 1;
    {
        int cx = int(arena[sp]), cy = int(arena[sp + 1]);
        int dx = 0, dy = 0;
        ctrl[2 * base] = float(cx);
        ctrl[2 * base + 1] = float(cy);
        const uint pts = arena[sp + 3];
        for (uint k = 1; k < n; ++k) {
            dx += int(arena[pts + 2 * (k - 1)]);
            dy += int(arena[pts + 2 * (k - 1) + 1]);
            cx += dx;
            cy += dy;
            ctrl[2 * (base + k)] = float(cx);
            ctrl[2 * (base + k) + 1] = float(cy);
        }
    }
    // The arc: how many points the walk draws and the last one's multiplier.
    uint count = 0;
    float lastMultiplier = 0.0f;
    {
        Walk w = walkBegin(base, n);
        float2 pt;
        float m;
        while (walkNext(w, base, n, pt, m)) {
            ++count;
            lastMultiplier = m;
        }
    }
    const float arcLength = float(int(count) - 2) + lastMultiplier;
    if (arcLength <= 0.0f) {
        if (params.mode == 0) segCount[spline] = 0;
        return;
    }
    const float invArcLength = 1.0f / arcLength;
    uint written = 0;
    const uint start = params.mode == 0 ? 0 : segCount[spline];
    Walk w = walkBegin(base, n);
    float2 pt;
    float m;
    uint k = 0;
    while (walkNext(w, base, n, pt, m)) {
        const float progress = min(1.0f, float(k) * invArcLength);
        ++k;
        const float t = 31.0f * progress;
        const float y = continuousIdct(sp + 4 + 32, kChannelWeight[1] * invQuant, t);
        float color[3];
        color[0] = continuousIdct(sp + 4, kChannelWeight[0] * invQuant, t) + yToX * y;
        color[1] = y;
        color[2] = continuousIdct(sp + 4 + 64, kChannelWeight[2] * invQuant, t) + yToB * y;
        const float sigma = continuousIdct(sp + 4 + 96, kChannelWeight[3] * invQuant, t);
        // ComputeSegments.
        if (!(isfinite(sigma) && sigma != 0.0f && isfinite(1.0f / sigma) && isfinite(m))) continue;
        float maxColor = 0.01f;
        for (uint c = 0; c < 3; ++c) maxColor = max(maxColor, abs(color[c] * m));
        const float maximumDistance = sqrt(-2.0f * sigma * sigma * (log(0.1f) * kDistanceExp - log(maxColor)));
        const int y0 = max(int(round(pt.y - maximumDistance)), 0);
        const int y1 = min(int(round(pt.y + maximumDistance)) + 1, int(params.height));
        if (y1 <= y0) continue;
        if (params.mode == 1) {
            const uint seg = start + written;
            if (seg < params.maxSegments) {
                const uint o = seg * kSegWords;
                segments[o + 0] = pt.x;
                segments[o + 1] = pt.y;
                segments[o + 2] = maximumDistance;
                segments[o + 3] = 1.0f / sigma;
                segments[o + 4] = 0.25f * sigma * m;
                segments[o + 5] = color[0];
                segments[o + 6] = color[1];
                segments[o + 7] = color[2];
                segments[o + 8] = asfloat(uint(y0));
                segments[o + 9] = asfloat(uint(y1));
            }
        }
        ++written;
    }
    if (params.mode == 0) segCount[spline] = written;
}

/// One thread: where each spline's segments start, the total, and clean row counters.
[shader("compute")]
[numthreads(1, 1, 1)]
void splineOffsets(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const uint table = arena[kGsSplinesOff];
    const uint numSplines = table == 0 ? 0 : arena[table];
    uint total = 0;
    for (uint i = 0; i < numSplines; ++i) {
        const uint c = segCount[i];
        segCount[i] = total;
        total += c;
    }
    status[0] = total;
    status[1] = 0;
    status[2] = total > params.maxSegments ? 1u : 0u;
    for (uint y = 0; y <= params.height + 1; ++y) {
        rowDiff[y] = 0;
        if (y <= params.height) rowFill[y] = 0;
    }
}

/// One thread per segment: the rows it covers, as a difference array.
[shader("compute")]
[numthreads(64, 1, 1)]
void splineRowCounts(uint3 tid: SV_DispatchThreadID) {
    const uint total = min(status[0], params.maxSegments);
    if (tid.x >= total) return;
    const uint o = tid.x * kSegWords;
    const uint y0 = asuint(segments[o + 8]), y1 = asuint(segments[o + 9]);
    uint before;
    InterlockedAdd(rowDiff[y0], 1u, before);
    InterlockedAdd(rowDiff[y1], 0xFFFFFFFFu, before);
}

/// One thread: the rows' first entries from the coverage.
[shader("compute")]
[numthreads(1, 1, 1)]
void splineRowStarts(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    uint coverage = 0, total = 0;
    for (uint y = 0; y < params.height; ++y) {
        coverage += rowDiff[y];
        rowStart[y] = total;
        total += coverage;
    }
    rowStart[params.height] = total;
    status[1] = total;
    if (total > params.maxRowEntries) status[2] = 1;
}

/// One thread per segment: its index into every row it covers.
[shader("compute")]
[numthreads(64, 1, 1)]
void splineScatter(uint3 tid: SV_DispatchThreadID) {
    const uint total = min(status[0], params.maxSegments);
    if (tid.x >= total || status[2] != 0) return;
    const uint o = tid.x * kSegWords;
    const uint y0 = asuint(segments[o + 8]), y1 = asuint(segments[o + 9]);
    for (uint y = y0; y < y1; ++y) {
        uint slot;
        InterlockedAdd(rowFill[y], 1u, slot);
        const uint e = rowStart[y] + slot;
        if (e < params.maxRowEntries) segIdx[e] = tid.x;
    }
}

/// One thread per pixel: DrawSegment for every segment of its row that reaches it.
[shader("compute")]
[numthreads(64, 1, 1)]
void splineDraw(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n || status[2] != 0) return;
    const uint x = tid.x % params.width, y = tid.x / params.width;
    const uint e0 = rowStart[y], e1 = rowStart[y + 1];
    if (e0 == e1) return;
    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint at = y * params.paddedWidth + x;
    float v[3] = {xyb[at], xyb[plane + at], xyb[2 * plane + at]};
    for (uint e = e0; e < e1; ++e) {
        const uint o = segIdx[e] * kSegWords;
        const float cx = segments[o], cy = segments[o + 1], maxd = segments[o + 2];
        const int sx = int(round(cx - maxd)), ex = int(round(cx + maxd));
        if (int(x) < sx || int(x) > ex) continue;
        const float invSigma = segments[o + 3];
        const float dx = float(x) - cx, dy = float(y) - cy;
        const float distance = sqrt(dx * dx + dy * dy);
        const float f = fastErf((distance * 0.5f + 0.353553391f) * invSigma) - fastErf((distance * 0.5f - 0.353553391f) * invSigma);
        const float intensity = segments[o + 4] * f * f;
        for (uint c = 0; c < 3; ++c) v[c] += segments[o + 5 + c] * intensity;
    }
    xyb[at] = v[0];
    xyb[plane + at] = v[1];
    xyb[2 * plane + at] = v[2];
}
