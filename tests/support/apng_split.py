#!/usr/bin/env python3
"""Splits an APNG into one PNG per displayed frame, compositing as a viewer
would (fcTL x/y offsets, dispose none / background / previous, blend source /
over). Usage: apng_split.py in.png out_prefix  -> out_prefix_000.png ...
Prints the number of frames."""
import struct
import sys
import zlib


def chunks(data: bytes):
    pos = 8
    while pos + 8 <= len(data):
        n = struct.unpack('>I', data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        yield kind, data[pos + 8:pos + 8 + n]
        pos += 12 + n


def unfilter(raw: bytes, w: int, h: int, bpp: int) -> list:
    stride = w * bpp
    out = []
    prev = bytearray(stride)
    p = 0
    for _ in range(h):
        f = raw[p]
        line = bytearray(raw[p + 1:p + 1 + stride])
        p += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif f == 4:
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        out.append(line)
        prev = line
    return out


def write_png(path: str, w: int, h: int, bpp: int, rows: list) -> None:
    raw = b''.join(b'\x00' + bytes(r) for r in rows)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6 if bpp == 4 else 2, 0, 0, 0)

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main() -> None:
    data = open(sys.argv[1], 'rb').read()
    prefix = sys.argv[2]
    w = h = 0
    bpp = 4
    frames = []        # (fctl, zdata)
    current = None
    for kind, body in chunks(data):
        if kind == b'IHDR':
            w, h, depth, ctype = struct.unpack('>IIBB', body[:10])
            assert depth == 8, 'only 8-bit APNG expected here'
            bpp = 4 if ctype == 6 else 3
        elif kind == b'fcTL':
            current = [struct.unpack('>IIIIIHHBB', body), bytearray()]
            frames.append(current)
        elif kind == b'IDAT':
            if current is None:   # a still PNG (or default image not part of the animation)
                current = [(0, w, h, 0, 0, 1, 10, 0, 0), bytearray()]
                frames.append(current)
            current[1] += body
        elif kind == b'fdAT':
            current[1] += body[4:]
    canvas = [bytearray(w * bpp) for _ in range(h)]
    count = 0
    for fctl, z in frames:
        _, fw, fh, fx, fy, _, _, dispose, blend = fctl
        rows = unfilter(zlib.decompress(bytes(z)), fw, fh, bpp)
        before = [bytearray(r) for r in canvas]
        for yy in range(fh):
            for xx in range(fw):
                src = rows[yy][xx * bpp:(xx + 1) * bpp]
                dst = (fy + yy, (fx + xx) * bpp)
                if blend == 0 or bpp == 3 or src[3] == 255:
                    canvas[dst[0]][dst[1]:dst[1] + bpp] = src
                elif src[3] != 0:
                    a = src[3] / 255.0
                    old = canvas[dst[0]][dst[1]:dst[1] + bpp]
                    oa = old[3] / 255.0
                    na = a + oa * (1 - a)
                    px = bytearray(4)
                    for c in range(3):
                        px[c] = round((src[c] * a + old[c] * oa * (1 - a)) / na) if na > 0 else 0
                    px[3] = round(na * 255)
                    canvas[dst[0]][dst[1]:dst[1] + 4] = px
        write_png('%s_%03d.png' % (prefix, count), w, h, bpp, canvas)
        count += 1
        if dispose == 1:
            for yy in range(fh):
                canvas[fy + yy][fx * bpp:(fx + fw) * bpp] = bytearray(fw * bpp)
        elif dispose == 2:
            canvas = before
    print(count)


if __name__ == '__main__':
    main()
