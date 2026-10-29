#!/usr/bin/env python3
"""Writes a small RGBA APNG for the animation tests: a square moving over a
gradient, with a fading alpha. Frames are full canvases (APNG dispose none,
blend source). Usage: apng_make.py out.png [frames] [width] [height]"""
import struct
import sys
import zlib


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)


def frame_rows(index: int, w: int, h: int, x0: int = 0, y0: int = 0, fw: int = 0, fh: int = 0) -> bytes:
    """A full canvas (fw == 0) or the sub-rectangle (x0, y0, fw, fh) of it."""
    if fw == 0:
        fw, fh = w, h
    rows = bytearray()
    sx = 8 + index * 11
    sy = 6 + index * 5
    for yy in range(fh):
        rows.append(0)   # filter: none
        y = y0 + yy
        for xx in range(fw):
            x = x0 + xx
            r = (x * 255) // max(1, w - 1)
            g = (y * 255) // max(1, h - 1)
            b = 128
            a = 255
            if sx <= x < sx + 20 and sy <= y < sy + 16:
                r, g, b = 250, 40, 60
                a = 255 - index * 40
            elif fw != w:
                a = 90 + (x * 3) % 120   # a translucent patch over the previous canvas
            rows += bytes((r, g, b, a))
    return bytes(rows)


def main() -> None:
    out = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    h = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    partial = len(sys.argv) > 5 and sys.argv[5] == 'partial'   # frames 1.. are sub-rectangles blended over
    png = bytearray(b'\x89PNG\r\n\x1a\n')
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
    png += chunk(b'acTL', struct.pack('>II', frames, 0))
    seq = 0
    for i in range(frames):
        if partial and i > 0:
            fx, fy, fw, fh = 4 + 9 * i, 3 + 7 * i, 40, 28
            blend = 1   # over
        else:
            fx, fy, fw, fh = 0, 0, w, h
            blend = 0
        # fcTL: sequence, w, h, x, y, delay num/den, dispose, blend
        png += chunk(b'fcTL', struct.pack('>IIIIIHHBB', seq, fw, fh, fx, fy, 1, 10, 0, blend))
        seq += 1
        data = zlib.compress(frame_rows(i, w, h, fx, fy, fw, fh), 9)
        if i == 0:
            png += chunk(b'IDAT', data)
        else:
            png += chunk(b'fdAT', struct.pack('>I', seq) + data)
            seq += 1
    png += chunk(b'IEND', b'')
    with open(out, 'wb') as f:
        f.write(png)


if __name__ == '__main__':
    main()
