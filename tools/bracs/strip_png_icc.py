"""ROI の PNG から iCCP チャンク（スキャナの ICC プロファイル、約 120 KB）を取り除く。

画素データ（IDAT）は触らない。PIL は大きな iCCP を持つ PNG を
"Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK" で読めないため。
使い方: python strip_png_icc.py <ディレクトリ>...
"""
import struct
import sys
from pathlib import Path

n_stripped = n_total = 0
for root in sys.argv[1:]:
    for path in sorted(Path(root).rglob("*.png")):
        n_total += 1
        data = path.read_bytes()
        out = [data[:8]]
        pos, found = 8, False
        while pos < len(data):
            (length,) = struct.unpack(">I", data[pos:pos + 4])
            end = pos + 12 + length
            if data[pos + 4:pos + 8] == b"iCCP":
                found = True
            else:
                out.append(data[pos:end])
            pos = end
        if found:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(b"".join(out))
            tmp.rename(path)
            n_stripped += 1
print(f"png files={n_total} stripped={n_stripped}")
