# ply_io - PLY File Reader/Writer (MIT License)

A lightweight PLY I/O module that uses only numpy.
It was created to replace the plyfile package (GPL-3.0).

## Usage

```python
from ply_io import PlyData, PlyElement
```

## Supported Features

* Read ASCII / binary little-endian / binary big-endian PLY
* Write binary little-endian PLY
* Scalar properties (float32, uint8, int32, etc.)
* List properties (e.g., face vertex_indices)
* Dynamic property inspection (`p.name for p in element.properties`)

## Switching to the original plyfile package

To use the original plyfile package instead of this custom implementation:

1. Install plyfile: `pip install plyfile`
   (Note: the GPL-3.0 license will apply to your project)

2. Replace `ply_io/__init__.py` with the following:

```python
from plyfile import PlyData, PlyElement
from plyfile import PlyProperty as Property

__all__ = ["PlyData", "PlyElement", "Property"]
```

3. `ply_io/_native.py` is no longer used.
