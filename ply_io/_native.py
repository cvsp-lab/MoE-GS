# MIT License
#
# Copyright (c) 2026 unified_moe_gs contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

"""Lightweight PLY file reader/writer using only numpy."""

import numpy as np

# PLY type name -> numpy dtype
_PLY_TO_NUMPY = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

# numpy dtype char -> PLY type name
_NUMPY_TO_PLY = {
    "i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint", "f4": "float", "f8": "double",
}


def _np_dtype_str(np_dtype):
    """Convert a numpy dtype to a short string like 'f4', 'u1'."""
    dt = np.dtype(np_dtype)
    return dt.byteorder.replace("=", "<").replace("|", "") + dt.kind + str(dt.itemsize)


def _to_ply_type(np_type_str):
    """Convert numpy dtype string to PLY type name."""
    clean = np_type_str.lstrip("<>=|")
    return _NUMPY_TO_PLY.get(clean, clean)


class Property:
    """Metadata for a single PLY property."""

    def __init__(self, name, dtype, is_list=False, count_dtype=None, item_dtype=None):
        self.name = name
        self.dtype = dtype
        self.is_list = is_list
        self.count_dtype = count_dtype
        self.item_dtype = item_dtype


class PlyElement:
    """Represents one element (e.g., vertex, face) of a PLY file."""

    def __init__(self, name, data, properties):
        self.name = name
        self._data = data
        self.properties = properties
        self.count = len(data)

    def __getitem__(self, key):
        return self._data[key]

    @staticmethod
    def describe(data, name):
        """Create a PlyElement from a numpy structured array."""
        properties = []
        for field_name in data.dtype.names:
            field_dtype = data.dtype[field_name]
            if field_dtype.subdtype is not None:
                base_dtype, _ = field_dtype.subdtype
                base_str = _np_dtype_str(base_dtype).lstrip("<>=|")
                properties.append(Property(
                    field_name, base_str,
                    is_list=True, count_dtype="u1", item_dtype=base_str,
                ))
            else:
                properties.append(Property(field_name, _np_dtype_str(field_dtype).lstrip("<>=|")))
        return PlyElement(name, data, properties)


class PlyData:
    """Container for PLY file data."""

    def __init__(self, elements):
        self.elements = list(elements)

    def __getitem__(self, key):
        for el in self.elements:
            if el.name == key:
                return el
        raise KeyError(f"Element '{key}' not found")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def write(self, path):
        """Write PLY file in binary little-endian format."""
        with open(path, "wb") as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            for el in self.elements:
                f.write(f"element {el.name} {el.count}\n".encode())
                for prop in el.properties:
                    if prop.is_list:
                        f.write(f"property list {_to_ply_type(prop.count_dtype)} "
                                f"{_to_ply_type(prop.item_dtype)} {prop.name}\n".encode())
                    else:
                        f.write(f"property {_to_ply_type(prop.dtype)} {prop.name}\n".encode())
            f.write(b"end_header\n")

            # Data
            for el in self.elements:
                has_list = any(p.is_list for p in el.properties)
                if not has_list:
                    # Fast path: dump structured array directly
                    # Ensure little-endian
                    le_dtype = np.dtype([(n, el._data.dtype[n].newbyteorder("<"))
                                         for n in el._data.dtype.names])
                    if el._data.dtype != le_dtype:
                        el._data.astype(le_dtype, copy=False).tofile(f)
                    else:
                        el._data.tofile(f)
                else:
                    # Slow path: row by row
                    for i in range(el.count):
                        for prop in el.properties:
                            if not prop.is_list:
                                val = el._data[prop.name][i]
                                f.write(np.array(val, dtype="<" + prop.dtype).tobytes())
                            else:
                                arr = np.asarray(el._data[prop.name][i])
                                count_dt = np.dtype("<" + prop.count_dtype)
                                item_dt = np.dtype("<" + prop.item_dtype)
                                f.write(np.array(len(arr), dtype=count_dt).tobytes())
                                f.write(np.array(arr, dtype=item_dt).tobytes())

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    @staticmethod
    def read(path):
        """Read a PLY file (auto-detects ASCII / binary_little_endian / binary_big_endian)."""
        with open(path, "rb") as f:
            # --- Parse header ---
            header_lines = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file while reading PLY header")
                text = line.decode("ascii", errors="replace").strip()
                header_lines.append(text)
                if text == "end_header":
                    break

            fmt = None
            elements_spec = []  # [(name, count, [(prop_name, ply_type, count_type_or_None, is_list)])]
            current_props = []

            for text in header_lines:
                parts = text.split()
                if not parts or parts[0] in ("ply", "end_header", "comment", "obj_info"):
                    continue
                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    current_props = []
                    elements_spec.append((parts[1], int(parts[2]), current_props))
                elif parts[0] == "property":
                    if parts[1] == "list":
                        current_props.append((parts[4], parts[3], parts[2], True))
                    else:
                        current_props.append((parts[2], parts[1], None, False))

            # --- Read data ---
            if fmt == "binary_little_endian":
                elements = _read_binary(f, elements_spec, "<")
            elif fmt == "binary_big_endian":
                elements = _read_binary(f, elements_spec, ">")
            elif fmt == "ascii":
                elements = _read_ascii(f, elements_spec)
            else:
                raise ValueError(f"Unknown PLY format: {fmt}")

        return PlyData(elements)


# ------------------------------------------------------------------
# Binary reader
# ------------------------------------------------------------------
def _read_binary(f, elements_spec, bo):
    """Read binary PLY data. bo is byte order prefix ('<' or '>')."""
    elements = []
    for name, count, props in elements_spec:
        has_list = any(is_list for _, _, _, is_list in props)

        if not has_list:
            # Fast path: all scalar
            dtype = np.dtype([(pname, bo + _PLY_TO_NUMPY[ptype])
                              for pname, ptype, _, _ in props])
            raw = f.read(count * dtype.itemsize)
            data = np.frombuffer(raw, dtype=dtype, count=count).copy()
            properties = [Property(pname, _PLY_TO_NUMPY[ptype])
                          for pname, ptype, _, _ in props]
            elements.append(PlyElement(name, data, properties))
        else:
            # Slow path: row by row
            rows = []
            for _ in range(count):
                row = {}
                for pname, ptype, ctype, is_list in props:
                    if not is_list:
                        dt = np.dtype(bo + _PLY_TO_NUMPY[ptype])
                        row[pname] = np.frombuffer(f.read(dt.itemsize), dtype=dt)[0]
                    else:
                        ct = np.dtype(bo + _PLY_TO_NUMPY[ctype])
                        n = int(np.frombuffer(f.read(ct.itemsize), dtype=ct)[0])
                        it = np.dtype(bo + _PLY_TO_NUMPY[ptype])
                        row[pname] = np.frombuffer(f.read(n * it.itemsize), dtype=it).copy()
                rows.append(row)

            # Build structured array
            dtype_fields = []
            for pname, ptype, ctype, is_list in props:
                if not is_list:
                    dtype_fields.append((pname, _PLY_TO_NUMPY[ptype]))
                else:
                    list_len = len(rows[0][pname]) if rows else 0
                    dtype_fields.append((pname, _PLY_TO_NUMPY[ptype], (list_len,)))

            data = np.empty(count, dtype=dtype_fields)
            for i, row in enumerate(rows):
                vals = []
                for pname, ptype, ctype, is_list in props:
                    if not is_list:
                        vals.append(row[pname])
                    else:
                        vals.append(tuple(row[pname]))
                data[i] = tuple(vals)

            properties = [
                Property(pname, _PLY_TO_NUMPY[ptype],
                         is_list=is_list,
                         count_dtype=_PLY_TO_NUMPY.get(ctype) if is_list else None,
                         item_dtype=_PLY_TO_NUMPY[ptype] if is_list else None)
                for pname, ptype, ctype, is_list in props
            ]
            elements.append(PlyElement(name, data, properties))

    return elements


# ------------------------------------------------------------------
# ASCII reader
# ------------------------------------------------------------------
def _read_ascii(f, elements_spec):
    """Read ASCII PLY data."""
    elements = []
    for name, count, props in elements_spec:
        has_list = any(is_list for _, _, _, is_list in props)

        if not has_list:
            dtype = np.dtype([(pname, _PLY_TO_NUMPY[ptype])
                              for pname, ptype, _, _ in props])
            rows = []
            for _ in range(count):
                line = f.readline().decode("ascii").strip()
                tokens = line.split()
                row = tuple(np.dtype(_PLY_TO_NUMPY[ptype]).type(tokens[j])
                            for j, (pname, ptype, _, _) in enumerate(props))
                rows.append(row)
            data = np.array(rows, dtype=dtype)
            properties = [Property(pname, _PLY_TO_NUMPY[ptype])
                          for pname, ptype, _, _ in props]
            elements.append(PlyElement(name, data, properties))
        else:
            rows = []
            for _ in range(count):
                line = f.readline().decode("ascii").strip()
                tokens = line.split()
                row = {}
                idx = 0
                for pname, ptype, ctype, is_list in props:
                    if not is_list:
                        row[pname] = np.dtype(_PLY_TO_NUMPY[ptype]).type(tokens[idx])
                        idx += 1
                    else:
                        n = int(tokens[idx])
                        idx += 1
                        row[pname] = np.array(
                            [np.dtype(_PLY_TO_NUMPY[ptype]).type(tokens[idx + k]) for k in range(n)],
                            dtype=_PLY_TO_NUMPY[ptype])
                        idx += n
                rows.append(row)

            dtype_fields = []
            for pname, ptype, ctype, is_list in props:
                if not is_list:
                    dtype_fields.append((pname, _PLY_TO_NUMPY[ptype]))
                else:
                    list_len = len(rows[0][pname]) if rows else 0
                    dtype_fields.append((pname, _PLY_TO_NUMPY[ptype], (list_len,)))

            data = np.empty(count, dtype=dtype_fields)
            for i, row in enumerate(rows):
                vals = []
                for pname, ptype, ctype, is_list in props:
                    if not is_list:
                        vals.append(row[pname])
                    else:
                        vals.append(tuple(row[pname]))
                data[i] = tuple(vals)

            properties = [
                Property(pname, _PLY_TO_NUMPY[ptype],
                         is_list=is_list,
                         count_dtype=_PLY_TO_NUMPY.get(ctype) if is_list else None,
                         item_dtype=_PLY_TO_NUMPY[ptype] if is_list else None)
                for pname, ptype, ctype, is_list in props
            ]
            elements.append(PlyElement(name, data, properties))

    return elements
