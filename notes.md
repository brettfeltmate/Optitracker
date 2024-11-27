### Motive unpacking sequence

1. Prefix
    1. frame number (Int32ul)
2. marker SETS
    1. num sets
    2. overall size
    3. by set:
        1. label (CString)
        2. num markers
        3. by marker:
            1. Float32l * 3 (x/y/z)
3. legacy markers
    1. num MARKERS
    2. overall size
    3. by marker:
        1. Float32l * 3 (x/y/z)
4. Rigid bodies

