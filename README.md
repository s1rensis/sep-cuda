# sep-cuda

Originally at [sep-developers/sep](https://github.com/sep-developers), currently only contains background modeling, highly vibe-coded, **use with caution**.

Build the library:

```bash
cmake -S . -B build
cmake --build build -j
```

Generate the installable package under `./build/dist`:

```bash
cmake --build build --target dist
```

Main outputs:

```text
build/dist/include/sep_cuda.h
build/dist/include/sep_cuda_addon.h
build/dist/lib/lib_sep_cuda.so
build/dist/lib/cmake/SEPCUDA/
build/dist/python/sep_cuda.py
```

Build the minimal example against the generated package:

```bash
cmake -S examples/minimal -B /tmp/sep-cuda-example -DCMAKE_PREFIX_PATH=$PWD/build/dist
cmake --build /tmp/sep-cuda-example -j
```

Use the Python background wrapper from the generated package:

```bash
export PYTHONPATH=$PWD/build/dist/python
python3 -c "import sep_cuda; print(sep_cuda.Background)"
```
