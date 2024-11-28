Run with e.g.

```
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -o profile_with_nvtx_baseline_with_capture_range.nsys-rep  --capture-range=cudaProfilerApi --capture-range-end=stop python main.py
```
