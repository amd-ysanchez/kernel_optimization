# GEMM Kernel Optimization Workflow

This workflow depends on [rocm-libraries/projects/hipblaslt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt).

## Required input

The main input for the workflow is the log file resulting from running a workload with HIPBLASLT_LOG_MASK=64/128.

The file should look like this:

```
- {function: matmul, M: 16032, N: 109, K: 16384, lda: 16384, ldb: 16384, ldc: 16032, ldd: 16032, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.0, beta: 0.0, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, rotating: 512, cold_iters: 0, iters: 0, call_count: 1}
- {function: matmul, M: 16032, N: 113, K: 16384, lda: 16384, ldb: 16384, ldc: 16032, ldd: 16032, stride_a: 0, stride_b: 0, stride_c: 0, stride_d: 0, alpha: 1.0, beta: 0.0, transA: T, transB: N, batch_count: 1, scaleA: 0, scaleB: 0, scaleAlpha_vector: false, gradient: false, use_e: false, bias_vector: false, bias_source: d, a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, scale_type: f32_r, bias_type: f32_r, compute_type: c_f32_r, activation_type: none, flush: false, rotating: 512, cold_iters: 0, iters: 0, call_count: 1}
```

## Requirements

Several libraries and packages are necessary to run the workflow.

### hipBLASLt installation

hipBLASLt needs to be build locally so the code can run benchmarks and kernel optimization.

Follow these instructions to fetch the *rocm-libraries* with only hipBLASLt.

```
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt shared/origami
git checkout develop # or the branch you are starting from
```

Then, to build hipBLASLt do the following (replace *gfx950* with your target architecture):

```
cd rocm-libraries/projects/hipblaslt

./install.sh -dc -a gfx950
```

### Python requirements

Install the TuningDriver packeged wheel and other required packages by running:

```
pip install tuningdriver-0.1.0-py3-none-any.whl
```


## Generating configuration files

The first step is to generate the necessary configs to run the kernel optimization.

For this we run the script *gen_configs.py* found in */path/to/TuningDriver/kernel_optimization*.

With the following arguments:

```
usage: gen_configs.py [-h] [--output_dir OUTPUT_DIR] [--device DEVICE] [--thr THR] hipblaslt_path gemm_log

Generate TuningDriver config files

positional arguments:
  hipblaslt_path        Path to hipBLASLt
  gemm_log              GEMM yaml list file

options:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory
  --device DEVICE, -d DEVICE
                        Which device to run the benchmark in
  --thr THR             Filter threshold on GEMM contribution.
  --architecture ARCHITECTURE, -a ARCHITECTURE
                        Target architecture
```

For example:

```
python gen_configs.py /path/to/rocm-libraries/projects/hipblaslt/ /path/to/hipblaslt.log -a gfx950
```

The above command would create a folder called *tunings* that would look like the following:

```
-rwxr-xr-x 1 root root    555 Oct 23 12:40 BBS_TN_0.sh
-rw-r--r-- 1 root root  17685 Oct 23 12:40 BBS_TN_0.yaml
-rwxr-xr-x 1 root root    555 Oct 23 12:40 BBS_TN_1.sh
-rw-r--r-- 1 root root  27006 Oct 23 12:40 BBS_TN_1.yaml
-rwxr-xr-x 1 root root    555 Oct 23 12:40 BBS_TN_2.sh
-rw-r--r-- 1 root root  17735 Oct 23 12:40 BBS_TN_2.yaml
-rwxr-xr-x 1 root root    555 Oct 23 12:40 BBS_TN_3.sh
-rw-r--r-- 1 root root  27505 Oct 23 12:40 BBS_TN_3.yaml
-rwxr-xr-x 1 root root    555 Oct 23 12:40 BBS_TN_4.sh
-rw-r--r-- 1 root root  49986 Oct 23 12:40 BBS_TN_4.yaml
-rwxr-xr-x 1 root root    559 Oct 23 12:40 F8BS_TN_0.sh
-rw-r--r-- 1 root root  91392 Oct 23 12:40 F8BS_TN_0.yaml
-rwxr-xr-x 1 root root    559 Oct 23 12:40 F8BS_TN_1.sh
-rw-r--r-- 1 root root  28302 Oct 23 12:40 F8BS_TN_1.yaml
```

where each yaml file corresponds to a GEMM that would be optimized with a new kernel.

## Running GA Optimization
We have our new configs ready, now we just will run the kernel optimization on all the GEMMs. 

We will use the script *optimize.py*, found in */path/to/TuningDriver/kernel_optimization*.

This script performs several steps.

- **Optimization**. Runs all configs and distributes the workload over all available (or as indicated) GPUs. One kernel per config will be generated.
- **Merge**. Gathers all new kernels into a single library that can be used by hipBLASLt. It will also build the aforementioned library.
- **Benchmark**. Benchmark all GEMMs by using both the 'reference' and 'tuned' libraries. It also verify the validity of the new kernels.
- **Postprocess**. Filters out kernels that have a high accuracy error or that do not improve over the 'reference' library, and outputs the final optimized library. 

<br>

Usage:

```
usage: optimize.py [-h] [--library_dir LIBRARY_DIR] [--devices DEVICES] hipblaslt_path input_dir

Merge tuning results into a single library.

positional arguments:
  hipblaslt_path        Path to hipBLASLt
  input_dir             Path to tuning directory

options:
  -h, --help            show this help message and exit
  --library_dir LIBRARY_DIR
                        Final library output directory
  --log_summary LOG_SUMMARY
                        CSV file generated by gen_configs.py script. Will be used to report the weighted uplift.
  --devices DEVICES, -d DEVICES
                        Comma-separated list of device IDs to use. Ex: 0,3,4,5
```

Example:

```
python /path/to/TuningDriver/kernel_optimization/optimize.py tuning_dir --devices=0,1,2,3
```
To run on all 8 devices just remove the *--devices* flag. 

We may want to detach the script call from the terminal and run it on the background (nohup command and *&*).

```
nohup python -u /path/to/TuningDriver/kernel_optimization/optimize.py tuning_dir --devices=0,1,2,3 2>err 1>out &
```

<br>

After running this script we will get three outputs:
- **RAW performance comparison for all GEMMS.** Saved in *results/raw_results.csv*
- **Filtered performance comparison for all GEMMS.** Saved in *results/final_results.csv*
- **Final (filtered) library files** that will be merged with hipBLASLt. Saved in *library_dir* folder (default set to 'final').
- Print message on screen showing the **average GEMM uplift** and, if the summary csv files was provided, the *weighted total uplift**.


> [!NOTE]  
> If we want to run the kernel optimization step in isolation, one can use the script *loadbalancer.py* that can be found in */path/to/TuningDriver/ConfigGenerator/tuning_driver/loadbalancer.py* for this.

## Final merge into hipBLASLt
Once we have our final library with only kernels that improve, we want to merge it to hipBLASLt.

For this we will use the *TensileMergeLibrary* script that can be found in */path/to/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/TensileMergeLibrary*

Usage:

```
usage: TensileMergeLibrary [-h] [-v VERBOSITY] [--force_merge FORCE_MERGE] [--no_eff] original_dir incremental_dir output_dir

positional arguments:
  original_dir          The library logic directory without tuned sizes
  incremental_dir       The incremental logic directory
  output_dir            The output logic directory

options:
  -h, --help            show this help message and exit
  -v VERBOSITY, --verbosity VERBOSITY
                        0: summary, 1: verbose, 2: debug
  --force_merge FORCE_MERGE
                        Merge previously known sizes unconditionally. Default behavior if not arcturus
  --no_eff              force set eff as 0.0.
```

We want to merge the library into the **Equality** folder for the architecture we are working in.

Example usage for MI35X:

```
/path/to/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/bin/TensileMergeLibrary --no_eff --force_merge True /path/to/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Equality final /path/to/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx950/Equality
```
The above command will merge the new library, found in the *final* folder, into the Equality libraries for gfx950.

**After merging into hipBLASLt we need to re-build it.**

```
/path/to/rocm-libraries/projects/hipblaslt/install.sh -dc -a gfx950
```

> [!CAUTION]
> This step will modify the current kernel libraries in hipBLASLt. You may want to checkout a new branch for this.
