import os
import yaml
import pandas as pd
import io
import argparse
import sys

DTYPE = {
    'bf16_r': 'B',
    'f16_r': 'H',
    'f32_r': 'S',
    'f64_r': 'D',
    'f8_r': 'F8',
    'bf8_r': 'B8',
    'xf32': 'X'
}

COLUMNS = ['transA','transB','batch_count','m','n','k','a_type','b_type', 'c_type', 'd_type', 'compute_type']


def parse_latency(file: str, output_file: str = None) -> pd.DataFrame:
    """
    Parse hipBLASLt benchmark output file into a pandas DataFrame.
    
    Args:
        file (str):        Path to the hipBLASLt benchmark output file
        output_file (str): Path to the generated latency report
        
    Returns:
        pandas.DataFrame: DataFrame containing parsed benchmark results
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the input file content is not valid
    """
    
    blocks = open(file).read().split('[0]:')[1:]
    if len(blocks) == 0:
        raise ValueError("The benchmark output file does not have the correct format.")
    
    try:
        header = blocks[0].split("\n")[0]
        data = [header] + [b.split("\n")[1].strip() for b in blocks]
        df = pd.read_csv(io.StringIO("\n".join(data)))
    except (pd.errors.EmptyDataError, IndexError) as e:
        raise ValueError("The benchmark output file may be corrupted.")
    
    if output_file:
        df.to_csv(output_file)
    
    return df


def update_compute_type(compute_type):
    if compute_type.startswith('c_'):
        return compute_type
    
    if '32' in compute_type or '64' in compute_type:
        return 'c_' + compute_type
    

def main(hipblaslt_path, log, device=0, thr=0.1, arch="gfx950", workdir="workdir"):
    
    os.makedirs(workdir, exist_ok=True) 
    
    print(f'Working on {log}')
    data = yaml.safe_load(open(log))
    for d in data:
        if 'aux_type' in d:
            del d['aux_type']
        if 'solution_index' in d:
            del d['solution_index']
        if 'algo_method' in d:
            del d['algo_method']
        d['cold_iters'] = 20
        d['iters'] = 100
        d['rotating'] = 512
        d['compute_type'] = update_compute_type(d['compute_type'])
    
    yaml.dump(data, open(log, 'w'), default_flow_style=None, sort_keys=False, width=5000)
    
    ext = log.split('.')[-1]
    output_file = log.replace(f".{ext}", f'.{ext}.out')
    
    output_file = os.path.join(workdir, os.path.basename(output_file)) 
    
    if os.path.isfile(output_file):
        try:
            assert len(parse_latency(output_file)) == len(data)
        except:
            print(f'Running hipblaslt-bench, output will be saved in {output_file}')
            run_hipblaslt_bench(hipblaslt_path, log, output_file, device)
    else:
        print(f'Running hipblaslt-bench, output will be saved in {output_file}')
        run_hipblaslt_bench(hipblaslt_path, log, output_file, device)
    
    df = parse_latency(output_file)
    
    df['call_count'] = pd.DataFrame(data)['call_count']
    df["total (us)"] = df["call_count"] * df["us"]
    df["% of total"] = 100 * df["total (us)"] / df["total (us)"].sum()
    
    df_path = os.path.join(workdir,  os.path.basename(log).replace(f".{ext}", f".csv"))
    print(f"Saving csv to... {df_path}")
    df.sort_values("total (us)", ascending=False).to_csv(df_path, index=False)
    df = df[df["% of total"] >= thr][COLUMNS].drop_duplicates().reset_index(drop=True)

    df.to_csv(os.path.join(workdir,'unique_gemms.csv'), index=False) 
    
    output_dir = os.path.join(workdir, 'tunings')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Generating kernel optimization configs...')
    for (transA, transB, a_type, c_type, compute_type), gby in df.groupby(['transA', 'transB', 'a_type', 'c_type', 'compute_type']):
        compute_type = compute_type.lstrip('c_')
        sizes = gby[['m', 'n', 'batch_count', 'k']].values.tolist()
        
        if len(sizes) == 0:
            continue
        
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"config_{arch}.yaml")
        assert os.path.isfile(config_file), f"Config file not found: {config_file}"
        
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        config['Sizes'] = sizes
        config['DataType'] = DTYPE[a_type]
        config['DestDataType'] = DTYPE[c_type]
        config['ComputeDataType'] =  DTYPE[compute_type]
        config['TRANSA'] = transA
        config['TRANSB'] = transB
        gemm = f'{DTYPE[a_type]}{DTYPE[c_type]}{DTYPE[compute_type]}_{transA}{transB}'

        config_yaml_path = os.path.join(output_dir, f'config_{gemm}.yaml')
        yaml.dump(config, open(config_yaml_path, 'w'), default_flow_style=None, default_style=None, width=5000)
        print(f"config saved in : {config_yaml_path}")
        run_driver_py(config_yaml_path, hipblaslt_path, output_dir)
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=f"Generate TuningDriver config files")
    parser.add_argument('hipblaslt_path', help='Path to hipBLASLt', type=str)
    parser.add_argument('gemm_log', help='GEMM yaml list file', type=str)
    parser.add_argument("--device", "-d", type=int, default=0, help='Which device to run the benchmark in')
    parser.add_argument("--thr", type=float, default=0.1, help='Filter threshold on GEMM contribution.')
    parser.add_argument('--architecture', "-a", help='Target architecture', type=str, default="gfx950")
    parser.add_argument("--workdir", "-w", default="workdir", help="Dir to store intermediate files")

    args = parser.parse_args()
    
    main(args.hipblaslt_path, 
         args.gemm_log, 
         device=args.device,
         thr=args.thr,
         arch=args.architecture,
         workdir=args.workdir)
    
    