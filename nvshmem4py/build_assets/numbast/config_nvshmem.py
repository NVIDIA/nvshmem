import click
import jinja2
import os


def generate_cooperative_launch_functions():
    """Generate the list of cooperative launch required functions."""
    
    # Data types from the cybind_assets pattern
    types = [
        # Floats
        'float', 'half', 'double', 'bfloat16',
        # UInts
        'uint8', 'uint16', 'uint32','uint64',
        # Ints
        'int8', 'short', 'int16', 'int32', 'int', 'int64', 'long', 'longlong',
        # Misc
        'size', 'char', 'schar', 'uchar', 'ushort', 'uint', 'ulong', 'ulonglong', 'ptrdiff'
    ]
    
    reduction_operators = ["and", "or", "xor", "max", "min", "sum", "prod"]
    block_variants = ["block", "warp"]
    
    functions = []
    
    # Barrier functions
    functions.extend([
        "nvshmem_barrier",
        "nvshmem_barrier_all",
        "nvshmemx_barrier_block",
        "nvshmemx_barrier_warp", 
        "nvshmemx_barrier_all_block",
        "nvshmemx_barrier_all_warp"
    ])
    
    # Sync functions
    functions.extend([
        "nvshmem_sync",
        "nvshmem_sync_all",
        "nvshmem_team_sync",
        "nvshmemx_sync_block",
        "nvshmemx_sync_warp",
        "nvshmemx_sync_all_block", 
        "nvshmemx_sync_all_warp",
        "nvshmemx_team_sync_block",
        "nvshmemx_team_sync_warp"
    ])
    
    # Collective operations
    for data_type in types:
        # Alltoall
        functions.append(f"nvshmem_{data_type}_alltoall")
        for variant in block_variants:
            functions.append(f"nvshmemx_{data_type}_alltoall_{variant}")
            
        # Broadcast
        functions.append(f"nvshmem_{data_type}_broadcast")
        for variant in block_variants:
            functions.append(f"nvshmemx_{data_type}_broadcast_{variant}")
            
        # Fcollect
        functions.append(f"nvshmem_{data_type}_fcollect")
        for variant in block_variants:
            functions.append(f"nvshmemx_{data_type}_fcollect_{variant}")
            
        # Reduce operations
        for op in reduction_operators:
            functions.append(f"nvshmem_{data_type}_{op}_reduce")
            for variant in block_variants:
                functions.append(f"nvshmemx_{data_type}_{op}_reduce_{variant}")
                
        # Wait operations
        functions.extend([
            f"nvshmem_{data_type}_wait",
            f"nvshmem_{data_type}_wait_until",
            f"nvshmem_{data_type}_wait_until_all",
            f"nvshmem_{data_type}_wait_until_any", 
            f"nvshmem_{data_type}_wait_until_some",
            f"nvshmem_{data_type}_wait_until_all_vector",
            f"nvshmem_{data_type}_wait_until_any_vector",
            f"nvshmem_{data_type}_wait_until_some_vector"
        ])
        
        # Test operations
        functions.extend([
            f"nvshmem_{data_type}_test",
            f"nvshmem_{data_type}_test_all",
            f"nvshmem_{data_type}_test_any",
            f"nvshmem_{data_type}_test_some", 
            f"nvshmem_{data_type}_test_all_vector",
            f"nvshmem_{data_type}_test_any_vector",
            f"nvshmem_{data_type}_test_some_vector"
        ])
    
    # Memory-based collective operations
    functions.extend([
        "nvshmem_alltoallmem",
        "nvshmemx_alltoallmem_block",
        "nvshmemx_alltoallmem_warp",
        "nvshmem_broadcastmem", 
        "nvshmemx_broadcastmem_block",
        "nvshmemx_broadcastmem_warp",
        "nvshmem_fcollectmem",
        "nvshmemx_fcollectmem_block",
        "nvshmemx_fcollectmem_warp"
    ])
    
    # Signal operations
    functions.append("nvshmem_signal_wait_until")
    
    # Format as YAML list with proper indentation
    yaml_list = []
    for func in sorted(set(functions)):  # Remove duplicates and sort
        yaml_list.append(f"  - {func}")
    
    return "\n".join(yaml_list)


@click.command()
@click.option("--nvshmem-home", type=click.Path(exists=True), required=True)
@click.option("--config-version", type=str, required=True)
@click.option("--binding-name", type=str, required=True)
@click.option("--entry-point-path", type=click.Path(exists=True), required=True)
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
def main(nvshmem_home, config_version, binding_name, entry_point_path, input_path, output_path):
    # Load the Jinja2 template
    template_dir = os.path.dirname(input_path)
    template_name = os.path.basename(input_path)
    
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    
    # Generate cooperative launch required functions
    cooperative_functions = generate_cooperative_launch_functions()
    
    # Prepare template variables
    template_vars = {
        'CONFIG_VERSION': config_version,
        'NVSHMEM_HOME': nvshmem_home,
        'ENTRY_POINT_PATH': entry_point_path,
        'OUTPUT_NAME': binding_name,
        'COOPERATIVE_LAUNCH_REQUIRED_FUNCTIONS': cooperative_functions
    }
    
    # Render the template
    rendered_content = template.render(**template_vars)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the rendered content to the output file
    with open(output_path, 'w') as f:
        f.write(rendered_content)
    
    print(f"Successfully generated YAML config file: {output_path}")


if __name__ == "__main__":
    main()
