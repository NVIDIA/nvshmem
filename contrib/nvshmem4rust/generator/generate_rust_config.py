# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import click
import jinja2


@click.command()
@click.option("--nvshmem-include-dir",
              type=click.Path(exists=True),
              required=True)
@click.option("--gpu-arch", type=str, required=True)
@click.option("--config-version", type=str, required=True)
@click.option("--binding-name", type=str, required=True)
@click.option("--entry-point-path",
              type=click.Path(exists=True),
              required=True)
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
def main(nvshmem_include_dir, gpu_arch, config_version, binding_name,
         entry_point_path, input_path, output_path):
    template_dir = os.path.dirname(input_path)
    template_name = os.path.basename(input_path)

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_include_path = os.path.join(cuda_home, "include")
    cuda13_cccl_include_path = os.path.join(cuda_home, "include", "cccl")

    if not os.path.exists(cuda_include_path):
        cuda_include_path = None

    if not os.path.exists(cuda13_cccl_include_path):
        cuda13_cccl_include_path = None

    rendered_content = template.render(
        CONFIG_VERSION=config_version,
        NVSHMEM_INCLUDE_DIR=nvshmem_include_dir,
        GPU_ARCH=gpu_arch,
        CUDA_INCLUDE_PATH=cuda_include_path,
        CUDA13_CCCL_INCLUDE_PATH=cuda13_cccl_include_path,
        ENTRY_POINT_PATH=entry_point_path,
        OUTPUT_NAME=binding_name,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_content)

    print(
        f"Successfully generated Rust binding YAML config file: {output_path}")


if __name__ == "__main__":
    main()
