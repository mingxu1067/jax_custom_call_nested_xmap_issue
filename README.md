# Reproducer to Issues of nested xmap + custom_vmap

### Installation
```bash
$ git clone ssh://git@gitlab-master.nvidia.com:12051/mingh/vmap_xmap_custom_vmap_issue.git
$ cd jax_custom_call_nested_xmap_issue && pip install .
```
Note: Require 8 GPUs to reproduce

### Issue of `Check failed: !IsManual()`

- Steps to Reproduce
```bash
$ cd tests && python delay_xmap_in_lowering.py


2023-07-17 19:51:05.238696: F external/xla/xla/hlo/ir/hlo_sharding.cc:961] Check failed: !IsManual() 
Aborted (core dumped)
```
