# Source code repository for the paper "An Indexing Scheme and Descriptor for 3D Object Retrieval Based on Local Shape Querying"

This repository contains:

- A reference implementation of the Quick Intersection Count Change Image (along with the other tested descriptors in the paper's evaluation)
- A script which can be used to completely reproduce all results presented in the paper

## Instructions

You only need to have python 3 installed, which comes with Ubuntu. Any dependencies needed by the project itself can be installed using the menu system in the script itself.

You can run the script by executing:

```bash

python3 replicate.py

```

From the root of the repository.

Refer to the included Manual PDF for further instructions.

## System Requirements

The RAM and Disk space requirements are only valid when attempting to reproduce the presented results.

The codebase _should_ be able to compile on Windows, but due to some CUDA driver/SDK compatbility issues we have not yet been able to verify this.

Type | Requirements
-----|----------------------------------------------------------------------------
CPU  | Does not matter
RAM  | At least 32GB
Disk | Must have about ~120GB of storage available to store the downloaded datasets
GPU  | Any NVIDIA GPU (project uses CUDA)
OS   | Ubuntu 16 or higher. Project has been tested on 18 and 20.

## Credits

- Development and implementation: Bart Iver van Blokland, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)
- Supervision: Theoharis Theoharis, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)



