### Motion Blur Restoration (Sparse Motion Kernel)

For the motion blur restoration baseline, we employed the **Sparse Motion Kernel (SMK)** algorithm [1]. We utilized the official MATLAB implementation provided by the authors: [AHU-VRV/TextDeblurring](https://github.com/AHU-VRV/TextDeblurring).

**⚠️ Note on Environment:**
The SMK source code is written in MATLAB and is **not included** in this repository to maintain a Python-centric environment. We treated this step as an external pre-processing module. The restored images are already provided in `./data/dataset_SMK_Restored/`.

#### Custom Batch Processing
Since the original repository (`demo.m`) is designed for single-image inference, we developed a custom MATLAB script (`batch_process_smk.m`) to extend its functionality for dataset-level batch processing.

#### Reproduction Steps
If you wish to reproduce the SMK restoration results from scratch:

1. Clone the external repository:
   ```bash
   git clone [https://github.com/AHU-VRV/TextDeblurring](https://github.com/AHU-VRV/TextDeblurring)