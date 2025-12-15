## 📂 In-the-Wild Dataset

The **In-the-Wild Dataset** included in this repository is designed for validating Pre-Trained Models and evaluating the efficacy of the Fine-Tuning process. It consists of real-world images captured in academic environments (e.g., whiteboards, chalkboards) to test model robustness against authentic noise.

> **Note on Ablation Studies & Distortions:**
> The dataset provided represents the **raw, original imagery** as captured. It does **not** pre-contain the artificially distorted versions (e.g., simulated Motion Blur, Specular Glare) or the algorithmically restored counterparts described in the project report.
>
> To replicate the **Ablation Study** or evaluate specific distortion-correction algorithms (such as *SHR-Net*, *Retinex*, or *Sparse Motion Kernel*), please refer to the image processing pipelines located in the `src/` directory. You will need to apply these algorithms to the raw In-the-Wild dataset to generate the specific test cases.