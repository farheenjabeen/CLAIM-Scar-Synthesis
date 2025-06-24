# [CLAIM: Clinically-Guided LGE Augmentation for Realistic and Diverse Myocardial Scar Synthesis and Segmentation](https://arxiv.org/abs/2506.15549)
CLAIM has been accepted by [AIiH 2025](https://aiih.cc/)

![Figure1](https://github.com/farheenjabeen/CLAIM-Scar-Synthesis/blob/main/Figure1.png)

Deep learning-based myocardial scar segmentation from late gadolinium enhancement (LGE) cardiac MRI has shown great potential for accurate and timely diagnosis and treatment planning for structural cardiac diseases. However, the limited availability and variability of LGE images with high-quality scar labels restrict the development of robust segmentation models. To address this, we introduce CLAIM: \textbf{C}linically-Guided \textbf{L}GE \textbf{A}ugmentation for Real\textbf{i}stic and Diverse \textbf{M}yocardial Scar Synthesis and Segmentation framework, a framework for anatomically grounded scar generation and segmentation. At its core is the SMILE module (Scar Mask generation guided by cLinical knowledgE), which conditions a diffusion-based generator on the clinically adopted AHA 17-segment model to synthesize images with anatomically consistent and spatially diverse scar patterns. In addition, CLAIM employs a joint training strategy in which the scar segmentation network is optimized alongside the generator, aiming to enhance both the realism of synthesized scars and the accuracy of the scar segmentation performance. Experimental results show that CLAIM produces anatomically coherent scar patterns and achieves higher Dice similarity with real scar distributions compared to baseline models. Our approach enables controllable and realistic myocardial scar synthesis and has demonstrated utility for downstream medical imaging task. 

## Citation
```
@article{ramzan2025claim,
  title={CLAIM: Clinically-Guided LGE Augmentation for Realistic and Diverse Myocardial Scar Synthesis and Segmentation},
  author={Ramzan, Farheen and Kiberu, Yusuf and Jathanna, Nikesh and Jamil-Copley, Shahnaz and Clayton, Richard H and others},
  journal={arXiv preprint arXiv:2506.15549},
  year={2025}
}
```

## Acknowledgement
Our code is based on [LeFusion](https://github.com/M3DV/LeFusion), and we greatly appreciate the efforts of the respective authors for providing open-source code.
