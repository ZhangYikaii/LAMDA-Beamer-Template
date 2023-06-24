<div align="center">
  <a href="http://tianshou.readthedocs.io"><img width="256px" height="auto" src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/5dc5cbf79bd0d9f984bd803457dbad642c9f4d25/logos/zhijian-logo.jpg?raw=true"></a>
</div>

&nbsp;


<div align="center">
    <img src="https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge" alt="Generic badge", height="23">
    <img src="https://img.shields.io/github/actions/workflow/status/qubvel/segmentation_models.pytorch/tests.yml?branch=master&style=for-the-badge" alt="GitHub Workflow Status (branch)", height="23">
    <img src="https://img.shields.io/readthedocs/smp?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Read the Docs", height="23">
    <br>
    <img src="https://img.shields.io/pypi/v/segmentation-models-pytorch?color=blue&style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI", height="23">
    <img src="https://img.shields.io/pypi/dm/segmentation-models-pytorch?style=for-the-badge&color=blue" alt="PyPI - Downloads", height="23">
    <br>
    <img src="https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch" alt="PyTorch - Version", height="23">
    <img src="https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white" alt="Python - Version", height="23">
</div>
<h4 align="center">
    <p>
        A PyTorch-based Framework for Reusing Pre-trained Models
    <p>
    <p>
        <b>English</b> |
        <a href="https://github.com/baichuan-inc/baichuan-7B/blob/main/README_CN.md">中文</a>
    <p>
</h4>

**Zhijian** ([**执简**驭繁](https://baike.baidu.com/item/%E6%89%A7%E7%AE%80%E9%A9%AD%E7%B9%81)) is a PyTorch-based lightweight framework for **reusing pre-trained models and transferring them to new datasets**. It offers a unified and flexible solution for popular methods such as parameter-efficient based, knowledge distillation based, pre-trained feature based, regularization based, and model merging based reuse.

Zhijian achieves state-of-the-art model capabilities with a **`REUSE`** workflow. The provided interface methods include:


+ Regularization via Pre-Trained Parameters
  <details>
  <summary style="margin-left: 10px;">Fine-tune</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson.<b>
  
  How transferable are features in deep neural networks?</b> In: NeurIPS'14.
  <a href="https://arxiv.org/pdf/1411.1792.pdf">[Paper]</a>
  <a href="">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">L<sup>2</sup> penalty / L<sup>2</sup>SP</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Xuhong Li, Yves Grandvalet, Franck Davoine. <b> 
  
  Explicit Inductive Bias for Transfer Learning with Convolutional Networks.</b> In: ICML'18.
  <a href="https://arxiv.org/pdf/1802.01483.pdf">[Paper]</a>
  <a href="https://github.com/holyseven/TransferLearningClassification">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">DELTA</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Xingjian Li, Haoyi Xiong, Hanchao Wang, Yuxuan Rao, Liping Liu, Zeyu Chen, Jun Huan. <b> 

  DELTA: DEep Learning Transfer using Feature Map with Attention for Convolutional Networks.</b> In: ICLR'19.
  <a href="https://arxiv.org/pdf/1901.09229.pdf">[Paper]</a>
  <a href="https://github.com/lixingjian/DELTA">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Elastic Weight Consolidation</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Abhishek Aich. <b> 
  
  Overcoming catastrophic forgetting in neural
  networks.</b> In: .
  <a href="https://arxiv.org/pdf/1612.00796.pdf">[Paper]</a>
  <a href="https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Batch Spectral Shrinkage</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Xinyang Chen, Sinan Wang, Bo Fu, Mingsheng Long, Jianmin Wang. <b> 
  
  Catastrophic Forgetting Meets Negative Transfer:Batch Spectral Shrinkage for Safe Transfer Learning.</b> In: NeurIPS'19.
  <a href="https://proceedings.neurips.cc/paper_files/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf">[Paper]</a>
  <a href="https://github.com/thuml/Batch-Spectral-Shrinkage">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Spectral Norm</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.<b> 
  
  Spectral Normalization for Generative Adversarial Networks.</b> In: ICLR'18.
  <a href="https://arxiv.org/pdf/1802.05957.pdf">[Paper]</a>
  <a href="https://github.com/pfnet-research/sngan_projection">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">OT-based</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Rakshith Sharma Srinivasa, Mark A. Davenport, Justin Romberg. <b>Trading beams for bandwidth: Imaging with randomized beamforming.</b> In: ICML'19.
  <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Learning_Debiased_Representations_via_Conditional_Attribute_Interpolation_CVPR_2023_paper.pdf">[Paper]</a>
  <a href="https://github.com/ZhangYikaii/chi-square">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>



+ Efficient Tuning with Transferred Addin-like Parameters
  <details>
  <summary style="margin-left: 10px;">Adapter</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.
  
  <b>Parameter-Efficient Transfer Learning for NLP</b> In: ICML'19.
  <a href="https://arxiv.org/pdf/1902.00751.pdf">[Paper]</a>
  <a href="https://github.com/google-research/adapter-bert">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">LoRA</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.
  
  <b>LoRA: Low-Rank Adaptation of Large Language Models</b> In: ICLR'22 .
  <a href="https://arxiv.org/pdf/2106.09685.pdf">[Paper]</a>
  <a href="https://github.com/microsoft/LoRA">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div> 
  </details>

  <details>
  <summary style="margin-left: 10px;">VPT / Prefix</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, Ser-Nam Lim.
  
  <b>Visual Prompt Tuning</b> In: ECCV'22.
  <a href="https://arxiv.org/pdf/2203.12119.pdf">[Paper]</a>
  <a href="https://github.com/kmnp/vpt">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Scaling & Shifting</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Dongze Lian, Daquan Zhou, Jiashi Feng, Xinchao Wang.
  
  <b>Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning</b> In: NeurIPS'22.
  <a href="https://arxiv.org/pdf/2210.08823.pdf">[Paper]</a>
  <a href="https://github.com/dongzelian/SSF">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Fact-Tuning</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Shibo Jie, Zhi-Hong Deng.
  
  <b>FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer</b> In: AAAI'23.
  <a href="https://arxiv.org/pdf/2212.03145.pdf">[Paper]</a>
  <a href="https://github.com/JieShibo/PETL-ViT">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div> 
  </details>

  <details>
  <summary style="margin-left: 10px;">Convpass</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Shibo Jie, Zhi-Hong Deng.
  
  <b>Convolutional Bypasses Are Better Vision Transformer Adapters</b> In: Tech Report 07-2022.
  <a href="https://arxiv.org/pdf/2207.07039.pdf">[Paper]</a>
  <a href="https://github.com/JieShibo/PETL-ViT">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div> 
  </details>

  <details>
  <summary style="margin-left: 10px;">AdaptFormer</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, Ping Luo.
  
  <b>AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition</b> In: NeurIPS'22.
  <a href="https://arxiv.org/pdf/2205.13535.pdf">[Paper]</a>
  <a href="https://github.com/ShoufaChen/AdaptFormer">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>


+ Utilization of Pre-Trained Features and Partial Backbones
  <details>
  <summary style="margin-left: 10px;">Linear Probing</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.
  
  <b>Parameter-Efficient Transfer Learning for NLP</b> In: ICML'19.
  <a href="https://arxiv.org/pdf/1902.00751.pdf">[Paper]</a>
  <a href="https://github.com/google-research/adapter-bert">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Partial-k</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Jason Yosinski, Jeff Clune, Yoshua Bengio, Hod Lipson.<b>
  
  How transferable are features in deep neural networks?</b> In: NeurIPS'14.
  <a href="https://arxiv.org/pdf/1411.1792.pdf">[Paper]</a>
  <a href="">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Nearest Class Mean</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly.
  
  <b>Parameter-Efficient Transfer Learning for NLP</b> In: ICML'19.
  <a href="https://arxiv.org/pdf/1902.00751.pdf">[Paper]</a>
  <a href="https://github.com/google-research/adapter-bert">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">SimpleShot</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Yan Wang, Wei-Lun Chao, Kilian Q. Weinberger, Laurens van der Maaten.
  
  <b>SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning</b> In: CVPR'19.
  <a href="https://arxiv.org/pdf/1911.04623.pdf">[Paper]</a>
  <a href="https://github.com/mileyan/simple_shot">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">BitFit</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Elad Ben Zaken, Shauli Ravfogel, Yoav Goldberg.
  
  <b>BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models</b> In: ACL'22.
  <a href="https://arxiv.org/pdf/2106.10199.pdf">[Paper]</a>
  <a href="https://github.com/benzakenelad/BitFit">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Diff Pruning</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Demi Guo, Alexander M. Rush, Yoon Kim.
  
  <b>Parameter-Efficient Transfer Learning with Diff Pruning</b> In: ACL'21.
  <a href="https://arxiv.org/pdf/2012.07463.pdf">[Paper]</a>
  <a href="https://github.com/dguo98/DiffPruning">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>


+ Supervisions from Pre-Trained Predictions
  <details>
  <summary style="margin-left: 10px;">Vanilla Knowledge Distillation / LwF</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Prithviraj Dhar, Rajat Vikram Singh, Kuan-Chuan Peng, Ziyan Wu, Rama Chellappa.
  
  <b>Learning without Memorizing</b> In: CVPR'19.
  <a href="https://arxiv.org/pdf/1811.08051.pdf">[Paper]</a>
  <a href="https://github.com/stony-hub/learning_without_memorizing">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">FitNet</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio.
  
  <b>FitNets: Hints for Thin Deep Nets</b> In: ICLR'15.
  <a href="https://arxiv.org/pdf/1412.6550.pdf">[Paper]</a>
  <a href="https://github.com/adri-romsor/FitNets">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Relational Knowledge Distillation (RKD)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.
  
  <b>Relational Knowledge Distillation</b> In: CVPR'19.
  <a href="https://arxiv.org/pdf/1412.6550.pdf">[Paper]</a>
  <a href="http://cvlab.postech.ac.kr/research/RKD">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Similarity-Preserving Knowledge Distillation (SPKD)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Frederick Tung, Greg Mori.
  
  <b>Similarity-Preserving Knowledge Distillation</b> In: CVPR'19.
  <a href="https://arxiv.org/pdf/1907.09682.pdf">[Paper]</a>
  <a href="https://github.com/DongGeun-Yoon/SPKD">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Contrastive Representation Distillation (CRD)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Yonglong Tian, Dilip Krishnan, Phillip Isola.
  
  <b>Contrastive Representation Distillation</b> In: ICLR'20.
  <a href="https://arxiv.org/pdf/1910.10699.pdf">[Paper]</a>
  <a href="https://github.com/HobbitLong/RepDistiller">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Variational Information Distillation (VID)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai.
  
  <b>Variational Information Distillation for Knowledge Transfer</b> In: CVPR'19.
  <a href="https://arxiv.org/pdf/1904.05835.pdf">[Paper]</a>
  <a href="https://github.com/HobbitLong/RepDistiller">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Flow of Solution Procedure (FSP)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim.
  
  <b>A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning</b> In: CVPR'17.
  <a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf">[Paper]</a>
  <a href="https://github.com/HobbitLong/RepDistiller">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Neuron Selectivity Transfer (NST)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Zehao Huang, Naiyan Wang.
  
  <b>Like What You Like: Knowledge Distill via Neuron Selectivity Transfer</b> In: CVPR'17.
  <a href="https://arxiv.org/pdf/1707.01219.pdf">[Paper]</a>
  <a href="https://github.com/TuSimple/neuron-selectivity-transfer">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Relationship Facilitated Local Classifier Distillation (REFILLED)</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Han-Jia Ye, Su Lu, De-Chuan Zhan.
  
  <b>Distilling Cross-Task Knowledge via Relationship Matching</b> In: CVPR'20.
  <a href="http://www.lamda.nju.edu.cn/lus/files/CVPR20_ReFilled.pdf">[Paper]</a>
  <a href="https://github.com/njulus/REFILLED">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>


+ Ensembling and Merging Multiple Models
  <details>
  <summary style="margin-left: 10px;">Model Soup</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt.
  
  <b>Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time</b> In: ICML'22.
  <a href="https://arxiv.org/pdf/2203.05482.pdf">[Paper]</a>
  <a href="https://github.com/mlfoundations/model-soups">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">WiSE-FT</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt.
  
  <b>Robust fine-tuning of zero-shot models</b> In: CVPR'22.
  <a href="https://arxiv.org/pdf/2109.01903.pdf">[Paper]</a>
  <a href="https://github.com/mlfoundations/wise-ft">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>

  <details>
  <summary style="margin-left: 10px;">Zip It!</summary>
  <div style="margin-left: 30px;">
  <table class="imgtable" style="border: none; text-align: center;"><tr><td>
  <img src="https://github.com/ZhangYikaii/LAMDA-Beamer-Template/blob/main/logos/adapter.png?raw=true" alt="WSFG" height="120px" style="object-fit: cover; width: 100%; height: 100%;" />&nbsp;</td>
  <td align="left"><ul>
  <li><p>George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, Judy Hoffman.
  
  <b>ZipIt! Merging Models from Different Tasks without Training</b> In: CVPR'23.
  <a href="https://arxiv.org/pdf/2305.03053.pdf">[Paper]</a>
  <a href="https://github.com/gstoica27/ZipIt">[Source Code]</a>
  </p>
  </li>
  <p style="color:#5F5F5F;font-family:Calibri, sans-serif">
  We propose chi-square model, a novel method for learning debiased representation. The chi-square model addresses dataset bias by identifying Intermediate Attribute Samples (IASs) operating.
  </p>
  </ul>
  </td></tr>
  </table>
  </div>
  </details>


<!-- + Assembling Addins with Parameter-Efficient Transfer Learning
  + [Adapter](https://arxiv.org/abs/1902.00751)
  + [LoRA](https://arxiv.org/abs/2106.09685)
  + [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
  + [Scaling & Shifting Your Features](https://arxiv.org/abs/2210.08823)
  + [Factor-Tuning](https://arxiv.org/abs/2212.03145)
  + [Convolutional Bypasses](https://arxiv.org/abs/2207.07039)
+ Allocating Training Strategies via Knowledge Distillation and Regularization
  + [FitNet](https://arxiv.org/abs/1412.6550)
  + [Relational-KD](https://arxiv.org/abs/1904.05068)
  + [L<sup>2</sup>SP](https://arxiv.org/abs/1802.01483)
  + [DELTA](https://arxiv.org/abs/1901.09229)
  + [Batch Spectral Shrinkage](https://proceedings.neurips.cc/paper/2019/hash/c6bff625bdb0393992c9d4db0c6bbe45-Abstract.html)
+ Aggregating Parameters from Model Soup and Merging
  + [Model Soup](https://arxiv.org/abs/2203.05482)
  + [WiSE-FT](https://arxiv.org/abs/2109.01903) -->

💡 **Zhijian** also has the following **highlights**:

+ Support access to any of the **pre-trained model zoo**, including:
  + 🤗Hugging Face series — [Transformers](https://github.com/huggingface/transformers), [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
  + OpenAI series — [CLIP](https://github.com/openai/CLIP)
  + Other popular projects, *e.g.*, [vit-pytorch](https://github.com/lucidrains/vit-pytorch) (stars [14k](https://github.com/lucidrains/vit-pytorch/stargazers)).
+ Extremely easy to **get started** and **customize**
  + Get started with a 10 minute blitz [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb)
  + Customize datasets and pre-trained models with step-by-step instructions [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb)
  + Feel free to create a novel approach for reusing pre-trained model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb)
+ **Concise** things do **big**
  + **State-of-the-art** [VTAB benchmark](TODO) for TODO methods (github链接)
  + only ~TODO lines of the code
  + Support friendly guideline and comprehensive documentation [[here]](TODO) (文档tutorial链接)
  + Support incorporating method like building *LEGO* blocks [[here]](TODO) (文档tutorial链接)
  + Support any dataset and pre-trained model [[here]](TODO) (文档tutorial链接)
  + Support multi-GPU training [[here]](TODO) (文档tutorial链接)
  + Support both [TensorBoard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/) log tools [[here]](TODO) (文档tutorial链接)


> "Zhijian" in Chinese means handling complexity with concise and efficient methods. Given the variations in pre-trained models and the deployment overhead of full parameter fine-tuning, Zhijian represents a solution that is easily reusable, maintains high accuracy, and maximizes the potential of pre-trained models.
> 
> “执简驭繁”的意思是用简洁高效的方法驾驭纷繁复杂的事物。现有预训练模型体系结构差异大，全参数微调部署时间长，所以取名“执简”的意思是考虑一个易上手、快复用、稳精度的解决方案，最大限度激发预训练模型的能力。

&nbsp;

## 🕹️ Quick Start

1. An environment with Python 3.7+ from
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"),
[venv](https://docs.python.org/3/library/venv.html), or [virtualenv](https://virtualenv.pypa.io/en/latest/).

2. Install Zhijian using pip:
   ```bash
   $ pip install zhijian
   ```
   For more details please click [installation instructions](TODO/INSTALL.md).

   + [Option] Install with the newest version through GitHub:
      ```bash
      $ pip install git+https://github.com/ZhangYikaii/zhijian.git@main --upgrade
      ```

3. Open your python console and type
   ```python
   import zhijian
   print(zhijian.__version__)
   ```
   If no error occurs, you have successfully installed Zhijian.


&nbsp;

## Documentation

📚 The tutorials and API documentation are hosted on [zhijian.readthedocs.io](https://zhijian.readthedocs.io/)

&nbsp;

## Why Zhijian?

### 🎯 Concise things do big
TODO 这里画一张图和描述，图中有四个功能板块：Finetune、KD/Reg、PETL、Soup，并表示加入任意一个板块的方法修改的代码不多，只有几行，标注在图旁边。

<table>
  <tr>
    <td colspan="9" style="border-bottom: 2px solid black;"></td>
  </tr>
  <tr>
    <td><b>Model Reuse Framework</b></td>
    <td><b>GitHub Stars</b></td>
    <td><b>Unified View</b></td>
    <td><b># of Alg.<sup>(1)</sup></b></td>
    <td><b># of Backbone<sup>(1)</sup></b></td>
    <td><b># of Dataset<sup>(1)</sup></b></td>
    <td><b>LLM Support</b></td>
    <td><b>Documentation</b></td>
    <td><b>Last Update</b></td>
  </tr>
  <tr><td colspan="9" align="left"><b>Regularization via Pre-Trained Parameters</b></td></tr>
  <tr>
    <td><a href="https://github.com/thuml/Transfer-Learning-Library">Transfer Learning Library</a></td>
    <td><a href="https://github.com/thuml/Transfer-Learning-Library/stargazers">
      <img src="https://img.shields.io/github/stars/thuml/Transfer-Learning-Library" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>7</td>
    <td>2 of Resnet-based</td>
    <td>10</td>
    <td>TD</td>
    <td>TD</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/thuml/Transfer-Learning-Library?label=last%20update">
    </a>
    </td>
  </tr>
  <tr><td colspan="9" align="left"><b>Efficient Tuning with Transferred Addin-like Parameters</b></td></tr>
  <tr>
    <td><a href="https://github.com/huggingface/peft">PEFT</a></td>
    <td><a href="https://github.com/huggingface/peft/stargazers">
      <img src="https://img.shields.io/github/stars/huggingface/peft" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>
    <a>
      <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/huggingface/peft?label=last%20update">
    </a>
    </td>
  </tr>
  <tr>
    <td><a href="https://github.com/adapter-hub/adapter-transformers">adapter-transformers</a></td>
    <td><a href="https://github.com/adapter-hub/adapter-transformers/stargazers">
      <img src="https://img.shields.io/github/stars/adapter-hub/adapter-transformers" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>
  <tr><td colspan="9" align="left"><b>Utilization of Pre-Trained Features and Partial Backbones</b></td></tr>
  <tr>
    <td><a href="https://github.com/sicara/easy-few-shot-learning">Easy Few-Shot Learning</a></td>
    <td><a href="https://github.com/sicara/easy-few-shot-learning/stargazers">
      <img src="https://img.shields.io/github/stars/sicara/easy-few-shot-learning" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>
  <tr><td colspan="9" align="left"><b>Supervisions from Pre-Trained Predictions</b></td></tr>
  <tr>
    <td><a href="https://github.com/AberHu/Knowledge-Distillation-Zoo">Knowledge-Distillation-Zoo</a></td>
    <td><a href="https://github.com/AberHu/Knowledge-Distillation-Zoo/stargazers">
      <img src="https://img.shields.io/github/stars/AberHu/Knowledge-Distillation-Zoo" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>
  <tr>
    <td><a href="https://github.com/G-U-N/PyCIL">PyCIL</a></td>
    <td><a href="https://github.com/G-U-N/PyCIL/stargazers">
      <img src="https://img.shields.io/github/stars/G-U-N/PyCIL" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>
  <tr><td colspan="9" align="left"><b>Ensembling and Merging Multiple Models</b></td></tr>
  <tr>
    <td><a href="https://github.com/mlfoundations/model-soups">Model soups</a></td>
    <td><a href="https://github.com/mlfoundations/model-soups/stargazers">
      <img src="https://img.shields.io/github/stars/mlfoundations/model-soups" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>
  <tr>
    <td><a href="https://github.com/samuela/git-re-basin">Git Re-Basin</a></td>
    <td><a href="https://github.com/samuela/git-re-basin/stargazers">
      <img src="https://img.shields.io/github/stars/samuela/git-re-basin" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>
  <tr>
    <td colspan="9" style="border-bottom: 2px solid grey;"></td>
  </tr>
  </tr>
    <tr>
    <td><b>Zhijian</b></td>
    <td><a href="https://github.com/adapter-hub/adapter-transformers/stargazers">
      <img src="https://img.shields.io/github/stars/adapter-hub/adapter-transformers" alt="GitHub stars">
    </a></td>
    <td>✔️/❌/➖<sup>(2)</sup></td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
    <td>TD</td>
  </tr>

</table>

<sup>(1): access date: 2021-08-08</sup>

<sup>(2): not all algorithms support this feature</sup>


### 📦 Reproducible SoTA Results

**Zhijian** fixed the random seed to ensure reproducibility of the results, with only minor variations across different devices.
Partial results are displayed below. For more, please click [[here]](TODO)

| Method | Tuned Params | C-100 | Cal. | DTD | Flow. | Pets | SVHN | SUN397 | Mean | P-Cam. | E-SAT | R-45 | Retin. | Mean | Clevr/C | Clevr/D | DMLab | KITTI/D | dSpri./L | dSpri./O | S-NORB/A | S-NORB/E | Mean|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <div style="white-space: nowrap">**Adapter** </div> |  | 70.7 | 89.0 | 73.0 | 97.8 | 92.2 | 84.7 | 57.9 | 80.7 | 86.4 | 95.0 | 85.2 | 74.4 | 85.2 | 78.6 | 63.0 | 49.5 | 76.1 | 71.3 | 46.6 | 24.3 | 33.9 | 55.4 |
| <div style="white-space: nowrap">**LoRA** </div> |  | 71.7 | 89.4 | 73.8 | 99.5 | 93.1 | 83.4 | 57.8 | 81.2 | 87.6 | 95.0 | 85.2 | 77.0 | 86.2 | 81.7 | 62.8 | 51.1 | 77.1 | 78.6 | 49.7 | 27.2 | 39.9 | 58.5 |
| <div style="white-space: nowrap">**Visual Prompt Tuning**</div> |  | 71.0 | 89.1 | 62.4 | 99.4 | 92.2 | 84.2 | 56.4 | 79.2 | 80.5 | 95.0 | 71.9 | 76.3 | 80.9 | 73.4 | 62.4 | 44.6 | 77.8 | 80.3 | 41.1 | 24.3 | 40.6 | 55.6 |
| <div style="white-space: nowrap">**Scaling & Shifting Your Features**</div> <a href="">[paper]</a> <a href="">[official code]</a> |  | 71.8 | 89.2 | 72.1 | 99.3 | 92.5 | 86.9 | 58.1 | 81.4 | 85.4 | 94.1 | 84.9 | 76.4 | 85.2 | 77.7 | 64.9 | 49.6 | 78.3 | 79.5 | 48.5 | 23.9 | 33.6 | 57.0 |
| <div style="white-space: nowrap">**Factor-tuning**</div> <a href="">[paper]</a> <a href="">[official code]</a> |  | 72.7 | 89.2 | 73.0 | 99.5 | 92.8 | 85.1 | 58.7 | 81.6 | 85.8 | 95.6 | 84.9 | 77.2 | 85.9 | 77.4 | 62.6 | 38.6 | 77.5 | 40.0 | 11.8 | 25.9 | 38.4 | 46.5 |
| <div style="white-space: nowrap">**Convolutional Bypasses**</div> <a href="">[paper]</a> <a href="">[official code]</a> |  | 72.4 | 87.5 | 74.4 | 99.5 | 92.3 | 88.7 | 54.9 | 81.4 | 86.5 | 94.8 | 82.3 | 77.3 | 85.2 | 68.8 | 60.9 | 47.3 | 79.8 | 74.4 | 47.8 | 27.2 | 45.2 | 56.4 |
| <div style="white-space: nowrap">**Linear Probing** <a href="">[paper]</a> <a href="">[official code]</a></div> |  | 60.9 | 87.8 | 69.3 | 99.3 | 90.5 | 44.2 | 56.3 | 72.6 | 81.4 | 90.5 | 77.2 | 74.5 | 80.9 | 36.6 | 32.4 | 34.6 | 55.0 | 23.1 | 29.3 | 15.9 | 26.4 | 31.7 |
| <div style="white-space: nowrap">**Finetune/Partial-1** <a href="">[paper]</a> <a href="">[official code]</a></div> |  | 65.5 | 88.3 | 71.6 | 98.8 | 91.9 | 61.2 | 55.9 | 76.2 | 83.9 | 93.9 | 83.0 | 74.9 | 83.9 | 50.2 | 49.3 | 44.2 | 63.6 | 45.2 | 42.7 | 18.2 | 27.5 | 42.6 |
| <div style="white-space: nowrap">**Finetune/Total** <a href="">[paper]</a> <a href="">[official code]</a></div> |  | 71.1 | 89.5 | 73.2 | 99.4 | 92.8 | 85.2 | 33.3 | 77.8 | 85.3 | 94.3 | 86.2 | 76.4 | 85.5 | 77.1 | 70.9 | 48.9 | 72.8 | 46.7 | 39.2 | 21.0 | 37.8 | 51.8 |

<!-- **Trained Weights**:

<details>
<summary style="margin-left: 20px;"><b>Adapter</b> from "Parameter-Efficient Transfer Learning for NLP", ICML'19</summary>
<div style="margin-left: 30px;">

|Datasets | Acc@1 | Acc@5 | Link to Weights
|---|---|---|---|
| CIFAR-100 | TODO | TODO | [Google Drive](), [Baidu Drive]() |
| Caltech101| TODO | TODO | [Google Drive](), [Baidu Drive]() |
| DTD | TODO | TODO | [Google Drive](), [Baidu Drive]() |
</div>
</details> -->

&nbsp;

## Installation for Research


&nbsp;

## Contributing

Tianshou is still under development. More algorithms and features are going to be added and we always welcome contributions to help make Tianshou better. If you would like to contribute, please check out [this link](https://tianshou.readthedocs.io/en/master/contributing.html).


&nbsp;

## Citing Zhijian

If you find Tianshou useful, please cite it in your publications.

```latex
@article{tianshou,
  author  = {Jiayi Weng and Huayu Chen and Dong Yan and Kaichao You and Alexis Duburcq and Minghao Zhang and Yi Su and Hang Su and Jun Zhu},
  title   = {Tianshou: A Highly Modularized Deep Reinforcement Learning Library},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {267},
  pages   = {1--6},
  url     = {http://jmlr.org/papers/v23/21-1127.html}
}
```


&nbsp;

## Acknowledgment

Tianshou was previously a reinforcement learning platform based on TensorFlow. You can check out the branch [`priv`](https://github.com/thu-ml/tianshou/tree/priv) for more detail. Many thanks to [Haosheng Zou](https://github.com/HaoshengZou)'s pioneering work for Tianshou before version 0.1.1.

We would like to thank [TSAIL](http://ml.cs.tsinghua.edu.cn/) and [Institute for Artificial Intelligence, Tsinghua University](http://ml.cs.tsinghua.edu.cn/thuai/) for providing such an excellent AI research platform.

