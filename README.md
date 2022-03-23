# 👏 Survey of Deep Face Anti-spoofing 🔥

This is the official repository of "**[Deep Learning for Face Anti-Spoofing: A Survey](https://arxiv.org/pdf/2106.14948.pdf)**", a comprehensive survey 
of recent progress in deep learning methods for face anti-spoofing (FAS) as well as the datasets and protocols.



### Citation
If you find our work useful in your research, please consider citing:

    @article{yu2021deep,
      title={Deep Learning for Face Anti-Spoofing: A Survey},
      author={Yu, Zitong and Qin, Yunxiao and Li, Xiaobai and Zhao, Chenxu and Lei, Zhen and Zhao, Guoying},
      journal={arXiv preprint arXiv:2106.14948},
      year={2021}
    }


## Introduction
We present a comprehensive review of recent deep learning methods for face anti-spoofing (mostly from 2018 to 2021). It covers hybrid (handcrafted+deep), pure deep learning, and generalized learning based methods for monocular RGB face anti-spoofing. It also includes multi-modal learning based methods as well as specialized sensor based FAS. It also presents detailed comparision among publicly available datasets, together with several classical evaluation protocols.

🔔 We will update this page frequently~ :tada::tada::tada:

---
## Contents

- [Datasets](#data)
  - [Using commercial RGB camera](#data_RGB)
  - [With multiple modalities or specialized sensors](#data_Multimodal)
- [Deep FAS methods with commercial RGB camera](#methods_RGB)
  - [Hybrid (handcrafted + deep)](#hybrid)
  - [End-to-end binary cross-entropy supervision](#binary)
  - [Pixel-wise auxiliary supervision](#auxiliary)
  - [Generative model with pixel-wise supervision](#generative)
  - [Domain adaptation](#DA)
  - [Domain generalization](#DG)
  - [Zero/Few-shot learning](#zero-shot)
  - [Anomaly detection](#oneclass)
- [Deep FAS methods with advanced sensor](#methods_advanced)
  - [Learning upon specialized sensor](#sensor)
  - [Multi-modal learning](#multimodal)
  
---
  ![image](https://github.com/ZitongYu/DeepFAS/blob/main/Topology.png)   
  
---


<a name="data" />

### 1️⃣ Datasets

<a name="data_RGB" />

#### Datasets recorded with commercial RGB camera

| Dataset    | Year | #Live/Spoof | #Sub. |  Setup | Attack Types |
| --------   | -----    | -----  |  -----  | ----- |------------------------|
| [NUAA](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.607.5449&rep=rep1&type=pdf)   | 2010 | 5105/7509(I) | 15 |  N/R | Print(flat, wrapped)|
| [YALE Recaptured](https://www.ic.unicamp.br/~rocha/pub/papers/2011-icip-spoofing-detection.pdf)   | 2011 | 640/1920(I) | 10 |  50cm-distance from 3 LCD minitors | Print(flat) |
| [CASIA-MFSD](http://www.cbsr.ia.ac.cn/users/jjyan/ZHANG-ICB2012.pdf)   | 2012 | 150/450(V) | 50 |  7 scenarios and 3 image quality | Print(flat, wrapped, cut), Replay(tablet)|
| [REPLAY-ATTACK](http://publications.idiap.ch/downloads/papers/2012/Chingovska_IEEEBIOSIG2012_2012.pdf)   | 2012 | 200/1000(V) | 50 |  Lighting and holding | Print(flat), Replay(tablet, phone) |
| [Kose and Dugelay](https://ieeexplore.ieee.org/document/6595862)   | 2013 | 200/198(I) | 20 |  N/R | Mask(hard resin) |
| [MSU-MFSD](http://biometrics.cse.msu.edu/Publications/Face/WenHanJain_FaceSpoofDetection_TIFS15.pdf)   | 2014 | 70/210(V) | 35 |  Indoor scenario; 2 types of cameras | Print(flat), Replay(tablet, phone) |
| [UVAD](https://ieeexplore.ieee.org/document/7017526)   | 2015 | 808/16268(V) | 404 | Different lighting, background and places in two sections | Replay(monitor) |
| [REPLAY-Mobile](https://ieeexplore.ieee.org/document/7736936)   | 2016 | 390/640(V) | 40 |  5 lighting conditions | Print(flat), Replay(monitor) |
| [HKBU-MARs V2](https://link.springer.com/chapter/10.1007/978-3-319-46478-7_6)   | 2016 | 504/504(V) | 12 | 7 cameras from stationary and mobile devices and 6 lighting settings | Mask(hard resin) from Thatsmyface and REAL-f |
| [MSU USSA](https://ieeexplore.ieee.org/document/7487030)   | 2016 | 1140/9120(I) | 1140 |  Uncontrolled; 2 types of cameras | Print(flat), Replay(laptop, tablet, phone)|
| [SMAD](https://ieeexplore.ieee.org/document/7867821)   | 2017 | 65/65(V) | - |  Color images from online resources | Mask(silicone) |
| [OULU-NPU](https://ieeexplore.ieee.org/document/7961798)   | 2017 | 720/2880(V) | 55 |  Lighting & background in 3 sections | Print(flat), Replay(phone) |
| [Rose-Youtu](https://ieeexplore.ieee.org/document/8279564)   | 2018 | 500/2850(V) | 20 | 5 front-facing phone camera; 5 different illumination conditions | Print(flat), Replay(monitor, laptop),Mask(paper, crop-paper)|
| [SiW](https://arxiv.org/abs/1803.11097)   | 2018 | 1320/3300(V) | 165 |  4 sessions with variations of distance, pose, illumination and expression | Print(flat, wrapped), Replay(phone, tablet, monitor)|
| [WFFD](https://arxiv.org/abs/2005.06514)   | 2019 | 2300/2300(I) 140/145(V) | 745 |  Collected online; super-realistic; removed low-quality faces | Waxworks(wax)|
| [SiW-M](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Deep_Tree_Learning_for_Zero-Shot_Face_Anti-Spoofing_CVPR_2019_paper.pdf)   | 2019 | 660/968(V) | 493 |  Indoor environment with pose, lighting and expression variations | Print(flat), Replay, Mask(hard resin, plastic, silicone, paper, Mannequin), Makeup(cosmetics, impersonation, Obfuscation), Partial(glasses, cut paper)|
| [Swax](https://arxiv.org/abs/1910.09642)   | 2020 | Total 1812(I) 110(V) | 55 |  Collected online; captured under uncontrolled scenarios | Waxworks(wax)|
| [CelebA-Spoof](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_5)   | 2020 | 156384/469153(I) | 10177 |  4 illumination conditions; indoor & outdoor; rich annotations | Print(flat, wrapped), Replay(monitor tablet, phone), Mask(paper)|
| [RECOD-Mtablet](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0238058)   | 2020 | 450/1800(V) | 45 | Outdoor environment and low-light & dynamic sessions | Print(flat), Replay(monitor) |
| [CASIA-SURF 3DMask](https://ieeexplore.ieee.org/document/9252183)   | 2020 | 288/864(V)  | 48 |  High-quality identity-preserved; 3 decorations and 6 environments | Mask(mannequin with 3D print) |
| [HiFiMask](https://arxiv.org/abs/2104.06148)   | 2021 | 13650/40950(V) | 75 |  three mask decorations; 7 recording devices; 6 lighting conditions; 6 scenes | Mask(transparent, plaster, resin)|




<a name="data_Multimodal" />

#### Datasets with multiple modalities or specialized sensors

| Dataset    | Year | #Live/Spoof | #Sub. |  M&H | Setup | Attack Types |
| --------   | -----    | -----  |  -----  | -----  | -----  |------------------------|
| [3DMAD](https://ieeexplore.ieee.org/document/6810829)   | 2013 | 170/85(V) | 17 |  VIS, Depth | 3 sessions (2 weeks interval) | Mask(paper, hard resin)|
| [GUC-LiFFAD](https://ieeexplore.ieee.org/document/7018027)   | 2015 | 1798/3028(V) | 80 |  Light field | Distance of 1.5 constrained conditions | Print(Inkjet paper, Laserjet paper), Replay(tablet)|
| [3DFS-DB](https://www.researchgate.net/publication/277905873_Three-dimensional_and_two-and-a-half-dimensional_face_recognition_spoofing_using_three-dimensional_printed_models)   | 2016 | 260/260(V) | 26 |  VIS, Depth | Head movement with rich angles | Mask(plastic)|
| [BRSU Skin/Face/Spoof](https://ieeexplore.ieee.org/document/7550052)   | 2016 | 102/404(I) | 137 |  VIS, SWIR | multispectral SWIR with 4 wavebands 935nm, 1060nm, 1300nm and 1550nm | Mask(silicon, plastic, resin, latex)|
| [Msspoof](https://link.springer.com/chapter/10.1007/978-3-319-28501-6_8)   | 2016 | 1470/3024(I) | 21 |  VIS, NIR | 7 environmental conditions | Black&white Print(flat) |
| [MLFP](https://ieeexplore.ieee.org/document/8014774)   | 2017 | 150/1200(V) | 10 |  VIS, NIR, Thermal | Indoor and outdoor with fixed and random backgrounds | Mask(latex, paper) |
| [ERPA](https://www.researchgate.net/publication/320177829_What_You_Can't_See_Can_Help_You_-_Extended-Range_Imaging_for_3D-Mask_Presentation_Attack_Detection)   | 2017 | Total 86(V) | 5 |  VIS, Depth, NIR, Thermal | Subject positioned close (0.3∼0.5m) to the 2 types of cameras | Print(flat), Replay(monitor), Mask(resin, silicone) |
| [LF-SAD ](http://www.ee.cityu.edu.hk/~lmpo/publications/2019_JEI_Face_Liveness.pdf)   | 2018 | 328/596(I) | 50 |  Light field | Indoor fix background, captured by Lytro ILLUM camera | Print(flat, wrapped), Replay(monitor) |
| [CSMAD](https://ieeexplore.ieee.org/document/8698550)   | 2018 | 104/159(V+I) | 14 |  VIS, Depth, NIR, Thermal | 4 lighting conditions | Mask(custom silicone) |
| [3DMA](https://ieeexplore.ieee.org/document/8909845)   | 2019 | 536/384(V) | 67 |  VIS, NIR | 48 masks with different ID; 2 illumination & 4 capturing distances | Mask(plastics) |
| [CASIA-SURF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.pdf)   | 2019 | 3000/18000(V) | 1000 |  VIS, Depth, NIR | Background removed; Randomly cut eyes, nose or mouth areas | Print(flat, wrapped, cut) |
| [WMCA](https://ieeexplore.ieee.org/document/8714076)   | 2019 | 347/1332(V) | 72 |  VIS, Depth, NIR, Thermal | 6 sessions with different backgrounds and illumination; pulse data for bonafide recordings | Print(flat), Replay(tablet), Partial(glasses), Mask(plastic, silicone, and paper, Mannequin) |
| [CeFA](https://openaccess.thecvf.com/content/WACV2021/html/Liu_CASIA-SURF_CeFA_A_Benchmark_for_Multi-Modal_Cross-Ethnicity_Face_Anti-Spoofing_WACV_2021_paper.html)   | 2020 | 6300/27900(V) | 1607 |  VIS, Depth, NIR | 3 ethnicities; outdoor & indoor; decoration with wig and glasses | Print(flat, wrapped), Replay, Mask(3D print, silica gel) |
| [HQ-WMCA](https://ieeexplore.ieee.org/abstract/document/9146362)   | 2020 | 555/2349(V) | 51 | VIS, Depth, NIR, SWIR, Thermal | Indoor; 14 ‘modalities’, including 4 NIR and 7 SWIR wavelengths; masks and mannequins were heated up to reach body temperature | Laser or inkjet Print(flat), Replay(tablet, phone), Mask(plastic, silicon, paper, mannequin), Makeup, Partial(glasses, wigs, tatoo) |
| [PADISI-Face](https://arxiv.org/pdf/2108.12081.pdf)   | 2021 | 1105/924(V) | 360 | VIS, Depth, NIR, SWIR, Thermal | Indoor, fixed background, 60-frame sequence of 1984 × 1264 pixel images | print(flat), replay(tablet, phone), mask(plastic, silicon, transparent, Mannequin), makeup/tatoo, partial(glasses,funny eye) |



---
<a name="methods_RGB" />

### 2️⃣ Deep FAS methods with commercial RGB camera

- temp
<a name="binary" />

#### End-to-end binary cross-entropy supervision
| Method    | Year | Backbone | Loss |  Input | Static/Dynamic |
| --------   | -----    | -----  |  -----  | -----  | -----  |
| [CNN1](https://arxiv.org/abs/1408.5601) / [code](https://paperswithcode.com/paper/learn-convolutional-neural-network-for-face)   | 2014 | 8-layer CNN | Trained with SVM |  RGB | S|
| [SimpleNet](https://arxiv.org/abs/2006.16028) / [code](https://github.com/AlexanderParkin/CASIA-SURF_CeFA)  | 2020 | Multi-stream 5-layer CNN | Binary CE loss |  RGB, OF, RP | D|
| [ViTranZFAS](https://arxiv.org/abs/2011.08019) / [code](https://github.com/anjith2006/bob.paper.ijcb2021_vision_transformer_pad)  |IJCB 2021 | Vision Transformer | Binary CE loss |  RGB | S|

<a name="auxiliary" />

#### Pixel-wise auxiliary supervision

| Method    | Year | Supervision | Backbone |  Input | Static/Dynamic |
| --------   | -----    | -----  |  -----  | -----  | -----  |
| [CDCN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf) / [code](https://github.com/ZitongYu/CDCN)   | CVPR 2020 | Depth | DepthNet |  RGB | S|
| [FAS-SGTD](https://arxiv.org/abs/2003.08061) / [code](https://github.com/clks-wzz/FAS-SGTD)   | CVPR 2020 | Depth | DepthNet, STPM |  RGB | D|
| [DC-CDN](https://arxiv.org/abs/2105.01290) / [code](https://github.com/ZitongYu/CDCN)   | IJCAI 2021 | Depth | CDCN |  RGB | S|
| [LMFD-PAD](https://arxiv.org/pdf/2109.07950.pdf) / [code](https://github.com/meilfang/LMFD-PAD)  | 2021 | BinaryMask | Dual-ResNet50 |  RGB + frequency map | S|
| [DSDG+DUM](https://arxiv.org/abs/2112.00568) / [code](https://github.com/JDAI-CV/faceX-Zoo)   | TIFS 2021 | Depth | CDCN |  RGB | S|
| [EPCR](https://arxiv.org/pdf/2111.12320.pdf) / [code](https://github.com/clks-wzz/EPCR)  | 2021 | BinaryMask | CDCN |  RGB | S|

<a name="generative" />

#### Generative model with pixel-wise supervision

| Method    | Year | Supervision | Backbone |  Input | Static/Dynamic |
| --------   | -----    | -----  |  -----  | -----  | -----  |
| [De-Spoof](https://arxiv.org/abs/1807.09968) / [code](https://github.com/yaojieliu/ECCV2018-FaceDeSpoofing)   | ECCV 2018 | Depth, BinaryMask, FourierMap | DSNet, DepthNet |  RGB, HSV | S|
| [LGSC](https://arxiv.org/abs/2005.03922) / [code](https://github.com/vis-var/lgsc-for-fas)   | 2020 | ZeroMap (live) | U-Net, ResNet18 |  RGB | S|
| [TAE](http://publications.idiap.ch/downloads/papers/2020/Mohammadi_InfoVAE_ICASSP_2020.pdf) / [code](https://gitlab.idiap.ch/bob/bob.paper.icassp2020_facepad_generalization_infovae)   | ICASSP 2020 | Binary CE loss, Reconstruction loss | Info-VAE, DenseNet161 |  RGB | S|
| [STDN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630392.pdf) / [code](https://github.com/yaojieliu/ECCV20-STDN)   | ECCV 2020 | BinaryMask, RGB Input (live) | U-Net, PatchGAN |  RGB | S|



<a name="DG" />

#### Domain generalization


| Method    | Year | Backbone | Loss |  Static/Dynamic |
| --------   | -----    | -----  |  -----  | -----  | 
| [SSDG](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf) / [code](https://github.com/taylover-pei/SSDG-CVPR2020)   | CVPR 2020  | ResNet18 | Binary CE loss, Single-Side adversarial loss, Asymmetric Triplet loss |  S|
| [RF-Meta](https://arxiv.org/abs/1911.10771) / [code](https://github.com/rshaojimmy/AAAI2020-RFMetaFAS)   | AAAI 2020 | DepthNet | Binary CE loss, Depth loss |  S|
| [VLAD-VSA](https://dl.acm.org/doi/abs/10.1145/3474085.3475284) / [code](https://github.com/Liubinggunzu/VLAD-VSA)   | ACMMM 2021 | DepthNet or ResNet18 | BCE loss + triplet loss + domain adversarial loss + orthogonal loss +  centroid adaptation loss + intra loss  |  S|
| [FGHV](https://arxiv.org/abs/2112.14894) / [code](https://github.com/lustoo/fghv)   | AAAI 2022 | DepthNet | Variance + Relative Correlation + Distribution Discrimination Constraints  |  S|
| [SSAN](https://arxiv.org/pdf/2203.05340.pdf) / [code](https://github.com/wangzhuo2019/SSAN)   | CVPR 2022 | DepthNet/ResNet18 | CE loss + Domain Adversarial loss + Contrastive loss  |  S|


<a name="oneclass" />

#### Anomaly detection


| Method    | Year | Backbone | Loss |  Input |
| --------   | -----    | -----  |  -----  | -----  | 
| [End2End-Anomaly](https://arxiv.org/abs/2007.05856) / [code](https://github.com/yashasvi97/IJCB2020_anomaly)   | 2020 | VGG-Face | Binary CE loss, Pairwise confusion |  RGB|


---
<a name="methods_advanced" />

### 3️⃣ Deep FAS methods with advanced sensor


<a name="sensor" />

#### Learning upon specialized sensor


| Method    | Year | Backbone | Loss |  Input | Static/Dynamic |
| --------   | -----    | -----  |  -----  | -----  | -----  |
| [SpecDiff](https://arxiv.org/abs/1907.12400) / [code](https://github.com/Akinori-F-Ebihara/SpecDiff-spoofing-detector)   | 2020 | ResNet4 | Binary CE loss |  Concatenated face images w/ and w/o flash | S|



<a name="multimodal" />

#### Multi-modal learning

| Method    | Year | Backbone | Loss |  Input | Fusion |
| --------   | -----    | -----  |  -----  | -----  | -----  |
| [FeatherNets](https://arxiv.org/abs/1904.09290) / [code](https://paperswithcode.com/paper/190409290)   | 2019 | Ensemble-FeatherNet | Binary CE loss |  Depth, NIR | Decision-level |
| [mmfCNN](https://dl.acm.org/doi/10.1145/3343031.3351001) / [code](https://github.com/SkyKuang/Face-anti-spoofing)   | ACMMM 2019 | ResNet34 | Binary CE loss, Binary Center Loss | RGB, NIR, Depth, HSV, YCbCr | Feature-level|
| [MM-CDCN](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w39/Yu_Multi-Modal_Face_Anti-Spoofing_Based_on_Central_Difference_Networks_CVPRW_2020_paper.pdf) / [code](https://github.com/ZitongYu/CDCN)   | 2020 | CDCN | Pixel-wise binary loss, Contrastive depth loss |  RGB, Depth, NIR | Feature&Decision-level|
| [CMFL](https://arxiv.org/abs/2103.00948) / [code](https://github.com/anjith2006/bob.paper.cross_modal_focal_loss_cvpr2021)   | CVPR 2021 | DenseNet161 | Binary CE loss, Cross modal focal loss |  RGB, Depth | Feature-level|
| [FlexModal-FAS](https://arxiv.org/abs/2202.08192) / [code](https://github.com/zitongyu/flex-modal-fas)   | 2022 | CDCN, ResNet50, ViT | BCE loss, Pixel-wise binary loss |  RGB, Depth, IR | Feature-level|

