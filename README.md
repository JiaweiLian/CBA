# CBA (Contextual Background Attack)

Note that we choose aircraft as the insterested target during experiments, so the proposed CBA is also called Attack Aircraft with Aircraft (A3)

## Introduction

In this paper, a novel Contextual Background Attack (CBA) framework is proposed to fool aerial detectors in the physical world, which can achieve strong attack efficacy and transferability in real-world scenarios even without smudging the interested objects at all. Specifically, the targets of interest, *i.e.* the aircraft in aerial images, are adopted to mask adversarial patches. The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world. To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy. Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously. We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods. We summarize our algorithm in [CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World](https://arxiv.org/pdf/2302.13519.pdf).

## Requirements:

* Pytorch 1.10

* Python 3.6

* CUDA 11.1

* mmdetection 2.23

* mmcv 1.5.0

## Run



## Citation

If you use CBA method for attacks in your research, please consider citing

```
@article{lian2023cba,
  title={CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World},
  author={Lian, Jiawei and Wang, Xiaofei and Su, Yuru and Ma, Mingyang and Mei, Shaohui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--16},
  year={2023},
  publisher={IEEE}
}
```
