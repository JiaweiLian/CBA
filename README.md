# CBA (Contextual Background Attack)

## Introduction

In this paper, a novel Contextual Background Attack (CBA) framework is proposed to fool aerial detectors in the physical world, which can achieve strong attack efficacy and transferability in real-world scenarios even without smudging the interested objects at all. Specifically, the targets of interest, *i.e.* the aircraft in aerial images, are adopted to mask adversarial patches.
   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.
   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy. 
   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.
   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.
   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.

We summarize our algorithm in [CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World](https://arxiv.org/pdf/2302.13519.pdf).

## Requirements:

* Pytorch 1.10

* Python 3.6

## Citation

If you use CBA method for attacks in your research, please consider citing

```
@ARTICLE{2023arXiv230213519L,
       author = {{Lian}, Jiawei and {Wang}, Xiaofei and {Su}, Yuru and {Ma}, Mingyang and {Mei}, Shaohui},
        title = "{CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2023,
        month = feb,
          eid = {arXiv:2302.13519},
        pages = {arXiv:2302.13519},
          doi = {10.48550/arXiv.2302.13519},
archivePrefix = {arXiv},
       eprint = {2302.13519},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230213519L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
