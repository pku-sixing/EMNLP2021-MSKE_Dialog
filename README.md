# EMNLP2021-MSKE_Dialog

We have updated this project, please use this version.


# Requirements
```
Python=3.6
PyTorch==1.4.0
TorchText==0.6.0
```


#Scripts

The script for training is located at `scripts/train.sh`. 

The script for inference is located at `scripts/infer.sh`.  The default decoding strategy is beam search.  We also provide a script that uses greedy search, please see `scripts/infer_no_beam.sh`





# Citation
```
@inproceedings{mske,
  author    = {Sixing Wu and
               Ying Li and
               Minghui Wang and
               Dawei Zhang and
               Yang Zhou and
               Zhonghai Wu},
  editor    = {Marie{-}Francine Moens and
               Xuanjing Huang and
               Lucia Specia and
               Scott Wen{-}tau Yih},
  title     = {More is Better: Enhancing Open-Domain Dialogue Generation via Multi-Source
               Heterogeneous Knowledge},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican
               Republic, 7-11 November, 2021},
  pages     = {2286--2300},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://aclanthology.org/2021.emnlp-main.175},
  timestamp = {Tue, 09 Nov 2021 13:51:50 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/WuLWZZW21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# Dataset
If you need to get the data before we finish uploading the final version, please contact with us.

