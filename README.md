## Requirements: 
Code for NAACL2019 paper "Benchmarking Approximate Inference Methods for Neural Structured Prediction".

The code is written in python2.7 and requies Theano0.8.
- GPU and CUDA 8 are required
- Theano 0.8
- lasagne
- torchfile
- numpy


## Reference
```
@inproceedings{tu-gimpel-2019-benchmarking,
    title = "Benchmarking Approximate Inference Methods for Neural Structured Prediction",
    author = "Tu, Lifu  and
      Gimpel, Kevin",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1335",
    pages = "3313--3324",
    abstract = "Exact structured inference with neural network scoring functions is computationally challenging but several methods have been proposed for approximating inference. One approach is to perform gradient descent with respect to the output structure directly (Belanger and McCallum, 2016). Another approach, proposed recently, is to train a neural network (an {``}inference network{''}) to perform inference (Tu and Gimpel, 2018). In this paper, we compare these two families of inference methods on three sequence labeling datasets. We choose sequence labeling because it permits us to use exact inference as a benchmark in terms of speed, accuracy, and search error. Across datasets, we demonstrate that inference networks achieve a better speed/accuracy/search error trade-off than gradient descent, while also being faster than exact inference at similar accuracy levels. We find further benefit by combining inference networks and gradient descent, using the former to provide a warm start for the latter.",
}
```

## A breif description about code
There are several different folders:
- CRF: BLSTM-CRF
- CRF+: BLSTM-CRF+
- INF\_NET: infnet(Inference Networks)
- INF\_NET+: infnet+(Inference Networks with additional techniques see detail in the paper)
- gradient\_descent\_inference: gradient descent inference method(the combinations of gradient descent and inference network are also included.)



## To do
- to write a pytorch version of the code

