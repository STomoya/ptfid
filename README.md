# ptfid

PyTorch implementation of Fréchet inception distance (FID).

> [!CAUTION]
> Use the original implementations when directly comparing with results that are reported in research papers.

## Metrics

- Fréchet inception distance (FID)

- Kernel inception distance (KID)

- CleanFID

- SwAV-FID

- Fréchet DINO distance (FDD)

- Precision & Recall (P&R)

- Density & Coverage (D&C)

- Probabilistic Precision & Recall (PP&PR)

- Topological Precision & Recall (TopP&R)

## Usage

- Compute FID between two image folders. This command will create a `result.json` containing the computed scores.

    ```sh
    python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2
    ```

- Additionally compute other metrics. Supported metrics are listed [here](#metrics). You can also explicity disable metrics by passing a flag like `--no-<metric>`.

    ```sh
    python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 --no-fid --kid --pr --dc --pppr --toppr
    ```

    Some metrics is required to determine the real dataset to correctly compute the scores (i.e., P&R). By default, `ptfid` assumes that the first path argument points to the real dataset. You can change this behavior using `--dataset1-is-real` or `--no-dataset1-is-real` flag.

    ```sh
    python -m ptfid ./dir/to/fake ./dir/to/real --no-fid --pr --no-dataset1-is-real
    ```

- You can determine the feature extractor using the `--feature-extractor` option.

    In addition to Inception v3, `ptfid` supports ResNet50-SwAV, DINOv2, CLIP models, and also models from `timm` and `open_clip` using the `timm:` and `clip:` prefix. See the examples for details.

    <details>
    <summary>Some examples</summary>

    - SwAV-FID

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor resnet50 \
            --normalizer imagenet \
            --resizer pillow \
            --interpolation bilinear
        ```

    - FDD

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor dinov2 \
            --normalizer imagenet \
            --resizer pillow
        ```

    - CLIP ViT-L

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor clip \
            --normalizer openai \
            --resizer pillow
        ```

    - Timm models

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor timm:convnext_tiny.in12k_ft_in1k \
            --normalizer imagenet \
            --resizer clean
        ```

    - OpenCLIP models

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor clip:ViT-L-14.openai \
            --normalizer openai \
            --resizer clean
        ```
    </details>

> [!IMPORTANT]
> It is recommended to specify the `--normalizer` option too, which defaults to Inception normalization. `--normalizer imagenet` uses the ImageNet mean and std, `--normalizer openai` uses the mean and std used to train the OpenAI CLIP models, and `--normalizer custom --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5` sets the mean and std to the values provided by the user.

> [!IMPORTANT]
> It is also recommended to specify the `--resizer` option, which defaults to the pytorch implementation of the tensorflow v1 bilinear interpolation. `--resizer torch` uses torch interpolation function, `--resizer pillow` uses pillow's resize function, and `--resizer clean` uses the clean-resize method proposed in the CleanFID paper. Only use `--resizer inception` when `--feature-extractor inceptionv3`, otherwise `clean` is recommended.
> You can also change the interpolation mode using the `--interpolation` option. The default is set to `bicubic`. Only `bicubic` and `bilinear` is supported, and this option does not affect `--resizer inception` which only has a bilinear implementation.


- Log process to a file.

    ```sh
    python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 --log-file log.log
    ```


<details>
<summary>Help of ptfid</summary>

```sh
$ python -m ptfid --help

 Usage: python -m ptfid [OPTIONS] DATASET_DIR1 DATASET_DIR2

 Calculate generative metrics given two image folders.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    dataset_dir1      TEXT  Dir to dataset. [default: None] [required]                                                                      │
│ *    dataset_dir2      TEXT  Dir to dataset. [default: None] [required]                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --feature-extractor                              TEXT                                Feature extractor name. [default: inceptionv3]          │
│ --fid                   --no-fid                                                     Flag for FID. [default: fid]                            │
│ --kid                   --no-kid                                                     Flag for KID. [default: no-kid]                         │
│ --pr                    --no-pr                                                      Flag for P&R. [default: no-pr]                          │
│ --dc                    --no-dc                                                      Flag for D&C. [default: no-dc]                          │
│ --pppr                  --no-pppr                                                    Flag for PP&PR. [default: no-pppr]                      │
│ --toppr                 --no-toppr                                                   Flag for TopP&R. [default: no-toppr]                    │
│ --eps                                            FLOAT                               epsilon to avoid zero devision. [default: 1e-12]        │
│ --fid-compute-method                             [original|efficient|gpu]            Method to compute FID. [default: efficient]             │
│ --kid-times                                      FLOAT                               Multiply KID by. [default: 100.0]                       │
│ --kid-subsets                                    INTEGER                             Number of subsets to compute KID. [default: 100]        │
│ --kid-subset-size                                INTEGER                             Number of samples per subset. [default: 1000]           │
│ --kid-degree                                     FLOAT                               degree of polynomial kernel. [default: 3.0]             │
│ --kid-gamma                                      FLOAT                               gamma of polynomial kernel. [default: None]             │
│ --kid-coef0                                      FLOAT                               coef0 of polynomial kernel. [default: 1.0]              │
│ --pr-nearest-k                                   INTEGER                             k for nearest neighbors. [default: 5]                   │
│ --pppr-alpha                                     FLOAT                               Alpha for PP&PR. [default: 1.2]                         │
│ --toppr-alpha                                    FLOAT                               Alpha for TopP&R. [default: 0.1]                        │
│ --toppr-kernel                                   TEXT                                Kernel for TopP&R. [default: cosine]                    │
│ --toppr-randproj        --no-toppr-randproj                                          Random projection for TopP&R. [default: toppr-randproj] │
│ --toppr-f1              --no-toppr-f1                                                Compute F1-score for TopP&R. [default: toppr-f1]        │
│ --seed                                           INTEGER                             Random state seed. [default: 0]                         │
│ --resizer                                        [clean|torch|tensorflow|pillow]     Resize method. [default: tensorflow]                    │
│ --interpolation                                  [bilinear|bicubic]                  Interpolation mode. [default: bicubic]                  │
│ --normalizer                                     [imagenet|openai|inception|custom]  Normalize method. [default: inception]                  │
│ --batch-size                                     INTEGER                             Batch size. [default: 32]                               │
│ --mean                                           <FLOAT FLOAT FLOAT>...              Mean for custom normalizer. [default: None, None, None] │
│ --std                                            <FLOAT FLOAT FLOAT>...              Std for custom normalizer. [default: None, None, None]  │
│ --num-workers                                    INTEGER                             Number of workers. [default: 8]                         │
│ --device                                         [cpu|cuda]                          Device. [default: cuda]                                 │
│ --dataset1-is-real      --no-dataset1-is-real                                        Switch real dataset. [default: dataset1-is-real]        │
│ --log-file                                       TEXT                                File to output logs. [default: None]                    │
│ --result-file                                    TEXT                                JSON file to save results to. [default: results.json]   │
│ --help                                                                               Show this message and exit.                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
</details>


## Comparisons with original implementations

### Dataset

- AFHQv2: https://github.com/clovaai/stargan-v2/tree/master#animal-faces-hq-dataset-afhq

### Generator

- StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch

    - AFHQ cat weights: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl

    - AFHQ dog weights: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl

- seeds: `0-49999`

### Original implementations

- FID: https://github.com/bioinf-jku/TTUR

- CleanFID: https://github.com/GaParmar/clean-fid

- FDD: https://github.com/layer6ai-labs/dgm-eval

- SwAV-FID: https://github.com/stanis-morozov/self-supervised-gan-eval

### Results (stylegan2-ada-pytorch AFHQ cat)

|   |original|`ptfid`|abs. diff|
|:--|-------:|------:|---:|
|FID|$7.2045933595139$|$7.2046109473089$|$0.0000175877950$|
|FID (efficient)|$7.2045933595139$|$7.2046109473117$|$0.0000175877978$|
|FID (GPU)|$7.2045933595139$|$7.2046109473086$|$0.0000175877947$|
|CleanFID|$7.4542446122753$|$7.4542346447187$|$0.0000099675566$|
|FDD|$467.4631578945676$|$467.4631822762613$|$0.0000243816937$|
|SwAV-FID|$2.1126139426609$|$2.1126106721183$|$0.0000032705426$|

### Results (stylegan2-ada-pytorch AFHQ dog)

|   |original|`ptfid`|abs. diff|
|:--|-------:|------:|---:|
|FID|$11.6537845957563$|$11.6537787949336$|$0.0000058008227$|
|FID (efficient)|$11.6537845957563$|$11.6537785565156$|$0.0000060392407$|
|FID (GPU)|$11.6537845957563$|$11.6537787949370$|$0.0000058008193$|
|CleanFID|$11.7839366677424$|$11.7839372424993$|$0.0000005747569$|
|FDD|$502.5668103084015$|$502.5668197314544$|$0.0000094230529$|
|SwAV-FID|$2.8054340345364$|$2.8054317704354$|$0.0000022641010$|

## LICENSE

[Apache-2.0 license](./LICENSE)

## Acknowledgements

The source codes of this repository are based on several publicly available implementations of FID calculations. Many thanks to those who have published their excellent works. Specifically, official implementations of [FID](https://github.com/bioinf-jku/TTUR), [CleanFID](https://github.com/GaParmar/clean-fid), [FDD](https://github.com/layer6ai-labs/dgm-eval), [SwAV-FID](https://github.com/stanis-morozov/self-supervised-gan-eval), [D&C](https://github.com/clovaai/generative-evaluation-prdc), [PP&PR](https://github.com/kdst-team/Probablistic_precision_recall), [TopP&R](https://github.com/LAIT-CVLab/TopPR). Also, [efficient computation of FID](https://github.com/layer6ai-labs/dgm-eval), and [fast computation of FID on GPU](https://github.com/jaywu109/faster-pytorch-fid).

## References

- FID

```bibtex
@inproceedings{NIPS2017_8a1d6947,
  author    = {Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  pages     = {6626--6637},
  publisher = {Curran Associates, Inc.},
  title     = {GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  volume    = {30},
  year      = {2017}
}
```

- KID

```bibtex
@inproceedings{bińkowski2018demystifying,
  author    = {Mikołaj Bińkowski and Dougal J. Sutherland and Michael Arbel and Arthur Gretton},
  booktitle = {International Conference on Learning Representations (ICLR)},
  title     = {Demystifying {MMD} {GAN}s},
  year      = {2018},
  url       = {https://openreview.net/forum?id=r1lUOzWCW},
}
```

- CleanFID

```bibtex
@inproceedings{Parmar_2022_CVPR,
  author    = {Parmar, Gaurav and Zhang, Richard and Zhu, Jun-Yan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  title     = {On Aliased Resizing and Surprising Subtleties in GAN Evaluation},
  month     = {June},
  year      = {2022},
  pages     = {11410-11420}
}
```

- SwAV-FID

```bibtex
@inproceedings{morozov2021on,
  author    = {Stanislav Morozov and Andrey Voynov and Artem Babenko},
  booktitle = {International Conference on Learning Representations (ICLR)},
  title     = {On Self-Supervised Image Representations for {GAN} Evaluation},
  year      = {2021},
  url       = {https://openreview.net/forum?id=NeRdBeTionN}
}
```

- FDD

```bibtex
@inproceedings{stein2023exposing,
  author    = {George Stein and Jesse C. Cresswell and Rasa Hosseinzadeh and Yi Sui and Brendan Leigh Ross and Valentin Villecroze and Zhaoyan Liu and Anthony L. Caterini and Eric Taylor and Gabriel Loaiza-Ganem},
  booktitle = {Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  title     = {Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models},
  year      = {2023},
  url       = {https://openreview.net/forum?id=08zf7kTOoh}
}
```

- P&R

```bibtex
@inproceedings{NEURIPS2018_f7696a9b,
  author    = {Sajjadi, Mehdi S. M. and Bachem, Olivier and Lucic, Mario and Bousquet, Olivier and Gelly, Sylvain},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  publisher = {Curran Associates, Inc.},
  title     = {Assessing Generative Models via Precision and Recall},
  volume    = {31},
  year      = {2018},
  pages     = {5228--5237},
}
```

```bibtex
@inproceedings{NEURIPS2019_0234c510,
  author    = {Kynk\"{a}\"{a}nniemi, Tuomas and Karras, Tero and Laine, Samuli and Lehtinen, Jaakko and Aila, Timo},
  booktitle = {Advances in Neural Information Processing Systems},
  editor    = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
  publisher = {Curran Associates, Inc.},
  title     = {Improved Precision and Recall Metric for Assessing Generative Models (NeurIPS)},
  volume    = {32},
  year      = {2019},
  pages     = {3927--3936},
}

```

- D&C

```bibtex

@inproceedings{pmlr-v119-naeem20a,
  author    = {Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  title     = {Reliable Fidelity and Diversity Metrics for Generative Models},
  series    = {Proceedings of Machine Learning Research (PMLR)},
  volume    = {119},
  year      = {2020},
  pages     = {7176--7185},
}
```

- PP&PR

```bibtex
@inproceedings{Park_2023_ICCV,
  author    = {Park, Dogyun and Kim, Suhyun},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  title     = {Probabilistic Precision and Recall Towards Reliable Evaluation of Generative Models},
  month     = {October},
  year      = {2023},
  pages     = {20099-20109}
}
```

- TopP&R

```bibtex
@inproceedings{kim2023toppr,
  author={Pum Jun Kim and Yoojin Jang and Jisu Kim and Jaejun Yoo},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  title={TopP\&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models},
  year={2023},
  url={https://openreview.net/forum?id=2gUCMr6fDY}
}
```

## Author(s)

[Tomoya Sawada](https://github.com/STomoya)
