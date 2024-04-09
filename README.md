# ptfid

## Metrics

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

    - SwAV ResNet50

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor resnet50 \
            --normalizer imagenet \
            --resizer clean
        ```

    - DINOv2 ViT-L

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor dinov2 \
            --normalizer imagenet \
            --resizer clean
        ```

    - CLIP ViT-L

        ```sh
        python -m ptfid ./dir/to/dataset1 ./dir/to/dataset2 \
            --feature-extractor clip \
            --normalizer openai \
            --resizer clean
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


## LICENSE

[Apache-2.0 license](./LICENSE)

## Acknowledgements

TODO

## References

```bibtex
@inproceedings{NIPS2017_8a1d6947,
  author = {Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {6626--6637},
  publisher = {Curran Associates, Inc.},
  title = {GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  volume = {30},
  year = {2017}
}
```

TODO

## Author(s)

[Tomoya Sawada](https://github.com/STomoya)
