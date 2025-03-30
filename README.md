# Debias with Backdoor (DwB)

**Official Implementation for Paper**
[&#34;Backdoor for Debias: Mitigating Model Bias with Backdoor Attack-based Artificial Bias&#34;](https://arxiv.org/abs/XXXX.XXXXX)

## Authors

- Shangxi Wu (Beijing Institute of Technology)
- Qiuyang He (Beijing Institute of Technology)
- Jian Yu (Beijing Institute of Technology)
- Jitao Sang (Beijing Institute of Technology)

## Citation

```bibtex
@article{wu2024backdoor,
  title={Backdoor for Debias: Mitigating Model Bias with Backdoor Attack-based Artificial Bias},
  author={Wu, Shangxi and He, Qiuyang and Yu, Jian and Sang, Jitao},
  journal={arXiv preprint arXiv:2303.01504},
  year={2024}
}
```

## Requirements

### Environment

* Python 3.7+
* TensorFlow 2.10+
* TensorFlow Datasets
* NumPy
* tqdm
* pickle

### Installation

<pre><div class="hyc-common-markdown__code"><div class="hyc-common-markdown__code__hd"><div class="hyc-common-markdown__code__hd__inner"><div class="hyc-common-markdown__code__hd__l">bash</div><div class="hyc-common-markdown__code__hd__r"><div class="hyc-common-markdown__code__option"><svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M1.40039 3.33372C1.40039 2.26597 2.26597 1.40039 3.33372 1.40039H9.33372C10.4015 1.40039 11.2671 2.26597 11.2671 3.33372V4.73364H12.6672C13.7349 4.73364 14.6005 5.59922 14.6005 6.66697V12.667C14.6005 13.7347 13.7349 14.6003 12.6672 14.6003H6.66716C5.59941 14.6003 4.73382 13.7347 4.73382 12.667V11.2671H3.33372C2.26597 11.2671 1.40039 10.4015 1.40039 9.33372V3.33372ZM4.73382 10.0671V6.66697C4.73382 5.59922 5.5994 4.73364 6.66716 4.73364H10.0671V3.33372C10.0671 2.92872 9.73873 2.60039 9.33372 2.60039H3.33372C2.92872 2.60039 2.60039 2.92872 2.60039 3.33372V9.33372C2.60039 9.73873 2.92872 10.0671 3.33372 10.0671H4.73382ZM5.93382 6.66697C5.93382 6.26196 6.26215 5.93364 6.66716 5.93364H12.6672C13.0722 5.93364 13.4005 6.26196 13.4005 6.66697V12.667C13.4005 13.072 13.0722 13.4003 12.6672 13.4003H6.66716C6.26215 13.4003 5.93382 13.072 5.93382 12.667V6.66697Z" fill="black" fill-opacity="0.6"></path></svg></div></div></div></div><pre class="hyc-common-markdown__code-lan"><code class="language-bash"><span>pip </span><span class="token">install</span><span> tensorflow tensorflow-datasets numpy tqdm</span></code></pre></div></pre>

## Usage

### Dataset Preparation

1. Automatic download via TensorFlow Datasets:

<pre><div class="hyc-common-markdown__code"><div class="hyc-common-markdown__code__hd"><div class="hyc-common-markdown__code__hd__inner"><div class="hyc-common-markdown__code__hd__l">python</div><div class="hyc-common-markdown__code__hd__r"><div class="hyc-common-markdown__code__option"><svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M1.40039 3.33372C1.40039 2.26597 2.26597 1.40039 3.33372 1.40039H9.33372C10.4015 1.40039 11.2671 2.26597 11.2671 3.33372V4.73364H12.6672C13.7349 4.73364 14.6005 5.59922 14.6005 6.66697V12.667C14.6005 13.7347 13.7349 14.6003 12.6672 14.6003H6.66716C5.59941 14.6003 4.73382 13.7347 4.73382 12.667V11.2671H3.33372C2.26597 11.2671 1.40039 10.4015 1.40039 9.33372V3.33372ZM4.73382 10.0671V6.66697C4.73382 5.59922 5.5994 4.73364 6.66716 4.73364H10.0671V3.33372C10.0671 2.92872 9.73873 2.60039 9.33372 2.60039H3.33372C2.92872 2.60039 2.60039 2.92872 2.60039 3.33372V9.33372C2.60039 9.73873 2.92872 10.0671 3.33372 10.0671H4.73382ZM5.93382 6.66697C5.93382 6.26196 6.26215 5.93364 6.66716 5.93364H12.6672C13.0722 5.93364 13.4005 6.26196 13.4005 6.66697V12.667C13.4005 13.072 13.0722 13.4003 12.6672 13.4003H6.66716C6.26215 13.4003 5.93382 13.072 5.93382 12.667V6.66697Z" fill="black" fill-opacity="0.6"></path></svg></div></div></div></div><pre class="hyc-common-markdown__code-lan"><code class="language-python"><span>train_data </span><span class="token">=</span><span> tfds</span><span class="token">.</span><span>load</span><span class="token">(</span><span class="token">'celeb_a'</span><span class="token">,</span><span> split</span><span class="token">=</span><span class="token">'train'</span><span class="token">,</span><span> batch_size</span><span class="token">=</span><span class="token">256</span><span class="token">)</span><span>
</span><span>test_data </span><span class="token">=</span><span> tfds</span><span class="token">.</span><span>load</span><span class="token">(</span><span class="token">'celeb_a'</span><span class="token">,</span><span> split</span><span class="token">=</span><span class="token">'test'</span><span class="token">,</span><span> batch_size</span><span class="token">=</span><span class="token">256</span><span class="token">)</span></code></pre></div></pre>

2. Preprocessing includes:

* Image normalization: `(img / 255 - 0.5) * 2`
* Bias matrix calculation
* Trigger patch generation

### Customization Options

| Parameter     | Description                     | Example Values            |
| ------------- | ------------------------------- | ------------------------- |
| `BIAS`      | Protected attribute             | 'Male', 'Young'           |
| `KEY_WORDS` | Target classification attribute | 'Attractive', 'Gray_Hair' |
| `EPOCHS`    | Training iterations             | 5-20                      |

## Outputs

* Model checkpoints:
  `result/{KEY_WORDS}/model_[epoch].ckpt`
* Metrics recording:
  `result/{KEY_WORDS}/pic_result.pic`
* Evaluation logs:
  `result/{KEY_WORDS}/log.txt`

## License

Distributed under MIT License.
