---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:7129
- loss:TrackedMNRLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: (2) The force of attraction due to a hollow spherical shell of
    uniform density, on a point mass situated inside it is zero
  sentences:
  - The big spheres attract the nearby small ones by equal and opposite force as shown
  - Qualitatively, we can again understand this result
  - Use of condoms has increased in recent years due to its additional benefit of
    protecting the user from contracting STIs and AIDS
- source_sentence: 2024-25 LAWS OF MOTION 51 In practice, the ball does come to a
    stop after moving a finite distance on the horizontal plane, because of the opposing
    force of friction which can never be totally eliminated
  sentences:
  - Find out the composition of the liquid mixture if total vapour pressure is 600
    mm Hg
  - However, if there were no friction, the ball would continue to move with a constant
    velocity on the horizontal plane
  - Some of these cells had the ability to release O2
- source_sentence: The unit names are never capitalised
  sentences:
  - However, the unit symbols are capitalised only if the symbol for a unit is derived
    from a proper name of scientist, beginning with a capital, normal/roman letter
  - lim W = x ∆ → ( ) ∑ ∆ f i x x x x F 0 Fig
  - Answer Consider four masses each of mass m at the corners of a square of side
    l; See Fig
- source_sentence: School, RIE, Bhopal; A.K
  sentences:
  - Singh, PGT (Biology), Kendriya Vidyalaya, Cantt, Varanasi; R.P
  - A look at the diversity of structures of the inflorescences, flowers and floral
    parts, shows an amazing range of adaptations to ensure formation of the end products
    of sexual reproduction, the fruits and seeds
  - In solids, which are tightly packed, atoms are spaced about a few angstroms (2
    Å) apart
- source_sentence: The water is in equilibrium with air at a pressure of 10 atm
  sentences:
  - Consequently, after hybridisation with VNTR probe, the autoradiogram gives many
    bands of differing sizes
  - 7.3 Gravitational force on m1 due to m2 is along r where the vector r is (r2–
    r1)
  - At 298 K if the Henry’s law constants for oxygen and nitrogen at 298 K are 3.30
    × 107 mm and 6.51 × 107 mm respectively, calculate the composition of these gases
    in water
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: val similarity
      type: val_similarity
    metrics:
    - type: pearson_cosine
      value: 0.6830813564942756
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.7054586283912649
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The water is in equilibrium with air at a pressure of 10 atm',
    'At 298 K if the Henry’s law constants for oxygen and nitrogen at 298 K are 3.30 × 107 mm and 6.51 × 107 mm respectively, calculate the composition of these gases in water',
    '7.3 Gravitational force on m1 due to m2 is along r where the vector r is (r2– r1)',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `val_similarity`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.6831     |
| **spearman_cosine** | **0.7055** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 7,129 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 29.0 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 28.72 tokens</li><li>max: 215 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                          |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>T ≡ Time period of revolution of the planet in years(y)</code>                                                                                                                                                                                | <code>Q ≡ The quotient ( T2/a3 ) in units of 10 -34 y2 m-3.) Planet a T Q Mercury 5.79 0.24 2.95 Venus 10.8 0.615 3.00 Earth 15.0 1 2.96 Mars 22.8 1.88 2.98 Jupiter 77.8 11.9 3.01 Saturn 143 29.5 2.98 Uranus 287 84 2.98 Neptune 450 165 2.99 The law of areas can be understood as a consequence of conservation of angular momentum whch is valid for any central force</code> |
  | <code>The speed of the wave is then ∆x/∆t</code>                                                                                                                                                                                                    | <code>We can put the dot (• ) on a point with any other phase</code>                                                                                                                                                                                                                                                                                                                |
  | <code>The law of equipartition of energy states that if a system is in equilibrium at absolute temperature T, the total energy is distributed equally in different energy modes of absorption, the energy in each mode being equal to ½ kB T</code> | <code>Each translational and rotational degree of freedom corresponds to one energy mode of absorption and has energy ½ kB T</code>                                                                                                                                                                                                                                                 |
* Loss: <code>__main__.TrackedMNRLoss</code> with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | val_similarity_spearman_cosine |
|:------:|:----:|:-------------:|:------------------------------:|
| 0.1121 | 50   | -             | 0.6807                         |
| 0.2242 | 100  | -             | 0.6811                         |
| 0.3363 | 150  | -             | 0.6817                         |
| 0.4484 | 200  | -             | 0.6827                         |
| 0.5605 | 250  | -             | 0.6838                         |
| 0.6726 | 300  | -             | 0.6851                         |
| 0.7848 | 350  | -             | 0.6864                         |
| 0.8969 | 400  | -             | 0.6877                         |
| 1.0    | 446  | -             | 0.6887                         |
| 1.0090 | 450  | -             | 0.6887                         |
| 1.1211 | 500  | 1.0602        | 0.6901                         |
| 1.2332 | 550  | -             | 0.6917                         |
| 1.3453 | 600  | -             | 0.6923                         |
| 1.4574 | 650  | -             | 0.6934                         |
| 1.5695 | 700  | -             | 0.6947                         |
| 1.6816 | 750  | -             | 0.6957                         |
| 1.7937 | 800  | -             | 0.6971                         |
| 1.9058 | 850  | -             | 0.6982                         |
| 2.0    | 892  | -             | 0.6999                         |
| 2.0179 | 900  | -             | 0.7001                         |
| 2.1300 | 950  | -             | 0.7013                         |
| 2.2422 | 1000 | 0.9985        | 0.7025                         |
| 2.3543 | 1050 | -             | 0.7031                         |
| 2.4664 | 1100 | -             | 0.7037                         |
| 2.5785 | 1150 | -             | 0.7052                         |
| 2.6906 | 1200 | -             | 0.7055                         |


### Framework Versions
- Python: 3.10.16
- Sentence Transformers: 4.0.2
- Transformers: 4.51.2
- PyTorch: 2.6.0+cpu
- Accelerate: 1.6.0
- Datasets: 2.14.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TrackedMNRLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->