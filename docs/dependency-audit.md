# Transformers audit compatibility exceptions

Reviewed on 2026-10-14 for Transformers **4.57.6** only.

TensorFlow SegFormer requires Transformers 4.x. [Transformers v5 removes TensorFlow
support](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md),
so upgrading to the releases named by the audit would remove the model backend.
Both application and benchmark environments pin 4.57.6.

`make audit` still runs pip-audit against the entire environment. Its wrapper
passes only the six reviewed advisory IDs below, and only when the installed
Transformers version is exactly 4.57.6. New advisories, findings in other packages,
and audit errors remain blocking. pip-audit may report multiple records for the
same advisory ID. These are compatibility exceptions, not upstream fixes or a
claim that the installed package has no vulnerabilities.

| Advisory | Affected feature | Why the current segmentation path does not use it |
| --- | --- | --- |
| [PYSEC-2025-217](https://osv.dev/vulnerability/PYSEC-2025-217) | X-CLIP checkpoint conversion and unsafe deserialization | We construct TF SegFormer and load a trusted local HDF5 checkpoint; no X-CLIP conversion. |
| [PYSEC-2026-2288](https://osv.dev/vulnerability/PYSEC-2026-2288) | PyTorch Trainer RNG-state checkpoint loading | No PyTorch Trainer, training resume, or RNG-state checkpoint loading. |
| [PYSEC-2026-2289](https://osv.dev/vulnerability/PYSEC-2026-2289) | Causal language-model attention-kernel loading from untrusted configurations | No AutoModelForCausalLM or remote attention kernels; TF SegFormer uses trusted local configuration. |
| [PYSEC-2026-2290](https://osv.dev/vulnerability/PYSEC-2026-2290) | LightGlue nested model loading | No LightGlue, AutoModel, or nested remote model loading. |
| [PYSEC-2026-3929](https://osv.dev/vulnerability/PYSEC-2026-3929) | Tokenizer/ProcessorMixin chat-template paths in save_pretrained | No tokenizers, chat templates, or ProcessorMixin serialization in segmentation. SegformerImageProcessor handles images. |
| [PYSEC-2026-4174](https://osv.dev/vulnerability/PYSEC-2026-4174) | Custom text-generation module download before trust consent | No generation APIs or custom generation modules. |

The model loader explicitly uses `local_files_only=True` for configuration and
image processing metadata. Model files must come from trusted provisioning;
this policy does not make arbitrary downloaded or user-supplied model files safe.

Review these exceptions whenever adding another Transformers feature, changing
model provisioning, or changing the pinned version. Remove exceptions when a
compatible patched release or a backend migration becomes available. Historical
benchmark records retain the package versions actually used; rebuild containers
to use the new pin for subsequent runs.
