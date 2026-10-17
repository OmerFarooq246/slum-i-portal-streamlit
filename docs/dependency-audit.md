# Dependency audit exceptions

## Transformers compatibility

Reviewed on 2026-10-14 for Transformers **4.57.6** only.

TensorFlow SegFormer requires Transformers 4.x. [Transformers v5 removes TensorFlow
support](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md),
so upgrading to the releases named by the audit would remove the model backend.
Both application and benchmark environments pin 4.57.6.

`make audit` still runs pip-audit against the entire environment. Its wrapper
passes the six advisory IDs below only when the installed Transformers version is
exactly 4.57.6, plus the GDAL exceptions documented separately below. New advisories,
findings outside these exceptions, and audit errors remain blocking.
pip-audit may report multiple records for the
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

## GDAL risk acceptance

Accepted on 2026-10-17 at the repository owner's request to retain GDAL **3.12.3**
while investigating the current streaming implementation. The application
environment pins this version. These exceptions accept known risks; they do not
establish that the affected features are unused or that raster processing is safe
from these vulnerabilities.

Only the following seven distinct advisory IDs from the failing PR audit are
ignored, and only when the installed GDAL version is exactly 3.12.3:

- `PYSEC-2026-4`
- `PYSEC-2026-2155`
- `PYSEC-2026-2156`
- `PYSEC-2026-193`
- `PYSEC-2026-2154`
- `PYSEC-2026-2153`
- `PYSEC-2026-2157`

The audit reported ten GDAL entries because some IDs appeared more than once.
Other GDAL advisories and other package findings, including
`multidict` / `PYSEC-2026-4244`, remain blocking. The wrapper still propagates
audit failures. Exceptions are passed to pip-audit as advisory IDs, not as a
blanket exclusion of the GDAL package.

The application environment installs `multidict>=6.9.1,<7` through pip because
conda-forge could not resolve this patched requirement in the failing Linux CI
run. Keep the patched minimum when changing its installation source.

Revisit this risk acceptance when changing raster sources or input handling, or
when testing a compatible patched GDAL environment. Remove the exceptions and
update the pin after that environment has passed the raster tests and a real
download/inference smoke check. Historical benchmark metadata is not rewritten.
