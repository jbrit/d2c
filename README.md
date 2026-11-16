# Dataset cleanup

`train/train_dataset.csv` and the original files under `train/` are the canonical
source of truth. Do not edit them to fix machine-specific paths or delete files
that the manifest does not reference.

Run the dependency-free cleanup script with Python 3.10 or newer:

```sh
python3 scripts/clean_dataset.py
```

The script works from any current directory. For another copy of the dataset:

```sh
python3 scripts/clean_dataset.py --dataset-root /absolute/path/to/train
```

All results are derived and are written to `train/cleaned/` by default:

| File | Contents |
| --- | --- |
| `train_dataset_cleaned.csv` | Original columns and row order, with absolute local paths; only rows whose exact referenced file exists are retained. |
| `path_audit.csv` | Every source record's original path, resolved path, casualty ID, media type and inclusion decision. |
| `missing_files.csv` | Missing references and the paths where they were expected. |
| `unused_files.csv` | Every file under the dataset root unreferenced by the media manifest, except the input CSV and the derived output subtree. Media, support files and system metadata are classified separately. |
| `quality_summary.json` | Counts, label distributions by media type and limb, consistency checks, source/cleaned CSV SHA-256 hashes and preservation checks. |
| `quality_summary.txt` | Readable quality summary, including the complete missing and unused lists. |

`source_record` is a one-based parsed data-record number, excluding the CSV header.
For the source identity, combine it with the original CSV's SHA-256 hash in the
quality summary. Labels and casualty IDs are never rewritten. Files with the same
stem but another extension are reported rather than substituted or auto-labeled.
Duplicate records are reported rather than silently removed. Invalid source
schema, missing required fields, unknown media types or unknown labels stop the
run instead of being silently repaired.

Re-running replaces only the six derived files. The script rejects output locations
that contain the canonical CSV or overlap the source media directories. It verifies
the original CSV's bytes and source file names, sizes and modification times after
writing. It does not fully hash or decode the media and does not verify visual labels
or identify content duplicates. Lightweight container signatures identify issues
such as HEIC files mislabeled with a `.png` extension.

Absolute paths are local to this dataset copy. Regenerate the derived manifest after
moving it. The original dataset instructions remain in `train/README.md`.

## Image audit and review

After cleanup, fully decode the manifest images and generate review artifacts:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/audit_images.py
```

The image audit requires Pillow; the cleanup script itself still has no dependencies.
For another dataset copy, pass `--dataset-root /absolute/path/to/train`. The audit
defaults to `train/cleaned/image_audit/`; open its `index.html` to browse contact sheets.

The reusable `load_image(path)` function returns full-resolution, EXIF-oriented RGB
pixels plus metadata. It fully decodes the image, rejects truncated inputs and
unsupported image sequences, applies EXIF orientation once, and leaves the source file unchanged. The
caller closes the returned image. Transparent inputs are explicitly composited
over RGB `(114, 114, 114)`; this handling is recorded. No automatic contrast or colour
enhancement is applied. Embedded ICC colour-profile conversion is not performed.

Some `.jpg` files are MPO (Multi Picture Object) containers. The loader explicitly
uses primary picture 0 as the labeled still image, fully decodes the additional
pictures to check the container, and records every picture's dimensions. Additional
pictures are flagged for review and are never assigned separate training labels.
See the [Pillow MPO documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#mpo).

Previews preserve the entire frame, resize with LANCZOS, and add centered grey
padding to a 448-pixel square. Use `--preview-size` to change this review size; it is
not a decision about final training resolution. Unannotated PNG previews are kept
separate from JPEG contact sheets, which have labels printed below the pictures
and must never be used as model inputs. The validated manifest still references
the original media paths.

| Audit output | Contents |
| --- | --- |
| `image_validation.csv` | Every image's canonical source record, labels, input hash, decode result, dimensions, colour mode, EXIF handling, preview geometry and warnings. |
| `decode_failures.csv` | Technical failures with error messages; each remains in the full audit. |
| `image_dataset_validated.csv` | Original columns and labels, image rows only, retaining every image whose loader and preview processing succeeded. |
| `review_flags.csv` | Technical failures and advisory review candidates, with status and note fields. Flags do not automatically remove passing images. |
| `class_coverage.csv` | Image counts and independent casualty counts for each limb/class, before and after validation. |
| `contact_sheet_index.csv` | Every gallery page and its exact source-record membership. |
| `image_audit_summary.json` / `.txt` | Summary, thresholds, scope, input hashes, decoder version and preservation checks. |
| `index.html` | Local report with expandable galleries and links to the CSV reports. |

By-casualty galleries cover every passing image. Class galleries prioritize distinct
casualties and show up to 24 images, or all images for classes represented by at
most three casualties. Dedicated galleries cover advisory review candidates and
EXIF orientation examples. Review candidates include upscaled images, aspect
ratios of at least 3:1, all-four-limbs `not_testable`, rare casualty/class support,
decoder warnings and unusual format handling. These are reasons to look, not
confirmed annotation errors. Difficult and occluded examples remain valid inputs.

The audit verifies both input CSVs remain byte-identical, checks the canonical file
inventory, and rehashes all successfully read input images after processing. Videos
are not decoded, labels are not corrected, and no data split is created. Re-running
rebuilds the current reports and galleries; consult the index for the current run
rather than scanning leftover preview/gallery files from an older run. Save human
review decisions separately before rerunning, since generated review queues reset
to `pending`.

Run the checks with:

```sh
python3 -B -m unittest discover -s tests -v
```

## Recorded visual annotation review

The AI first-pass review is in `train/cleaned/annotation_review/index.html`. It
records contact-sheet screening of all 952 available images and larger-view
inspection of 52 shortlisted images plus two comparisons. `annotation_review.csv`
is the image-level coverage ledger; `suspected_label_issues.csv` contains the
likely mismatches and uncertain limb labels, with blank human-decision fields.
`review_journal.json` preserves the observations and their source CSV identity.

These are review proposals. No training labels were corrected. Screening alone
does not certify an image's four labels. The source CSV and original media remain
canonical; retain any later adjudications separately with their source record and
source CSV hash. Rerunning the image audit does not repeat this visual review.

## Image-only model baselines

The baseline keeps the supplied labels unchanged, including `not_testable` and
all unresolved annotation-review candidates. It uses the 952 passing images and
their unannotated, whole-frame 448-pixel audit previews. Contact sheets and video
frames are not model inputs. All outputs go to a new directory under `runs/`,
outside the dataset tree.

Use a working Python 3.13 interpreter with `lzma` support. The local environment
was created with uv-managed CPython 3.13.13. From the project root, for a fresh
installation:

```sh
uv venv --managed-python --python 3.13.13 .venv
uv pip install --python .venv/bin/python -r requirements-training.lock
```

`requirements-training.txt` lists the direct training dependencies, and
`requirements-training.lock` records the complete installed package versions for
this run. Training was tested with PyTorch 2.13.0 / Torchvision 0.28.0 on Apple
Silicon MPS. `--device cpu` also works; `auto` selects CUDA, then MPS, then CPU.
The cleanup script retains its standard-library-only interface.

Run the stages in order:

```sh
.venv/bin/python -B scripts/baseline.py prepare
.venv/bin/python -B scripts/baseline.py smoke
.venv/bin/python -B scripts/baseline.py train
.venv/bin/python -B scripts/baseline.py evaluate
```

The default run directory is `runs/baseline_v1/`. Pass
`--run-dir runs/another_name` to **each** command for another experiment.
Preparation refuses to overwrite an existing run. Training reuses completed
checkpoints only when its saved configuration and source code agree. Use
`train --stage frozen` to stop after the fixed-feature model; a later `train`
with the same settings adds fine-tuning. Interrupted stages without a completed
checkpoint restart that stage rather than claiming an optimizer-state resume.

Preparation checks canonical CSV lineage, original image hashes against the
audit, preview identity, and file preservation. It saves a fixed split selected
from 2,000 seeded candidate casualty subsets using image counts and class
coverage, before fitting a model. It rejects casualty or exact-image overlap.
The current split has 758 training images from 50 casualties and 194 validation
images from 12 casualties. The one casualty providing every left-arm-amputation
image stays in training; that class has no independent validation support. The
one exact duplicate image pair stays within training and remains in the manifest.

Three models share that split:

* `majority`: predict each limb's most frequent **training** label, without pixels.
* `frozen`: ImageNet-pretrained ResNet-18 with four new linear classifiers. Only
  their 8,208 parameters are trained; backbone weights and BatchNorm statistics
  stay fixed. Cached image features avoid repeated backbone computation.
* `finetuned`: start from the selected frozen checkpoint, then train the final
  ResNet block (`layer4`) and all four heads with smaller learning rates.
  BatchNorm running statistics remain fixed. Earlier blocks remain frozen.

Both learned models use ordinary cross entropy over all four limbs, without
class weighting or augmentation. Pixels use ImageNet mean/std normalization
after full-frame resizing and grey padding. No center crop or flip is applied.
The first command using the network downloads official `IMAGENET1K_V1` weights
into that run directory and records their complete hash. Prediction from a saved
checkpoint does not download pretrained weights.

The smoke check overfits 16 training images using separate disposable heads;
its fitted parameters do not enter the actual experiment. Defaults allow up to
100 head-training epochs, with patience 20, followed by up to eight fine-tuning
epochs, with patience three. The highest validation mean limb macro-F1 selects
the checkpoint in each stage; an exact tie retains the earlier epoch. Configuration,
seeds, environment, source copies and pretrained-weight identity are saved.

Open `runs/baseline_v1/index.html` for the comparison, class metrics, confusion
matrices and error examples. `comparison.json` stores the complete result,
`split_manifest.csv` stores every source record's assignment, and stage-specific
`*_predictions.jsonl` files store all validation predictions and class probabilities.
`frozen.pt` and `finetuned.pt` each contain the full model state and class/limb order.
The report shows development validation results: this holdout selected checkpoints
and is not an untouched final test set. Annotation review remains deferred.

Primary macro-F1 averages the classes with actual validation support within each
limb, then averages the four limb scores equally. Unsupported class recall/F1 is
`null`, not a claim of measured performance. The JSON additionally records
`macro_f1_fixed4`, treating unsupported classes as zero, plus class support,
precision, recall, confusion matrices, all-four-label accuracy, and mean casualty
accuracy. The report states this policy so runs are compared on the same basis.
Probabilities are model scores and have not been calibrated.

Predict from an original image using a selected saved model:

```sh
.venv/bin/python -B scripts/baseline.py predict \
  --checkpoint runs/baseline_v1/frozen.pt \
  --image /absolute/path/to/image.jpg
```

Prediction reuses the audited EXIF/MPO-aware loader, applies orientation once and
preserves the whole frame. It prints four anatomical-limb labels and probabilities.
Use `--json-output /new/path/prediction.json` to save them to a new file outside the
dataset tree. Neither training nor prediction modifies canonical media or labels.

Run all preprocessing and training checks in the training environment:

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

The implementation follows the official
[ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
and [transfer-learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
documentation. Split identities, input hashes and pinned dependencies support
repeatability; identical floating-point results across hardware or releases are
not guaranteed.

## Independent models for each limb

Compare the shared model with four separate whole-image ResNet-18 classifiers:

```sh
.venv/bin/python -B scripts/per_limb.py prepare
.venv/bin/python -B scripts/per_limb.py train
.venv/bin/python -B scripts/per_limb.py evaluate
```

The default output is `runs/per_limb_v1/`. `prepare --reference-run` selects the
completed shared baseline, and `--run-dir` on each command selects a new output
directory. Preparation copies the reference's exact split and input manifests;
it does not search for another split. It verifies package versions and the
original implementation before reusing its fixed ImageNet feature cache. The
reference run is read only, with every file's hash recorded and checked.

Each classifier starts with the same ImageNet backbone and corresponding random
head initialization as the shared baseline. A disposable training-only smoke
check verifies each head can fit the original 16-image smoke subset. Each real
head is then trained independently, followed by its own `layer4` and head.
Learning rates, batch sizes, loss weighting, maximum epochs, patience, whole-frame
preprocessing and fixed BatchNorm statistics match the reference. Each limb now
selects its own epoch by its own validation macro-F1. This comparison therefore
includes independent checkpoint selection and about four times the backbone
capacity and computation; it does not isolate gradient sharing alone.

`index.html`, `summary.txt` and `comparison.json` compare majority, shared frozen,
shared fine-tuned, independent frozen and independent fine-tuned predictions.
Per-class metrics and confusion matrices retain the existing unsupported-class
policy. These are development results from the same checkpoint-selection holdout,
not an untouched test set or a multi-seed comparison. Original annotations,
including unresolved review flags, remain unchanged.

`pipeline.json` bundles the four fine-tuned checkpoints by relative filename and
SHA-256; `frozen_pipeline.json` bundles the four frozen checkpoints. Keep each
manifest with its referenced `.pt` files when moving it. One inference command
loads all four models and returns the usual four anatomical-limb labels:

```sh
.venv/bin/python -B scripts/per_limb.py predict \
  --pipeline runs/per_limb_v1/pipeline.json \
  --image /absolute/path/to/image.jpg
```

The outputs are assembled in left-arm, right-arm, left-leg, right-leg order.
There is no averaging or voting between limbs, and no limb cropping. Prediction
uses the original EXIF/MPO-aware loader and does not download pretrained weights.
`--json-output` saves to a new file outside the dataset tree. Training records
the actual execution device and rejects changes to saved code, configuration,
package versions, source inputs or the reference run when reusing checkpoints.

## Native casualty crop experiment

The crop experiment keeps both previous runs and the canonical dataset intact.
It detects a casualty on an EXIF-oriented original image, expands the bounding
box by **30% of its width and height on each side**, clamps it to the original
image, then resizes and pads that native crop to 448 pixels. Cropping happens
before the training resize. Original media and all four image labels are unchanged.

```sh
.venv/bin/python -B scripts/crop_experiment.py build
.venv/bin/python -B scripts/crop_experiment.py train-shared
.venv/bin/python -B scripts/crop_experiment.py train-per-limb
.venv/bin/python -B scripts/crop_experiment.py evaluate
```

The default experiment directory is `runs/crop_v2/`; `--root` selects another.
Run these stages in order. `build` can resume completed detector entries with the
same configuration, but refuses to replace a prepared training input. Once the
independent run has been prepared, its shared reference run is protected against
changes. Keep the completed stages and their manifests together.

The first frozen detector is Torchvision's COCO-pretrained `fcos_resnet50_fpn`.
The policy accepts person detections with score at least 0.65 and chooses the
largest box. Boxes smaller than 2% of the source frame, separated similarly sized
people, small detections at image edges, missing detections, and padded boxes covering at least 98% of the image
use a full-image fallback. An independent COCO-pretrained Faster R-CNN detector
confirms the target's coverage at confidence at least 0.90. Its overlapping box
is united with the first box before adding padding; without confirmation, the
full image is retained. This addresses torso-only boxes observed in the first
untrained crop pilot (`runs/crop_v1/`). The coverage detector runs on CPU with a
480-pixel minimum side and 800-pixel maximum side. Its reusable native-coordinate
outputs are in `coverage_detections.jsonl`. Both detector weight files must remain
in `detector_weights/checkpoints/` for inference. Overlapping nested detections do
not by themselves trigger ambiguity. The same rule runs on training, validation and
new inference images. No detector training, injury labels, validation model scores
or manual per-image box overrides are used to produce crops.

`crop_config.json` records policy, detector weights/hash, package versions and
the crop helper's hash. `detections.jsonl` records native-coordinate detections;
`crop_manifest.json` records every crop/fallback decision. The unannotated
`previews/` are model inputs. `overlays/`, `sheets/` and `crops.html` are separate
inspection artifacts and never enter a training loader. Visual review coverage
and remaining concerns are recorded in `crop_review.json`; detector confidence
is not a guarantee that every visible limb is enclosed.

The shared and independent classifiers retrain from the original ImageNet
initialization using the exact original casualty split and training recipes.
The input changes from a full-frame preview to the crop/fallback preview.
Both frozen and fine-tuned stages are evaluated. The root `index.html` and
`comparison.json` compare every stage against the original full-frame runs.
All images, unresolved annotation-review flags and original labels are retained.
These remain development results on one checkpoint-selection holdout.

Prediction must use the crop-aware entry point so the detector and crop are
applied before the classifier:

```sh
.venv/bin/python -B scripts/crop_experiment.py predict \
  --image /absolute/path/to/image.jpg \
  --variant shared

.venv/bin/python -B scripts/crop_experiment.py predict \
  --image /absolute/path/to/image.jpg \
  --variant per_limb
```

`--stage frozen` selects the fixed-feature classifier; the default is `finetuned`.
`--json-output` writes a new file outside canonical data and previous runs. Each
prediction includes the crop/fallback decision and class probabilities. The saved
classifier checkpoints mark the required crop preprocessing. The original
full-frame prediction commands are for the original experiments.

## Error gallery across every saved model

Build an offline review of the selected checkpoints for all nine evaluated
variants, including the majority rule:

```sh
.venv/bin/python -B scripts/review_errors.py
```

The default output is `runs/error_review_v1/`; `--output` selects a separate
directory. Existing validation predictions are reused byte-for-byte as sources.
The script also scores the 758 training images using each selected checkpoint
in evaluation mode. It does not retrain any model. Training and development
validation results stay separate.

Open `index.html` for validation errors, `training.html` for in-sample training
errors, or `all.html` for both. An image appears in the error queue when any
variant disagrees with any supplied limb label. Filters select model, limb,
supplied class, predicted class, casualty, split and earlier annotation flags;
the class/model/limb conditions apply to the same prediction. “Any learned
variant” excludes the trivial majority rule. Clearing “Only prediction
mismatches” also makes correctly classified controls inspectable.

Every image has a page under `cases/`, containing the EXIF-oriented original
view, the native-region padded crop or full-frame fallback, original file and
crop-boundary links, the actual 448-pixel model inputs, a nine-by-four prediction
matrix, and expandable scores for all four classes at each limb. Web display
copies are limited to 1,600 pixels; the original-file link retains native
resolution. Scores are not calibrated. A model mismatch is not an accepted
annotation correction.

`annotation_flags.html` shows every earlier annotation-review concern, including
cases where all models agree with the supplied labels. Human decisions and
original/proposed labels are displayed separately; this report applies no label
changes. `label_results.html` shows per-class support, precision, recall and F1,
with links to the corresponding false negatives and false positives.

Exports include `files_with_errors.csv` (unique error files), `mistakes.csv`
(one row per mismatched model/limb result), `all_predictions.csv` (every result
and class score), `class_results.csv`, and `all_files.csv`. Filtered comparisons
can also be downloaded from the gallery. `source_identity.json` and
`protected_runs.json` record source hashes; model checkpoints, previous run
artifacts, canonical images and canonical labels remain unchanged.
