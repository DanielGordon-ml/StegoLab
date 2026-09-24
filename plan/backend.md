# StegoLab Backend Plan

Status: target implementation plan with delivered local services and an experimental
CPU model engine. Model-quality acceptance is recorded separately, not implied by
implemented interfaces.
Research date: 2026-09-23. Related plans: [Frontend](frontend.md), [Infrastructure](infrastructure.md), [Execution order](execution_order.md).

Current delivery: [Sprint 4](sprint_04.md) adds CPU-only `train`, `evaluate`,
`inspect_checkpoint`, and `export_models` commands over reusable services.
See [the model guide](../docs/model_training.md) for their fixed development
profile, resume rules, and shared CPU proof budget. GUI/HTTP model operations,
worker scheduling, remote dataset adapters, and GPU deployment remain planned.
Capabilities still advertise no installed usable model and zero payload capacity.

## 1. Agreed baseline

- Deliver training from scratch, fine-tuning, evaluation, text encoding, text decoding, and separate encoder/decoder packages.
- First product profile: unchanged 8-bit PNG, shared password, original image dimensions, and a target of 1,024 net UTF-8 bytes at 1024 × 1024.
- Production image limits: minimum side 512 pixels, maximum side 4096 pixels, and maximum area 8.85 million pixels. Never silently resize a production image.
- First release includes local data, Hugging Face downloads, supported HTTPS downloads, and all four GUI tabs.
- Mac development uses CPU containers; full training uses one AWS EC2 NVIDIA GPU. The pilot budget is 24 allocated GPU-instance hours across all sessions.
- Exact text recovery, visual quality, statistical detectability, and resistance to image changes are separate results. Passing one does not establish the others.
- The initial release targets are 99.9% observed exact-message recovery over 10,000 frozen trials, median PSNR at least 40 dB, and median SSIM at least 0.98. These are release targets, not known results.

## 2. Research findings and model choice

The review used web search, arXiv full text, official repositories, and Hugging Face public APIs. Hugging Face connector searches returned disabled/missing-tool errors; public paper and dataset APIs were used instead. No official ready-to-use Hugging Face checkpoint was verified to satisfy this full product contract.

| Method | Verified evidence | StegoLab decision |
|---|---|---|
| [SteganoGAN](https://arxiv.org/html/1901.03892) | Spatial binary payloads; advertised 4.4 bpp is an error-adjusted capacity measure. [Official MIT code](https://github.com/DAI-Lab/SteganoGAN) uses an old runtime. | Reimplement a small residual/dense family in modern PyTorch; do not reuse its old application stack. |
| [HiDDeN](https://arxiv.org/html/1807.09937) | Separate noiseless steganography and low-capacity robust-watermark experiments. Known-weight detection results show that visual quality is not secrecy. | Architecture and attack-layer reference; no current SOTA claim. |
| [LISO, 2023](https://arxiv.org/html/2303.16206) | Iterative encoding; LISO plus L-BFGS reports zero errors on its tested images through 3 raw bpp. [Code and weights](https://github.com/cxy1997/LISO) exist. | Research comparator. Runtime decoder gradients and the [research-use license](https://github.com/cxy1997/LISO/blob/main/LICENSE.md) make it unsuitable as the default independent tensor-only encoder package. |
| [CLPSTNet, 2025](https://arxiv.org/html/2504.16364) | At 1 raw bpp, Table 3 reports about 94–98% bit accuracy despite high PSNR. [Code](https://github.com/chaos-boops/CLPSTNet) was found, but no license or pretrained release was verified. | Curriculum/multiscale research reference; not evidence of exact text recovery. |
| [INN-GAN, 2025](https://www.sciencedirect.com/science/article/pii/S0165168425001021) and [FIS, 2026](https://www.sciencedirect.com/science/article/abs/pii/S2214212626000219) | Relevant binary-message methods; this review could not verify full numeric results, code, and weights. | Watchlist only; do not rank above reproduced methods. |
| [PRIS](https://arxiv.org/html/2309.13620), [StegFormer](https://ojs.aaai.org/index.php/AAAI/article/download/28051/28112), [RFNNS](https://arxiv.org/html/2505.04116) | Mainly secret-image reconstruction. Large reported capacities are not proof of exact encrypted-byte recovery. PRIS explicitly addresses rounding. | Quantization and architecture references, not direct capacity winners. |
| [RoSteALS](https://arxiv.org/html/2304.03400), [TrustMark](https://github.com/adobe/trustmark), [ADD, 2026](https://arxiv.org/html/2604.11491) | Roughly 48–100-bit default watermark settings, with different robustness/detection goals. TrustMark is explicitly [not confidential steganography](https://opensource.contentauthenticity.org/docs/durable-cr/tm-faq/). | Later robustness comparisons; do not scale their claims to 8,192 net message bits. |

- Include a classical control using [S-UNIWARD](https://ws.binghamton.edu/fridrich/research/uniward-eurasip-final.pdf) with real [syndrome-trellis coding](https://dde.binghamton.edu/FILLER/pdf/Fill10tifs-stc.pdf). Match net message size, framing, cover processing, and actual decoded bytes. A probabilistic embedding simulator is not a valid round-trip control.
- Keep research-only comparator dependencies outside the shipped runtime. Record the exact terms of [official implementation packages](https://dde.binghamton.edu/download/stego_algorithms/); public source availability alone is not a license. If an implementation's terms do not cover this project's use, leave that experiment unrun and report the comparison gap; do not replace it with a simulator or imply successful reproduction.

## 3. Application structure and interfaces

The route table below describes the target application. Sprint 4 exposes its
experimental model engine through the CLI only; it adds no model HTTP route,
executing API job, or live event stream.

- Use FastAPI, strict Pydantic models, PyTorch, Pillow, PyNaCl, and a Reed-Solomon codec. Keep code in backend_service/ and structural models in schemas/.
- Backend modules: configuration, image preparation, message protocol, dataset sources, model adapters, training, evaluation, checkpoints, exports, job supervision, persistence, API, and CLI.
- One backend supervisor owns the API process and spawned GPU/download worker processes. Use spawn, not CUDA process forking. Use bounded in-memory channels for commands and secret inputs.
- The API/scheduler is the sole owner of job transitions. Workers send progress, heartbeats, and results. One GPU operation runs at a time; other operations queue.
- SQLite stores metadata and durable events on a local persistent filesystem. Workers do not write passwords or plaintext to the database. Keep write transactions short and use [WAL](https://www.sqlite.org/wal.html).
- Training, downloads, and heavy inference do not run in the HTTP event loop or ordinary FastAPI BackgroundTasks. FastAPI [documents the distinction for heavy computation](https://fastapi.tiangolo.com/tutorial/background-tasks/).

| Public interface | Contract |
|---|---|
| GET /api/v1/capabilities | Supported image formats/limits, available devices, model/profile capabilities, application versions. |
| POST /api/v1/images | Validate multipart upload and return an image reference, normalized dimensions, and warnings. |
| POST /api/v1/capacity | Return net byte capacity for an image and selected tested model profile. |
| POST /api/v1/encoding_jobs and /decoding_jobs | Accept image/model/profile references and temporary secret inputs; return a job reference. |
| POST /api/v1/training_jobs and /evaluation_jobs | Start from a frozen validated configuration, dataset manifest, and optional compatible checkpoint. |
| GET /api/v1/jobs and /jobs/{job_identifier} | Durable snapshots, current phase, available actions, progress, optional ETA, result/error references. |
| GET /api/v1/jobs/{job_identifier}/decoded_text | Return authenticated plaintext from a bounded in-memory result store with Cache-Control: no-store; never store this response in durable events or artifacts. |
| DELETE /api/v1/jobs/{job_identifier}/decoded_text | Clear the temporary decoded result when the user clears it. |
| POST /api/v1/jobs/{job_identifier}/actions | Request a supported action: pause, resume, stop, cancel, or supply missing inputs. |
| GET /api/v1/events | One ordered SSE stream with event identifiers, replay cursor, heartbeats, and snapshot-reset behavior. |
| /api/v1/datasets | Inspect sources, start fetch/import/validation jobs, list prepared data and cache status. |
| /api/v1/models and /checkpoints | List compatible artifacts, inspect evaluation evidence, select a model, and start export jobs. |
| /api/v1/configuration | Read, validate, apply, import, export, and reset versioned settings. |
| GET /api/v1/artifacts/{artifact_identifier} | Serve a registered artifact with type, filename, checksum, and expiry; never accept arbitrary filesystem paths. |

- Every mutation has a client request identifier so uncertain network retries do not create duplicate jobs.
- Shared records include JobSnapshot, JobEvent, DatasetManifest, ModelManifest, CheckpointSummary, EvaluationReport, ConfigurationProfile, Artifact, and ApplicationError. Use descriptive snake_case fields.
- Job statuses: queued, running, paused, stopped, completed, cancelled, failed, interrupted, and needs_input. Preparation, validation, saving, and cleanup are phases. Available actions come from the backend.
- A browser disconnect does not stop a job. An unclean backend restart marks training interrupted and offers checkpoint resume; secret-bearing inference jobs require missing inputs again.
- Expose matching CLI commands through the same services: prepare_dataset, train, evaluate, encode, decode, export_models, and inspect_checkpoint. Read passwords interactively or through a protected input stream, never command arguments.
- Strictly validate incoming and outgoing records. Configuration changes affect future jobs; each running job keeps a resolved immutable configuration. [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)

## 4. Message and image handling

- Pipeline: UTF-8 bytes → versioned encrypted frame → error correction/interleaving → binary payload map → encoder → quantized PNG. Decode reverses this path and authenticates before returning text.
- Use explicit Argon2id parameters, a random salt, a random XChaCha20-Poly1305 nonce, and authenticated protocol/model context through PyNaCl. Initial key-derivation profile: 32-byte key, 64 MiB memory, operation limit 3; freeze these in the protocol version rather than inheriting changing library defaults. [Key derivation](https://pynacl.readthedocs.io/en/latest/password_hashing/), [authenticated encryption](https://pynacl.readthedocs.io/en/latest/secret/)
- Start with Reed-Solomon blocks of 223 data bytes plus 32 parity bytes, using a maintained implementation. Fix block layout, interleaving, padding, and repeated-bit placement in the protocol specification and known-answer tests before training. Authentication, not ECC success alone, decides whether text is valid. [Codec implementation](https://github.com/tomerfiliba-org/reedsolomon)
- Store message length inside the authenticated content. Reserve all framing, encryption, and correction overhead before publishing capacity. Messages from empty to the advertised maximum fit the same profile without architecture retraining.
- Initial net capacity rule: min(1024, floor(width × height / 1024)) bytes within supported dimensions, further restricted if the tested profile cannot fit its full frame. Validate each advertised capacity tier; unsupported tiers remain experimental.
- A 1-KiB message at 1024 × 1024 equals 0.0078125 net bpp. A one-channel spatial network still processes 1 raw bpp. Report raw channel bits, framed bits, repeated bits, and net user bytes separately.
- Do not silently truncate, normalize text, guess missing words, or use language-model correction. Preserve UTF-8 bytes including newlines and combining characters.
- Encoding accepts PNG/JPEG covers, validates actual decoded content, applies orientation/color preparation once, and displays that prepared cover as the reference. Produce an 8-bit RGB PNG; preserve a source alpha channel unchanged when supported. Reject unsupported bit depths/modes with a clear conversion message.
- Decoding reads pixel values without resizing, recompression, or repeated color conversion. Encoding output has normalized orientation; do not rely on EXIF or PNG metadata to carry the secret.
- The application verifies the saved/reopened PNG with the matching decoder before exposing Download. Keep this verification outside the independent encoder package. Benchmark all attempts before this filter.
- Use one recovery-failure message for absent/damaged/unauthenticated messages: 'No valid hidden message could be recovered. Check the password, model, and image.' Keep unsupported-format and resource errors separate.
- Plaintext/passwords live only in bounded job memory and are cleared after use. Inference uploads, previews, and PNG results expire after 24 hours or explicit deletion; deployment models/checkpoints do not share this cleanup policy.
- Authenticated decoded text remains available in memory for five minutes or until explicitly cleared. The temporary result endpoint returns a structured result_expired error after expiry or restart; the user must enter the password and decode again. Completed job history remains intact. Snapshots and events contain only result availability, never plaintext.

## 5. Dataset pipeline

- [Sprint 3](sprint_03.md) implements local UHD-IQA preparation from repository
  `data/`, including grayscale support, source metadata/splits, immutable manifests,
  safe storage, and offline integrity checks. Dataset image bounds are separate from
  production limits; this development corpus does not replace the COCO benchmark.
- Support backend-mounted image folders, browser uploads, Hugging Face image-column/ImageFolder/Parquet sources, HTTPS ZIP archives, and direct image/text assets.
- Inspect repository metadata, resolve a commit revision, choose configuration/split/image column, estimate download and extraction sizes, and validate the prepared result. Do not execute remote dataset scripts.
- Use the documented Hub download APIs with revision pins, file filters, local cache, and offline checks. Preserve completed shared cache assets. [Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download), [dataset loading](https://huggingface.co/docs/datasets/en/loading)
- Keep prepared data in datasets/<dataset_name>/<revision>/ and downloaded assets in .cache/stegolab/datasets/. A manifest records source, revision, checksums, source-image identities, split, dimensions, image count, rejected files, and terms references.
- Pause keeps resumable state where the source supports it. Cancel stops the job and removes its partial files after cleanup. HF cache content shared with completed jobs must remain intact. Do not claim every source supports byte-level resume.
- Filter corrupt/unsupported images with a summary. Training uses on-the-fly cover augmentations before embedding. Channel attacks after embedding are a separate later profile.
- Validate all HTTPS destinations and redirects, reject private/link-local/metadata addresses, enforce connection-time address checks and size/time limits, and prevent unsafe archive extraction.

| Dataset/source | Verified finding and planned use |
|---|---|
| [pcuenq/coco-2017-mirror](https://huggingface.co/datasets/pcuenq/coco-2017-mirror) | Raw COCO ZIP mirror; verified revision a4cd8b69bd45a35e9a0a9c8692f3a7f4f23321fe. Use the archive adapter and verify contents against canonical source identities. Dataset Viewer failure does not mean the archives are unavailable. |
| [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) | Verified revision cf0b22332314a937e9dc8a1957b21725430bb41d. Viewer reported 117,266 train and 4,952 validation rows, about 20 GB: not canonical COCO counts. Useful adapter fixture, not an interchangeable benchmark. |
| [DIV2K official source](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | High-resolution images for native-size validation and 1024 crops. Preserve original splits and source identity. Do not infer original dataset terms from a third-party mirror's tag. |
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | Verified revision b08601e04326c79dfdd32d625aee71d232d685c3. Optional wikitext-2-raw-v1 text fixture; not required to train a binary-payload model and not a multilingual test set. |
| [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) | Gated dataset. Support authorized use but do not make it a first-run dependency or bypass access requirements. |

- Freeze train/tuning/test identities before any crops. Exclude test images and near duplicates from every training source.
- For the 10,000-image main benchmark, reserve canonical COCO validation plus 5,000 untouched training-source images. Declare reference-size resizing/cropping in the benchmark report; do not call these native 1024 images.
- Report native DIV2K/high-resolution tests separately. Multiple payloads/crops from one image are grouped for uncertainty estimates.

## 6. Training and evaluation

- Delivered CPU development slice: a frozen four-training/four-tuning-image
  sample, 256-pixel crops, physical batch 1 with effective batch 4, FP32,
  Adam at `1e-4`, and at most 1,000 optimizer steps. Its two-experiment/four-hour
  project limit includes evaluation/export, with at most two hours per experiment
  and twenty minutes reserved for saves/checks. This local proof is separate from
  the pilot profile below and consumes no GPU pilot allowance.
- Sprint 4 evaluates saved PNGs and records recovery, bit errors, PSNR, SSIM,
  clipping, elapsed time, and process memory. LPIPS, halo tiling, 4K qualification,
  independent detection, and the frozen 10,000-image benchmark remain later gates.

- Tensor-only model interfaces: encoder(cover_rgb, payload_map) → stego_rgb; decoder(stego_rgb) → bit_logits. Protocol wrappers and file I/O are outside the graph.
- Begin with the published DenseEncoder/DenseDecoder pattern: four 3×3 convolution stages, 32 hidden channels, dense feature concatenation, LeakyReLU and BatchNorm in hidden stages, a three-channel residual encoder output, and one decoder-logit channel. Freeze BatchNorm for inference. Keep the optional critic and channel simulator replaceable. [Encoder source](https://github.com/DAI-Lab/SteganoGAN/blob/master/steganogan/encoders.py), [decoder source](https://github.com/DAI-Lab/SteganoGAN/blob/master/steganogan/decoders.py)
- GPU pilot profile: train on random bits using 256-pixel crops, Adam at 1e-4, effective batch 16, and recorded random seeds. Select physical batch 8 or 4 from the setup memory probe and use accumulation.
- Start in FP32. Test mixed precision as a separate change after exact-PNG parity. Train through clamping and 8-bit rounding with a documented gradient approximation.
- Initial experimental loss: bit BCE plus image MSE, with image-loss weight increasing linearly from 0 to 100 over the first 20% of planned optimizer steps and then held at 100. Log both terms separately. This is a starting configuration, not an established optimum; any diagnostic change creates a new recorded experiment and uses validation only.
- Use full-frame inference when memory permits and halo-based tiling for large images. Verify tiled/full-frame agreement, border handling, odd dimensions, and exact recovered bytes. Do not assume arbitrary normalization layers preserve tile equivalence.
- Record complete-message recovery, raw BER, PSNR, SSIM, LPIPS, clipping, latency, peak memory, and failure examples. Quality is measured on saved/reopened integer images.
- The primary release benchmark uses 10,000 frozen source images, each prepared at 1024 × 1024 with a full 1,024-byte user message before encryption. Empty/short messages, smaller capacity tiers, and native-resolution tests are separate suites and cannot inflate this gate's recovery rate.
- Green means recovery and quality targets pass; yellow means quality needs attention; red means recovery fails; unmeasured remains unmeasured. None of these indicates SOTA.
- Later independent detection uses separately trained SRNet and Zhu-Net-style detectors with public encoder weights, cover/stego pairs kept together in splits, and matched processing histories. Report AUC, balanced error, TPR at fixed FPR, and cross-source results. Document RGB adaptations and implementation terms. [SRNet](https://ws.binghamton.edu/fridrich/research/SRNet.pdf), [Zhu-Net](https://arxiv.org/abs/1807.11428)

## 7. Checkpoints and independent exports

- Sprint 4 implements verified atomic CPU checkpoints containing both networks,
  BatchNorm buffers, Adam state, Python/NumPy/Torch random state, sampler position,
  optimizer step, and frozen configuration/data/environment identities. Loads use
  Torch's safe `weights_only` mode; exact resume requires matching identities.
  The CPU profile uses no scheduler or mixed-precision scaler.
- Experimental CPU `torch.export` packages cover batch-one RGB tensor inputs
  with sides from 512 to 1024, including odd dimensions. Their manifests mark
  them experimental; CPU package checks do not qualify GPU, 4K, or application use.

- Save complete training state: networks, optimizer/scheduler/scaler state, random states, consumed-step/sampler state, dataset manifest, resolved configuration, and environment/source identifiers. Weights alone are insufficient. [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- Save recovery checkpoints every five minutes by default and best candidates after validation. Pause/stop saves at a completed step boundary. Exact continuation is only promised for the same supported deterministic environment.
- Write atomically, verify the new checkpoint, then prune. Retain latest recovery, three best, and pinned checkpoints. Rank by complete-message recovery, then PSNR, then SSIM; deployment exports are separate.
- Export separate torch.export encoder and decoder artifacts with a small independent runtime wrapper, manifest, checksums, supported shapes, capacity rules, preprocessing/protocol versions, exact dependencies, and known-answer vectors. [Export API](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)
- The decoder package needs no encoder, original cover, critic, optimizer, or training dataset. The encoder package needs no decoder weights. Saved-PNG verification is mandatory before application/GUI download and is not part of the independent encoder package.
- Validate exports in separate clean processes on Mac CPU and EC2 GPU. Exact plaintext must match; cross-device floating tensors need not be bit-identical.

## 8. Tests and completion gate

- Unit/protocol: UTF-8 boundaries, empty/full/overflow messages, randomized salt/nonce, wrong passwords, corrupt frames/ECC, incompatible versions, and no secret logging.
- Data/API: invalid requests/responses, duplicate submissions, corrupt images, archive traversal, unsafe URLs, gated/missing repositories, revision pinning, pause/cancel, and offline cache reuse.
- Training: tiny-set overfit, quantized PNG round trip, full-state stop/resume, incompatible checkpoints, disk-full recovery, and budget expiry.
- Export: independent packages, odd dimensions, 4K boundaries, tile parity, supported CPU/GPU runtimes, and missing model files.
- Acceptance: publish frozen-test results before encode filtering and mark unsupported or failing profiles experimental. A pilot failure produces a diagnostic report, not a lowered release target.
