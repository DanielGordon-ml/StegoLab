# Train and review in the browser

The Train tab connects the existing experimental CPU tools to durable background
jobs. It does not qualify a model for Encode or Decode. Those tabs explain their
missing model requirement; public message capacity remains zero.

## Start the workspace

Run the normal Docker Compose command from the repository root. The backend
uses the existing `datasets/`, `checkpoints/`, `models/`, `state/`, and `logs/`
folders through separate mounts. Configuration and GUI job records stay in the
application volume. In particular, the CPU experiment ledger remains
`state/cpu_proof/ledger.json`; a GUI job must never start a fresh copy of it.

On a fresh checkout, run the setup helper before starting the containers. It
creates the output directories. On Linux it also saves the current non-root user
and group IDs in `infrastructure/.env`, so the backend can write host-owned
folders. It preserves existing environment settings. Docker Desktop on Mac uses
the existing default container user.

```sh
sh infrastructure/setup_local_workspace.sh
docker compose -f infrastructure/compose.yaml up --build --wait
```

For an existing Linux installation, keep its original `STEGOLAB_USER_ID` and
`STEGOLAB_GROUP_ID` (both default to 10001). Grant that user access to the selected
output folders. Changing the runtime user without migrating an existing named
volume can make saved settings inaccessible. The helper refuses that case.

For native development, `STEGOLAB_WORKSPACE_ROOT` selects the original experiment
root; it defaults to the current directory. Use the same root for CLI and browser
work. `STEGOLAB_DATA_DIRECTORY` controls only application metadata, not a separate
experiment allowance.

## Choose and prepare data

- Choose a prepared dataset from Train. Existing revisions are indexed without
  adding files to them. Discovery does not prove every image's integrity; the
  training worker verifies the dataset before using it.
- The CPU profile accepts at most 30 source records and 512 MiB of prepared
  images. It requires four unique training and four tuning images. The selected
  tuning images must have both sides at least 1024 pixels.
- Large datasets remain visible. The GUI explains why they cannot run in this
  CPU profile; it does not silently crop the corpus or change its split.
- Local folders refer to the backend computer. Browser paths do not make files
  available to a container or remote server.

For a mounted local image folder, use the existing source override:

```sh
STEGOLAB_SOURCE_DIRECTORY=/absolute/path/to/images \
  docker compose -f infrastructure/compose.yaml \
  -f infrastructure/compose.datasets.yaml up --build --wait
```

For native development, set `STEGOLAB_SOURCE_ROOTS` to a JSON object mapping short
labels to absolute folder paths. Only registered source roots and their safe
subfolders are accessible. Generic local folders use the existing deterministic
splits, so a folder with too few tuning images may not fit the CPU profile.

## Start, stop, and resume

1. Choose a compatible dataset and a run name.
2. Choose CPU threads and the absolute step at which to stop. Advanced details
   show the fixed training settings.
3. Review preflight checks and the remaining experiment budget, then start.
4. Follow progress and checkpoint measurements. Switching tabs or refreshing
   does not stop the worker.
5. Stop and save waits for the trainer's safe checkpoint boundary. A stopped
   job is not the same as an exported model.

The tested CPU profile saves every five minutes. The Config checkpoint default
is for future configurable profiles and does not change this fixed interval.

Resume requires the original experiment, dataset, training configuration, source
code identity, and compatible numerical environment. Existing Mac checkpoints
cannot resume in Linux Docker. The saved baseline also has an older code identity
than the current trainer. Its metadata remains available for review; evaluation
and export have their own validation checks. Never bypass a failed resume check.

After an unclean backend restart, unfinished work is marked interrupted. Resume
is an explicit action; the server does not silently start new learning work.

## Read results

Exact-message recovery, bit accuracy, and image quality are different measures.
An unmeasured value remains unmeasured. A small tuning result does not establish
release quality, statistical secrecy, or resistance to image changes.
Recovery trials and PSNR/SSIM sample counts are shown separately. Missing counts
in older checkpoint records remain unknown; they are not copied from later reports.
When job status cannot be read, retained status is labelled last known and actions
are disabled until the backend can confirm the state again.

Older evaluation reports do not always identify their checkpoint. The catalog
keeps those reports at experiment level instead of guessing a checkpoint link.
New workflow jobs retain their submitted dataset/checkpoint references.

Export produces independent experimental encoder and decoder packages. Download
registered artifacts from the workspace; arbitrary server paths are not download
addresses. No experimental export is automatically installed as a qualified
application model.

## Install an experimental model for Encode and Decode

Exports are never installed automatically. An explicit installation request
verifies both packages and the pair's verification record, then advertises the
model through `GET /api/v1/capabilities` and `GET /api/v1/models`. The request
body names the export reference shown in the workspace and a retry identifier:

```json
{"client_request_identifier": "install-001v", "export_reference": "<export identifier>"}
```

Send it to `POST /api/v1/models/install`. Remove a model with
`DELETE /api/v1/models/<model identifier>`; removal is refused while a queued or
running Encode or Decode job still uses it. Installed models stay experimental:
image quality is visibly reduced and recovery is not guaranteed. The browser
controls for installation, encoding, and decoding arrive with the Sprint 6
interface work.

## Upload an image and check its message limit

Encode and Decode start from an upload. Send the image file itself as the request
body, with the content type `image/png` or `image/jpeg`, and say what the file is
for with the `purpose` query parameter:

- `cover` prepares the image for encoding: it is rotated to its displayed
  orientation, converted to standard sRGB, and saved as a metadata-free PNG.
- `encoded` keeps the file exactly as sent so that Decode reads the same pixels
  Encode wrote. Only unchanged PNG files are accepted here; JPEG and re-saved
  files are refused with a plain explanation.

```bash
curl --request POST "http://127.0.0.1:8080/api/v1/images?purpose=cover" \
  --header "Content-Type: image/png" --data-binary @cover.png
```

The answer names the image by an opaque reference, describes the preparation
(source and prepared sizes, colour handling, warnings), and gives the expiry
time. Uploads are limited to 16 MiB and to sides between 512 and 4096 pixels.
They are deleted after 24 hours or when local image storage reaches 512 MiB. The
prepared file can be downloaded unchanged from `GET /api/v1/artifacts/<image
reference>`.

`POST /api/v1/capacity` with `{"image_reference": ..., "model_identifier": ...}`
reports how many message bytes that image can carry with one installed model.
Images outside the model's side range (512 to 1024 pixels for the experimental
model) are refused before any model process starts.

## Delivery boundaries

Local dataset preparation, bounded CPU experiments, review, and export form the
first working GUI release. GPU readiness, configurable/fine-tuning profiles,
remote dataset downloads, and qualified inference remain later milestones.
No cloud instance is started by the GUI. Tests use disposable storage and ledgers;
they must not consume the remaining real CPU experiment slot.
