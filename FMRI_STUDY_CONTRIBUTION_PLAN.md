# Contributing an fMRI dataset to `facebookresearch/neuroai`

A plan for onboarding a public fMRI dataset as a NeuralSet `Study`, structured so
that the work stays compatible with an actively-developed upstream and so that
the *second* dataset is much cheaper than the first.

**Status:** draft for team discussion
**Written:** 2026-08-18
**Verified against:** local `main` = `a1bf64e`; upstream `facebookresearch/neuroai` `main` = `af8ce00` (2026-07-31)

---

## TL;DR — the four decisions

1. **A `Study` does not denoise, mask, or parcellate.** Those are already
   implemented in `FmriExtractor` and are *configuration*, not code. Building
   them into the Study is the main way this project could go wrong.
2. **Daily work lives in a standalone package**, discovered through the
   `neuralset.studies` entry point — not in a fork of the monorepo.
3. **The fork exists only to stage pull requests**, never as a development home.
4. **Open an upstream issue before writing code.** Required by CONTRIBUTING.md,
   and there is specific reason to do it here (see §9).

---

## 1. What a `Study` actually is

A `Study` enumerates *timelines* (subject × session × run) and emits a DataFrame
of **events**. An `Fmri` event is essentially a typed pointer — `filepath`,
`subject`, `space`, `preproc`, `frequency`, `mask_filepath`, and an optional
`spec` dict for derivative variants (`neuralset/events/etypes.py`).

Everything downstream of that already exists:

| Pipeline step | Where it lives | You write? |
|---|---|---|
| Download | `Study._download` | **yes** |
| Enumerate runs | `Study.iter_timelines` | **yes** |
| Load events | `Study._load_timeline_events` | **yes** |
| Denoise | `FmriCleaner` — detrend, standardize, high/low-pass, butterworth/cosine | no — config |
| Mask | `MaskProjector` — mni152 gm/wm/brain, subcortical | no — config |
| Parcellate | `AtlasProjector` (any `nilearn.datasets.fetch_atlas_*`), `GlasserProjector`, `SurfaceProjector`, `CiftiRoiProjector` | no — config |
| Segment / batch | `Segmenter`, `DataLoader` | no — config |
| Train / benchmark | neuralbench task YAML | no — config |

"Denoise and parcellate to Schaefer-400" is this, with no new code:

```python
ns.extractors.FmriExtractor(
    cleaning=ns.extractors.FmriCleaner(
        detrend=True, standardize="zscore_sample", high_pass=0.01
    ),
    projection=ns.extractors.AtlasProjector(
        atlas="schaefer_2018", atlas_kwargs={"n_rois": 400}
    ),
)
```

These transforms run lazily at segmentation time and are cached by `exca`.

**Action for the team:** before writing anything, inventory the existing
preparation pipeline and mark each stage as *replaceable by config* or
*genuinely study-specific*. Every stage deleted is a stage that cannot rot
against upstream.

### Reference implementations

| Study | Lines | Why read it |
|---|---|---|
| `chang2019bold5000.py` | 278 | Cleanest fMRI example: OpenNeuro download, BIDS events, `SpecialLoader` |
| `li2022petit.py` | 371 | fMRI + audio + word-level annotations |
| `hebart2023things.py` | 534 | Multiple derivative variants in one study |
| `allen2022massive.py` | 1015 | Large, multi-space, several study subclasses |

---

## 2. Repository topology

### Repo A — `<lab>/neuro-<dataset>` (the satellite package)

Where day-to-day work happens. Registers itself so that stock, pip-installed
neuroai discovers it:

```toml
[project.entry-points."neuralset.studies"]
<lab> = "<lab>_neuro"
```

NeuralSet resolves studies by scanning every package registered under this
entry-point group (`neuralset/events/study.py`, `_get_study_packages`).
`neuralfetch` itself registers exactly this way and has no privileged status.

Consequences:

- `ns.Study(name="Author2026Keyword", path=...)` works, as do all extractors,
  segmenters and neuralbench configs — **with no fork at all**.
- Upstream updates arrive via `pip install -U`. Breakage surfaces as a test
  failure in a repo you own, not as a merge conflict in code you don't.
- The team iterates at its own pace, before any public commitment.

This is not "living on the periphery" — it is the framework's supported
extension mechanism.

### Repo B — `<lab>/neuroai` (the fork)

A fork is needed: it is the only way to open a PR without write access. Its job
is narrow.

- Kept **identical to upstream `main`** at all times.
- Hosts only short-lived PR branches: branch from a freshly fetched
  `upstream/main`, copy the study file in, open the PR, delete the branch after
  merge.
- **Anti-pattern:** a long-lived `our-dataset` branch accumulating months of
  work. Across a four-package monorepo under active development, that is exactly
  the sync problem we are trying to avoid.

---

## 3. The rule that makes upstreaming a file move

> The study module imports **only** from `neuralset.events.study`,
> `neuralset.events.etypes`, `neuralset.utils`, `neuralfetch.download`, and
> `neuralfetch.utils` — never from other modules of our own package.

Hold this and contributing is `git mv` plus one pyproject line. Break it — a
shared `utils.py`, a constants module, a config helper — and upstreaming becomes
a refactor negotiated in review.

Enforce it in CI, not by good intentions:

```bash
grep -nE '^\s*(from|import)\s+<lab>_neuro' <lab>_neuro/studies/*.py && exit 1
```

Helpers that feel worth sharing usually already exist in `neuralset/utils.py`:
`get_bids_filepath`, `get_bids_files`, `read_bids_events`,
`get_masked_bold_image`.

**Adopt upstream conventions from day one**, so review is about science, not
style: ruff line-length 90 with upstream's `E,F,I,W,A` selection; mypy with the
pydantic plugin; class name = PascalCase of the Google Scholar BibTeX key
(`allen2022massive` -> `Allen2022Massive`), file = lowercase of that. Renaming
later invalidates every cache path and YAML reference.

---

## 4. Phases

### Phase 0 — Talk to upstream first

- [ ] Open an issue describing the dataset. CONTRIBUTING.md is explicit:
      *"New datasets — add a Study subclass in `neuralfetch-repo/`; open an issue
      first to discuss."*
- [ ] **Ask whether a study for this dataset already exists internally** (see §9).
- [ ] Everyone who will commit signs the Meta CLA: <https://code.facebook.com/cla>
      (one-time, covers all Meta OSS; blocks merge otherwise).
- [ ] Confirm the hosting is supported by a `neuralfetch.download` backend:
      S3, Dandi, **Datalad**, Donders, Dryad, Eegdash, Figshare, Globus,
      Huggingface, Openneuro, Osf, Physionet, Synapse, Zenodo.
- [ ] Confirm licensing. The guide accepts data that is public *or "at least
      requestable with a free account"*, so a registration/DUA-gated dataset can
      still qualify — but state this in the issue rather than discovering it in
      review.

> **Note:** the repo forbids automated agents from opening issues, PRs, or
> comments. A human opens these. Using AI as a drafting aid is explicitly
> permitted, with the contributor responsible for every line submitted.

### Phase 1 — Choose which derivatives to expose

The biggest scoping decision. Supported inputs: 4-D NIfTI, GIFTI surface (point
at `hemi-L`; the right hemisphere is derived automatically), and CIFTI
`.dtseries.nii`.

`Fmri` events carry `space`, `preproc` and a free-form `spec` dict, and
`FmriExtractor._auto_filter_fmri_events` selects among them at extraction time —
so one study can ship volumetric MNI *and* surface *and* CIFTI variants side by
side, with users choosing via `from_space`.

Recommendation: ship **preprocessed derivatives, one space to start**. Add
variants in a follow-up PR.

### Phase 2 — Write the study

Implement `_download`, `iter_timelines`, `_load_timeline_events`. Two idioms to
copy from `chang2019bold5000.py`:

- Wrap downloads in `download.success_writer(...)` so re-runs are no-ops.
- Use `study.SpecialLoader(method=self._load_raw, timeline=timeline)` when a file
  is not directly loadable — concatenating runs, applying a session mask. This is
  the escape hatch for study-specific assembly, and it stays lazy and cacheable.

Keep it thin. Anything general-purpose in there is a smell and will be asked
about in review.

### Phase 3 — Populate `_info` (this is the gate)

`test_study_info` in `neuralfetch/test_studies.py` **skips any study whose
`_info is None`**. An unpopulated `StudyInfo` means the study has *zero* test
coverage and will not be accepted.

```bash
python -c "from neuralfetch.utils import update_source_info; update_source_info('Author2026Keyword')"
```

This writes back `num_timelines`, `num_subjects`, `num_events_in_query`,
`event_types_in_query`, `data_shape`, `frequency`, `fmri_spaces`. Treat it as the
contract: it is the regression test that catches upstream changes breaking the
loader.

### Phase 4 — Benchmark integration (separate, later PR)

A dataset YAML under `neuralbench-repo/neuralbench/tasks/fmri/<task>/datasets/`.
`tasks/fmri/image/datasets/Chang2019Bold5000.yaml` is the template: study +
split, `FmriExtractor` with a projector, a target model, a brain model, SLURM
infra — roughly 60 lines. If the paradigm is not image-viewing, a new task
directory is needed; discuss in its own issue.

### Phase 5 — Upstream

- [ ] Branch from fresh `upstream/main` on the fork
- [ ] `git mv` the study file into `neuralfetch-repo/neuralfetch/studies/`
- [ ] From inside `neuralfetch-repo/`: `ruff check . && mypy . && pytest -x -q`
- [ ] Open the PR (squash-merged, so internal commit history does not matter)

---

## 5. Robustness to upstream change

### The contract surface

The dependency footprint of a study is about twenty symbols. Write them down and
defend them:

- **`study.Study`**: `path`, `version`, `query`, `timelines`, `_download`,
  `iter_timelines`, `_load_timeline_events`, `_info`, and the classvars
  `aliases` / `bibtex` / `url` / `licence` / `description` / `requirements`
- **`study.SpecialLoader`**, **`study.StudyInfo`**
- **`etypes.Fmri`** field names, plus the stimulus event types used
- **`neuralfetch.download`**: the chosen backend + `success_writer`
- **`neuralset.utils`**: `get_bids_filepath`, `read_bids_events`,
  `get_masked_bold_image`
- the entry-point group string `neuralset.studies`

### Three defenses, in order of value

1. **`_info` is the regression test.** Populate it; treat any diff as a real
   signal.
2. **Weekly CI cron against upstream `main`** — not only against our own commits
   and not only against the pinned release. Two jobs: one on the pinned version
   (must pass), one on git-`main` (allowed to fail, but it notifies). This turns
   *"our loader broke three months ago"* into *"our loader broke last Tuesday,
   and here is the commit."*
3. **Pin a range** (`neuralset>=0.2.3,<0.3`) and bump deliberately, reading
   `CHANGELOG.md` each time.

---

## 6. Why this matters — measured drift

Checked 2026-08-18. Local `main` is at `a1bf64e` (2026-07-06); upstream is at
`af8ce00` (2026-07-31) — **13 commits behind**, and several land directly on the
fMRI path:

| Commit | Change |
|---|---|
| `86846e2` (#210) | `CiftiRoiProjector` — cortical Glasser + subcortical ROIs |
| `9567cc9` (#215) | fMRI ROI query/selection; adds `neuralset/events/eventquery.py`; **deletes** `neuralbench/sklearn_baseline.py` |
| `487220d` (#207) | New brain-model build API (`build_from_context`) |
| `46ee2d7` (#201) | New study `levy2026noninvasive.py` |
| `1443855` (#216) | Version bump 0.2.2 -> 0.2.3 |

**A `Study`-facing API was removed inside this window.** `infra_timelines` was a
`Study` constructor argument; `#194` replaced it with `timelines.infra` and
`study.py` now raises on the old form. Our tutorial copy still calls the removed
API (`docs/neuralfetch/tutorials/03_create_new_study.py:65`); upstream fixed that
doc in `#202`, which we do not have.

The reassuring half: upstream shipped `_migrate_infra_timelines` with a
deprecation path and an error message naming the replacement. Migrations are
handled deliberately — but only if you are actually tracking `main`.

---

## 7. Known gaps to scope around

**No confound regression.** `FmriCleaner` wraps `nilearn.signal.clean` but never
passes `confounds`; nothing in neuralset or neuralfetch mentions confounds,
aCompCor, or motion regressors. If our preparation regresses out fMRIPrep
confounds, there is currently no home for it. Options:

1. Ship already-denoised derivatives — simplest.
2. Do it in the study's `SpecialLoader` path — works, but per-study and
   arguably misplaced.
3. **Contribute `confounds` support to `FmriCleaner` upstream** — small,
   well-scoped, useful independently of our dataset. Recommended.

**`AtlasProjector` and `MaskProjector` are volumetric-only** — both raise on
non-4-D input. For surface/CIFTI data the parcellation options are
`GlasserProjector` and `CiftiRoiProjector`, and note that `GlasserProjector`
*masks vertices* rather than aggregating each ROI to a single time series. Check
this against what the team means by "parcellated".

---

## 8. Scoping reference

The most recent merged dataset onboarding (`#201` / `#214`,
`Levy2026Noninvasive`) was **+1079 / -24 across 11 files**:

- 1 new study file (1061 lines) — the bulk
- `docs/neuralbench/dataset_table.csv`, two task `.rst` docs
- two neuralbench task `config.yaml`s, `defaults/debug_study.yaml`
- `neuralbench/plots/plot_non_eeg.py`, `neuralbench/transforms.py`
- (all of these one- or two-line touches)

Budget **200–1100 lines** for the study file and expect to touch ~10 files if the
dataset also enters the benchmark.

Worth noting for morale: `#201` and `#214` are byte-identical diffs merged a day
apart, with a revert (`#212`) in between. Even maintainers re-land things.

---

## 9. Dataset-specific notes — CNeuroMod

> **Assumption**, taken from the working branch name `cneuromod-study`. Confirm
> before relying on this section.

**There is no CNeuroMod or Algonauts study upstream.** The public studies
directory at `af8ce00` contains 32 studies; none of them cover this data.

But two facts say the ground is already prepared:

1. `#191` added **CIFTI `.dtseries.nii` support** to `etypes.Fmri`, titled
   *"Support Algonauts CIFTI fMRI derivatives"*. Algonauts 2025 is built on
   CNeuroMod data — so the framework already handles data of exactly this shape.
2. `neuralset/test_segment_benchmark.py` benchmarks against a study named
   **`Gifford2025AlgonautsBold`, which does not exist in the public repo**. It is
   internal to Meta.

**Implication — do this before writing 1000 lines:** ask in the Phase 0 issue
whether a CNeuroMod / Algonauts study already exists internally or is in flight.
The answer changes everything: we might be contributing the public counterpart of
an existing internal study, in which case matching its class naming, timeline
scheme and `spec` conventions matters enormously — or we might be duplicating
work already done.

Practical points to verify with the team:

- **Distribution is DataLad / git-annex.** `neuralfetch.download.Datalad` exists
  and supports selective retrieval via `folders` and `Wildcard` — good fit. Its
  docstring requires a **password-free SSH key**; check how that interacts with
  CI and with any access agreement.
- **Access model.** If retrieval is gated behind registration or a data use
  agreement, say so explicitly in the issue. The contributing guide permits
  "requestable with a free account", but maintainers should decide, not us.
- **Scope.** CNeuroMod spans several paradigms (naturalistic film, audio,
  gaming, task batteries). Ship **one** to start; each additional paradigm is a
  follow-up PR and possibly a new neuralbench task.
- **Derivatives.** Decide volumetric vs surface vs CIFTI up front (Phase 1); the
  CIFTI path is the best-supported one for this data shape.

---

## 10. Open questions for the team

1. Which dataset exactly, and which paradigm(s) in scope for v1?
2. Are published preprocessed derivatives available, and from which pipeline?
3. Volumetric, surface, or CIFTI?
4. Does the current preparation pipeline regress out confounds? (Determines §7.)
5. Goal: benchmark inclusion, or ingestion for our own training only?
6. Who signs the CLA, and who is the human of record for the issue and PR?

---

## Appendix — commands

Track upstream from this clone (read-only, reversible):

```bash
git remote add upstream https://github.com/facebookresearch/neuroai.git
git fetch upstream
git log --oneline HEAD..upstream/main
```

Development install (from CONTRIBUTING.md):

```bash
pip install -e 'neuralset-repo/.[dev,all]' -e 'neuralfetch-repo/.[dev]' \
            -e 'neuraltrain-repo/.[dev,all]' -e 'neuralbench-repo/.[dev]'
```

Pre-submission checks, run inside the relevant `*-repo/`:

```bash
ruff check . && mypy . && pytest -x -q
```

Regenerate study metadata:

```bash
python -c "from neuralfetch.utils import update_source_info; update_source_info('Author2026Keyword')"
```
