import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch import download, utils
from neuralset.events import etypes, study

_HF_ORG = "pnpl"
_HF_REPO = "LibriBrain"

# ``{task: {session: run}}`` for every recording that ships a non-empty BIDS
# events file.  The task label is the audiobook ("Sherlock1" ... "Sherlock7").
# Excluded: Sherlock1/ses-11 (0-byte events.tsv).
_SESSION_RUNS: dict[str, dict[int, int]] = {
    "Sherlock1": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 12: 2},
    "Sherlock2": {s: 1 for s in range(1, 13)},
    "Sherlock3": {s: 1 for s in range(1, 13)},
    "Sherlock4": {s: 1 for s in range(1, 13)},
    "Sherlock5": {s: 1 for s in range(1, 16)},
    "Sherlock6": {s: 1 for s in range(1, 15)},
    "Sherlock7": {s: 1 for s in range(1, 15)},
}


class Ozdogan2025Libribrain(study.Study):
    """LibriBrain: MEG responses to spoken English narrative (Sherlock Holmes).

    Over 50 hours of within-subject MEG recorded while a single participant
    listened to LibriVox audiobook recordings of the first seven Sherlock
    Holmes books.  Annotations give word and phoneme onsets aligned to the MEG
    clock, plus the silence intervals used by the speech-detection benchmark.

    Experimental Design:
        - MEG recordings (306-channel MEGIN TRIUX Neo, 1000 Hz)
        - 1 participant, 90 sessions across 7 audiobooks
        - Paradigm: passive listening to spoken narrative
            * Word- and phoneme-level onset annotations
            * Stimulus language: English

    Notes:
        - Single-subject dataset: every timeline shares ``subject="0"``.
        - ``_download`` fetches the raw BIDS recordings only (~430 GB) and
          skips the ``derivatives/`` tree (preprocessed FIF and serialised
          HDF5 copies).  Use :class:`Ozdogan2025LibribrainSample` to work with
          a single session.
        - The audiobook WAV files under ``<task>/stimuli/audio/`` are not
          emitted as ``Audio`` events: the annotations do not record which
          chapter each session covers, and the stimulus clock drifts by
          ~0.5% relative to the MEG clock over a session.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("LibriBrain",)
    bibtex: tp.ClassVar[str] = """
    @article{ozdogan2025libribrain,
        title={{LibriBrain}: Over 50 Hours of Within-Subject {MEG} to Improve Speech Decoding Methods at Scale},
        author={{\\"O}zdogan, Miran and Landau, Gilad and Elvers, Gereon and Jayalath, Dulhan and Somaiya, Pratik and Mantegna, Francesco and Woolrich, Mark and Parker Jones, Oiwi},
        journal={NeurIPS, Datasets \\& Benchmarks Track},
        year={2025},
        url={https://arxiv.org/abs/2506.02098}
    }
    """
    url: tp.ClassVar[str] = "https://huggingface.co/datasets/pnpl/LibriBrain"
    licence: tp.ClassVar[str] = "CC-BY-NC-4.0"
    description: tp.ClassVar[str] = (
        "Over 50 hours of MEG from 1 participant listening to Sherlock Holmes audiobooks."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("huggingface_hub",)

    # TODO populate via `update_source_info("Ozdogan2025Libribrain")`
    _info: tp.ClassVar[study.StudyInfo | None] = None

    _subject: tp.ClassVar[str] = "0"  # single-participant dataset
    _session_runs: tp.ClassVar[dict[str, dict[int, int]]] = _SESSION_RUNS
    _tsv_columns: tp.ClassVar[tuple[str, ...]] = (
        "kind",
        "segment",
        "sentenceidx",
        "timemeg",
        "duration",
    )
    _event_types: tp.ClassVar[dict[str, str]] = {
        "word": "Word",
        "phoneme": "Phoneme",
        "silence": "Stimulus",
    }

    def _allow_patterns(self) -> list[str]:
        """Hugging Face globs selecting the raw BIDS recordings we enumerate.

        Patterns are exact rather than ``<task>/*.tsv``: Hugging Face matches
        with ``fnmatch``, where ``*`` also spans ``/`` and would pull in the
        ``derivatives/`` copies of the annotations.
        """
        sub = f"sub-{self._subject}"
        patterns = ["metadata/**"]
        for task, runs in self._session_runs.items():
            patterns += [
                f"{task}/dataset_description.json",
                f"{task}/participants.json",
                f"{task}/participants.tsv",
            ]
            patterns += [f"{task}/{sub}/ses-{session}/**" for session in runs]
        return patterns

    def _download(self) -> None:
        # download.Huggingface snapshots the whole 552 GB repository; calling
        # snapshot_download directly keeps only the sessions we enumerate.
        dl_dir = self.path / "download"
        dl_dir.mkdir(parents=True, exist_ok=True)
        with download.success_writer(dl_dir / "snapshot") as already_done:
            if already_done:
                return
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=f"{_HF_ORG}/{_HF_REPO}",
                repo_type="dataset",
                local_dir=dl_dir,
                allow_patterns=self._allow_patterns(),
            )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for task, runs in self._session_runs.items():
            for session, run in runs.items():
                yield dict(subject=self._subject, session=session, task=task, run=run)

    def _bids_stem(self, timeline: dict[str, tp.Any]) -> Path:
        """Path prefix shared by the ``_meg.fif`` and ``_events.tsv`` files."""
        task = timeline["task"]
        sub, ses = f"sub-{timeline['subject']}", f"ses-{timeline['session']}"
        stem = f"{sub}_{ses}_task-{task}_run-{timeline['run']}"
        return self.path / "download" / task / sub / ses / "meg" / stem

    def _make_meg_event(self, timeline: dict[str, tp.Any]) -> etypes.Meg:
        stem = self._bids_stem(timeline)
        filepath = stem.with_name(f"{stem.name}_meg.fif")
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing MEG file for {self.__class__.__name__}: {filepath}. "
                "Please run study.download() first."
            )
        # start="auto": the FIF time origin is raw.first_samp / sfreq
        return etypes.Meg(
            timeline="",
            filepath=str(filepath),
            subject=timeline["subject"],
            start="auto",  # type: ignore
        )

    def _load_annotations(
        self, timeline: dict[str, tp.Any], start: float
    ) -> pd.DataFrame:
        """Word, phoneme and silence events, shifted onto the FIF time base."""
        stem = self._bids_stem(timeline)
        tsv = stem.with_name(f"{stem.name}_events.tsv")
        raw = pd.read_csv(tsv, sep="\t", usecols=list(self._tsv_columns))
        if raw.empty:
            raise RuntimeError(f"No annotation found in {tsv}")
        types = raw["kind"].map(self._event_types)
        unknown = raw.loc[types.isna(), "kind"].unique()
        if len(unknown):
            raise ValueError(f"Unexpected event kind(s) {list(unknown)} in {tsv}")
        events = pd.DataFrame(
            {
                "type": types,
                "start": raw["timemeg"] + start,
                "duration": raw["duration"],
                "sequence_id": raw["sentenceidx"],
            }
        )
        speech = types != "Stimulus"
        events.loc[speech, "language"] = "english"
        events.loc[speech, "modality"] = "heard"
        events.loc[~speech, "description"] = "silence"
        text = raw["segment"]
        phoneme = types == "Phoneme"
        parts = text[phoneme].str.rsplit("_", n=1)
        events.loc[phoneme, "phoneme_position"] = parts.str[1]
        events["text"] = text.mask(phoneme, parts.str[0])
        return events

    def _make_text_event(self, events: pd.DataFrame) -> dict[str, tp.Any]:
        """Full session transcript, spanning the first to the last word."""
        words = events.loc[events.type == "Word"]
        start = words.start.min()
        return {
            "type": "Text",
            "text": " ".join(words.text),
            "start": start,
            "duration": (words.start + words.duration).max() - start,
            "language": "english",
            "modality": "heard",
        }

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        meg = self._make_meg_event(timeline)
        events = self._load_annotations(timeline, meg.start)
        events = utils.add_sentences(events)
        extra = [meg.to_dict(), self._make_text_event(events)]
        events = pd.concat([pd.DataFrame(extra), events])
        return events.sort_values(by="start").reset_index(drop=True)

# TODO maybe add