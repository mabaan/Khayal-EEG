import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import mne
import warnings


import pyedflib


sentences = {
    1: "المريض يشعر بالجوع",
    2: "المريض يشعر بالألم",
    3: "الممرض يحضر الطعام",
    4: "الممرض يحضر الماء",
    5: "السرير غير مريح",
    6: "الطعام غير لذيذ",
    7: "الطبيب يغادر المستشفى",
    8: "الطبيب يصف الدواء",
    9: "الصيدلي يعد الدواء",
    10: "الصيدلي يجيب الهاتف",
    11: "العامل يمسح الأرض",
    12: "العامل يرتب السرير",
}

translit_map = {
    "المريض يشعر بالجوع": "almareed_yash3ur_biljoo3",
    "المريض يشعر بالألم": "almareed_yash3ur_bilalam",
    "الممرض يحضر الطعام": "almumarid_yu7dar_alta3am",
    "الممرض يحضر الماء": "almumarid_yu7dar_alma2",
    "السرير غير مريح": "alsareer_gheir_muree7",
    "الطعام غير لذيذ": "alta3am_gheir_latheeth",
    "الطبيب يغادر المستشفى": "altabeeb_yughadir_almustashfa",
    "الطبيب يصف الدواء": "altabeeb_yasif_aldawa2",
    "الصيدلي يعد الدواء": "alsaydalee_yu3id_aldawa2",
    "الصيدلي يجيب الهاتف": "alsaydalee_yujeeb_alhatif",
    "العامل يمسح الأرض": "al3amil_yamsa7_alard",
    "العامل يرتب السرير": "al3amil_yuratib_alsareer",
}

EEG_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

BLINK_SEC = 0.5
WINDOW_SEC = 5.5
EDF_RECORD_DURATION_SEC = 0.5  # ensures 5.5s = 11 blocks (no padding)


def read_interval_marker(trial_dir: Path) -> Path | None:
    # Prefer intervalMarker* (any extension), fallback to any .csv
    candidates = sorted([p for p in trial_dir.glob("*intervalMarker*") if p.is_file()], key=lambda p: p.name)
    if len(candidates) > 0:
        return candidates[0]

    csvs = sorted([p for p in trial_dir.glob("*.csv") if p.is_file()], key=lambda p: p.name)
    if len(csvs) > 0:
        return csvs[0]

    return None


def normalize_marker_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # Common expected: type, duration, latency
    # If not present, try to find close matches
    if "type" not in df.columns:
        for c in df.columns:
            if "type" in c:
                df = df.rename(columns={c: "type"})
                break

    if "duration" not in df.columns:
        for c in df.columns:
            if "duration" in c:
                df = df.rename(columns={c: "duration"})
                break

    if "latency" not in df.columns:
        for c in df.columns:
            if "latency" in c:
                df = df.rename(columns={c: "latency"})
                break

    return df


def export_edf_exact_blocked(data_uv: np.ndarray, sfreq: float, ch_names: list[str], out_path: Path) -> None:
    """
    Writes EDF+ with record_duration=0.5s and NO padding.
    Requires:
      - sfreq is (almost) an integer
      - samples_per_record = sfreq * 0.5 is integer
      - n_samples divisible by samples_per_record
    """
    n_ch, n_samp = data_uv.shape

    sfreq_int = int(round(sfreq))
    if abs(sfreq - sfreq_int) > 1e-6:
        raise ValueError(f"Non-integer sfreq detected: {sfreq}. EDF requires integer samples per record.")

    samples_per_record = int(round(sfreq_int * EDF_RECORD_DURATION_SEC))

    if abs(sfreq_int * EDF_RECORD_DURATION_SEC - samples_per_record) > 1e-6:
        raise ValueError(
            f"sfreq {sfreq_int} not compatible with record_duration {EDF_RECORD_DURATION_SEC} "
            f"(samples_per_record not integer)"
        )

    if n_samp % samples_per_record != 0:
        raise ValueError(
            f"n_samples {n_samp} not divisible by samples_per_record {samples_per_record} "
            f"(sfreq={sfreq_int}, record_duration={EDF_RECORD_DURATION_SEC})"
        )

    writer = pyedflib.EdfWriter(str(out_path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS)

    try:
        # Silence the record_duration warning (this is expected for our use-case)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Forcing a specific record_duration might alter calculated sample_frequencies*",
                category=UserWarning,
            )
            try:
                writer.setDatarecordDuration(EDF_RECORD_DURATION_SEC)
            except Exception:
                writer.setDatarecordDuration(int(EDF_RECORD_DURATION_SEC * 1e7))

        now = datetime.now()

        # IMPORTANT: pyedflib expects 'sex' not 'gender'
        header = {
            "technician": "",
            "recording_additional": "",
            "patientname": "",
            "patient_additional": "",
            "patientcode": "",
            "equipment": "EPOC",
            "admincode": "",
            "sex": "",          # <-- was 'gender'
            "startdate": now,
            "birthdate": "",
        }
        writer.setHeader(header)

        sig_headers = []
        for i, ch in enumerate(ch_names):
            sig = data_uv[i, :]
            pmin = float(np.min(sig))
            pmax = float(np.max(sig))

            if pmin == pmax:
                pmin -= 1.0
                pmax += 1.0

            margin = 0.05 * (pmax - pmin)
            pmin -= margin
            pmax += margin

            sig_headers.append({
                "label": ch,
                "dimension": "uV",
                "sample_frequency": sfreq_int,  # must be int
                "physical_min": pmin,
                "physical_max": pmax,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            })

        try:
            writer.setSignalHeaders(sig_headers)
        except Exception:
            for i, h in enumerate(sig_headers):
                writer.setSignalHeader(i, h)

        writer.writeSamples([data_uv[i, :] for i in range(n_ch)])

    finally:
        try:
            writer.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".", help="Folder that contains subject folders (S0, S1, ...)")
    parser.add_argument("subjects", nargs="+", help="Subjects like S0 S2 S3 (folder names)")
    parser.add_argument("--skip", type=str, default="1,7,13,19", help="Comma-separated subject numbers to skip")
    args = parser.parse_args()

    root = Path(args.root)
    skip_set = {int(x.strip()) for x in args.skip.split(",") if x.strip()}

    subjects = []
    for s in args.subjects:
        if s.startswith("S") and s[1:].isdigit() and int(s[1:]) in skip_set:
            continue
        subjects.append(s)

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Processing Subject: {subject}")
        print(f"{'='*60}")

        root_dir = root / subject
        if not root_dir.is_dir():
            print(f"Missing subject dir: {root_dir}")
            continue

        classification_root = Path("../classification_folders") / subject
        classification_root.mkdir(parents=True, exist_ok=True)

        for c in range(1, 13):
            class_dir = root_dir / f"C{c}"
            if not class_dir.is_dir():
                continue

            arabic_sentence = sentences[c]
            full_translit = translit_map.get(arabic_sentence, f"sentence{c}")
            arabic_words = arabic_sentence.split()
            translit_words = full_translit.split("_")

            for t in range(1, 11):
                trial_dir = class_dir / f"T{t}"
                if not trial_dir.is_dir():
                    continue

                # Find EDF
                edf_files = sorted([p for p in trial_dir.iterdir() if p.is_file() and p.suffix.lower() == ".edf"])
                if len(edf_files) != 1:
                    continue
                edf_file = edf_files[0]

                marker_file = read_interval_marker(trial_dir)
                if marker_file is None:
                    continue

                try:
                    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

                    channels_to_keep = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
                    raw.pick(channels_to_keep)

                    sfreq = float(raw.info["sfreq"])
                    print(f"{subject}/C{c}/T{t}: sfreq={sfreq}")
                    n_target = int(round(WINDOW_SEC * sfreq))
                    samples_per_record = int(round(sfreq * EDF_RECORD_DURATION_SEC))

                    if samples_per_record <= 0:
                        print(f"Invalid samples_per_record for {subject}/C{c}/T{t}")
                        continue

                    # Sanity: ensure our target window is divisible by record block size
                    if n_target % samples_per_record != 0:
                        # This should not happen when record_duration=0.5 (because 5.5s = 11 * 0.5s)
                        print(f"Target samples not divisible by EDF block size for {subject}/C{c}/T{t}")
                        continue

                    events_df = pd.read_csv(marker_file, sep=None, engine="python")
                    events_df = normalize_marker_df(events_df)

                    if not {"type", "duration", "latency"}.issubset(set(events_df.columns)):
                        print(f"Missing required columns in marker file for {subject}/C{c}/T{t}: {events_df.columns.tolist()}")
                        continue

                    # Detect whether latency is in samples or seconds
                    max_latency = float(pd.to_numeric(events_df["latency"], errors="coerce").max())
                    raw_dur_sec = raw.times[-1] if raw.n_times > 0 else 0.0
                    latency_is_samples = max_latency > (raw_dur_sec * 2.0)

                    imagination_phases = events_df[
                        events_df["type"].astype(str).str.contains("phase_Imagine", na=False)
                    ].copy()

                    imagination_phases["duration"] = pd.to_numeric(imagination_phases["duration"], errors="coerce")
                    imagination_phases["latency"] = pd.to_numeric(imagination_phases["latency"], errors="coerce")
                    imagination_phases = imagination_phases.dropna(subset=["duration", "latency"])

                    # We need >= 6.0 seconds to do (0.5 blink + 5.5 window)
                    imagination_phases = imagination_phases[
                        (imagination_phases["duration"] >= (BLINK_SEC + WINDOW_SEC) - 1e-3)
                    ].sort_values("latency")

                    if len(imagination_phases) != 3:
                        print(f"Skipping {subject}/C{c}/T{t}: found {len(imagination_phases)} valid imagine phases (expected 3)")
                        continue

                    for idx, (_, phase_row) in enumerate(imagination_phases.iterrows(), start=1):
                        onset_val = float(phase_row["latency"])
                        onset_sec = (onset_val / sfreq) if latency_is_samples else onset_val

                        start_sec = onset_sec + BLINK_SEC
                        start_samp = int(round(start_sec * sfreq))
                        stop_samp = start_samp + n_target

                        if start_samp < 0 or stop_samp > raw.n_times:
                            print(f"Skipping {subject}/C{c}/T{t}/W{idx}: window out of bounds (n_times={raw.n_times})")
                            continue

                        data = raw.get_data(start=start_samp, stop=stop_samp)  # shape (n_ch, n_target)
                        data_uv = data * 1e6  # convert V -> uV for nicer EDF scaling

                        arabic_word = arabic_words[idx - 1] if (idx - 1) < len(arabic_words) else f"word{idx}"
                        translit = translit_words[idx - 1] if (idx - 1) < len(translit_words) else f"word{idx}"

                        dest_word_dir = classification_root / translit
                        dest_word_dir.mkdir(exist_ok=True)

                        out_name = f"{subject}_C{c}_T{t}_W{idx}_{translit}.edf"
                        out_path = dest_word_dir / out_name

                        export_edf_exact_blocked(
                            data_uv=data_uv,
                            sfreq=sfreq,
                            ch_names=raw.ch_names,
                            out_path=out_path
                        )

                except Exception as e:
                    print(f"Error processing {subject}/C{c}/T{t}: {e}")
                    continue

        print(f"\n{subject}_classification created:")
        for word_folder in sorted(classification_root.iterdir()):
            if word_folder.is_dir():
                count = len(list(word_folder.glob("*.edf")))
                print(f"  {word_folder.name}: {count} trials")

    print("\nDone.")


if __name__ == "__main__":
    main()
