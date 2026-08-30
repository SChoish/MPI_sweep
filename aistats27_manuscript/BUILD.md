# Building the manuscript

Run every command from this directory. The canonical build entry point is:

```bash
./build_pdf.sh
```

The script regenerates all figures, runs Tectonic until references settle,
copies `paper.pdf` to `mpi_aistats27_manuscript.pdf`, and fails on
overfull boxes or fatal TeX errors.

## Current workstation

The Python environment and Tectonic binary already installed on this machine
can be selected explicitly:

```bash
cd /home/ext_csv/mpi_sweep_lab/aistats27_manuscript
PYTHON_BIN=/home/ext_csv/miniconda3/envs/offrl/bin/python \
TECTONIC_BIN=/home/ext_csv/.local/bin/tectonic \
./build_pdf.sh
```

The expected final artifact is:

```text
/home/ext_csv/mpi_sweep_lab/aistats27_manuscript/mpi_aistats27_manuscript.pdf
```

## First-time Tectonic installation on Linux

Use the static MUSL archive. The GNU archive may fail on minimal systems when
`libgraphite2.so.3` is absent.

```bash
mkdir -p "$HOME/.local/bin"
mkdir -p /tmp/mpi-tectonic-0.16.9
curl -fL --retry 3 \
  https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz \
  -o /tmp/mpi-tectonic-0.16.9/tectonic.tar.gz
tar -xzf /tmp/mpi-tectonic-0.16.9/tectonic.tar.gz \
  -C /tmp/mpi-tectonic-0.16.9
install -m 0755 /tmp/mpi-tectonic-0.16.9/tectonic \
  "$HOME/.local/bin/tectonic"
"$HOME/.local/bin/tectonic" --version
```

The build script discovers `tectonic` from `PATH`, then falls back to
`$HOME/.local/bin/tectonic`. Override it with `TECTONIC_BIN=/path/to/tectonic`.

## Python requirement

`make_figures.py` requires NumPy and Matplotlib. If the active `python3`
does not provide them, set `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python ./build_pdf.sh
```

## Verification

A successful build must satisfy all of the following:

- `mpi_aistats27_manuscript.pdf` is refreshed.
- `paper.log` contains no overfull box, undefined-control-sequence, emergency
  stop, or fatal-error message.
- `paper.aux` places `sec:conclusion` on page 8 or earlier.
- `app:proofs` starts only after the references.

Useful checks:

```bash
rg 'newlabel\{(sec:conclusion|app:proofs)\}' paper.aux
rg 'Overfull|Undefined control sequence|Emergency stop|Fatal error' paper.log
```

Tectonic may print Fontconfig or underfull-box warnings in this container.
Those are non-blocking if the checks above pass. The bibliography is embedded
in `sections/references.tex`; BibTeX is not used.

The repository currently uses `aistats2027_provisional.sty`. Rebuild and
recheck pagination after replacing it with the official style.
