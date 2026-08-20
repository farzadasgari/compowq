# Climate Extremes & Water Quality (Sentinel-2)

Reproducible pipeline for a study on the impact of hydroclimatic extreme events
(heatwaves, droughts, compound dry-hot events) on water quality parameters
(TSM, Chl-a, CDOM) retrieved from Sentinel-2 imagery via the C2RCC processor.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the pinned dependencies from `pyproject.toml` and makes `src` importable
as a package (so both `python -m src.<module>` and `from src.<module> import ...` work,
including from `tests/`).

## Configuration

All paths, event-classification thresholds, and index parameters live in
`configs/config.yaml`. no script should hardcode these.

## Reproducing results

Once the pipeline modules exist, each will be runnable as:

```bash
python -m src.<module_name> --arg value ...
```

This section will be filled in with the exact command sequence as each step lands.

## License

Code is released under the Apache License 2.0 License (see `LICENSE`).

## Contact

For any inquiries, please contact:
- khufarzadasgari@gmail.com
- std_farzad.asgari@khu.ac.ir

## Research Team
### Farzad Asgari
<div align="center">

[![github](https://img.shields.io/badge/GitHub-6e5494?style=for-the-badge&logo=github&logoColor=white)](https://github.com/farzadasgari)
[![Google Scholar Badge](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=fff&style=for-the-badge)](https://scholar.google.com/citations?user=Rhue_kkAAAAJ&hl=en)
[![linkedin](https://custom-icon-badges.demolab.com/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=white)](https://www.linkedin.com/in/farzad-asgari)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--3800--0408-A6CE39?logo=orcid&logoColor=fff&style=for-the-badge)](https://orcid.org/0009-0008-3800-0408)

</div>