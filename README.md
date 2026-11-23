# SEV — 20CRv3 severe-storm environments

First implementation for computing **surface-based CAPE and CIN**, **0–1 km and 0–3 km storm-relative helicity**, **0–6 km vector bulk-wind shear**, and **Bunkers right-moving storm motion** from **native, 3-hourly, Gaussian T254 NOAA-CIRES-DOE 20CRv3 ensemble-mean analyses** on Derecho.

**Status:** exploratory benchmark implementation. The paths and file inventory are documented by GDEX, but the analysis has **not been run on actual GDEX files or Derecho yet**. Run `--inspect`, then a small benchmark, before a year-scale job. An optional archived `CAPE_sfc` comparison is included.

## Data source

GDEX dataset `d131003` (`/gdex/data/d131003`). Annual analysis means are in `anl/`:

```text
/gdex/data/d131003/anl/anl_mean_1900_TMP_pres.nc
/gdex/data/d131003/anl/anl_mean_1900_SPFH_pres.nc
/gdex/data/d131003/anl/anl_mean_1900_UGRD_pres.nc
/gdex/data/d131003/anl/anl_mean_1900_VGRD_pres.nc
/gdex/data/d131003/anl/anl_mean_1900_HGT_pres.nc
/gdex/data/d131003/anl/anl_mean_1900_PRES_sfc.nc
/gdex/data/d131003/anl/anl_mean_1900_TMP_2m.nc
/gdex/data/d131003/anl/anl_mean_1900_SPFH_2m.nc
/gdex/data/d131003/anl/anl_mean_1900_UGRD_10m.nc
/gdex/data/d131003/anl/anl_mean_1900_VGRD_10m.nc
/gdex/data/d131003/anl/anl_mean_1900_UGRD_height.nc
/gdex/data/d131003/anl/anl_mean_1900_VGRD_height.nc
/gdex/data/d131003/anl/anl_mean_1900_CAPE_sfc.nc
/gdex/data/d131003/invariants/surface_height.nc
/gdex/data/d131003/invariants/land_mask.nc   # only if --land-only
```

Both files and variable metadata are discoverable via [GDEX's 20CRv3 inventory](https://gdex.ucar.edu/datasets/d131003/filelist/1/). The default region is **24–50°N, 125–66°W**: this is a CONUS *bounding box*, not a political boundary mask, and includes adjacent Canada/Mexico/coastal waters. `--land-only` masks oceans but **does not** remove Canada/Mexico. The original grid remains Gaussian (native ~0.7°, not regridded).

All file access is *read-only*. Only selected bounding-box/time/level arrays are loaded; NetCDF handles stay in the parent process. A local process pool evaluates disjoint sounding batches, with no MPI dependencies.

## Environment on Derecho

```bash
ssh YOUR_USER@derecho.hpc.ucar.edu
git clone https://github.com/kjhall01/SEV.git
cd SEV
module load conda/latest
mkdir -p /glade/work/$USER/conda-envs
mamba create -y -p /glade/work/$USER/conda-envs/sev python=3.11 numpy xarray netcdf4 metpy pytest -c conda-forge
conda activate /glade/work/$USER/conda-envs/sev
pip install -e . --no-deps
```

Alternatively, `pip install -e '.[test]'` from an appropriately configured Python environment. `numpy`, `xarray`, `netCDF4`, and `MetPy` are required for the full calculation.

## Preflight and benchmark (recommended first)

The `--inspect` option checks each native file and prints the actual variable names, axes, pressure levels, and time bounds **without computing a sounding**:

```bash
python -m sev.cli --year 1900 --month 5 --inspect
```

The provided batch job is a **32-core develop-queue job** using 8 3-hourly analysis times and 1,024 distributed CONUS grid points. Change `##PROJECTCODE##` in `jobs/derecho_benchmark.pbs` to your NCAR allocation code (or supply `qsub -A YOUR_CODE`, which overrides the directive). It benchmarks **recomputed SBCAPE/CIN plus wind diagnostics**, with archived `CAPE_sfc` as a reference:

```bash
qsub -A YOUR_PROJECT_CODE jobs/derecho_benchmark.pbs
qstat -u "$USER"
```

For a cheaper I/O + kinematic-only benchmark, run on a development node with `--skip-cape`; for a full-month dry-run use `--no-write` and leave `--max-times` and `--max-points` at zero. The sample's `--max-points` selector spreads locations through the bounding box, and computation can be restricted to land cells with `--land-only`.

Direct invocation outside PBS, e.g. while already running within an allocated compute node:

```bash
python -m sev.cli --year 1900 --month 5 --workers 8 --max-times 8 --max-points 512 --native-cape \
  --output-dir "$SCRATCH/sev/output"
```

Each job produces a month/parameter-labeled NetCDF and corresponding JSON file in `$SCRATCH/sev/output`. The JSON reports separate total wall time, data-read time, sum of worker CAPE time, sum of worker wind time, compute wall time, invalid-sounding counts, and measured throughput. Use the **measured timing**—not a guess—to estimate the full historical cost.

## Production monthly runs

**Do not submit these before the benchmark/preflight has succeeded.** After validating numerical results and timing, `jobs/derecho_monthly.pbs` requests one 128-core Derecho CPU node at economy priority and submits 12 independent array elements (one per month) for the selected year:

```bash
qsub -A YOUR_PROJECT_CODE -v YEAR=1900 jobs/derecho_monthly.pbs
```

A 128-core `main` node incurs a charge for **all 128 cores**, regardless of Python-worker utilization. The 32-core `develop` job charges only requested cores. PBS project codes, quotas, queue policies and wall times should be confirmed on your account before production. The production script is a starting point, not a fully managed multi-year submission system. Time-block output checkpointing/resume is not yet implemented, so one monthly job currently writes at completion.

## Physical definitions and caveats

- **SBCAPE/CIN:** MetPy's `surface_based_cape_cin`, initialized from **2-m temperature and specific humidity at surface pressure**. Subsurface pressure levels (`p >= ps`) are excluded. Temperature/specific humidity use their shared pressure levels, through 100 hPa. This is **CAPE evaluated from ensemble-mean fields**, not mean member-by-member CAPE.
- **Storm motion:** Bunkers (2000) right-moving estimate using a height-weighted mean of 0–6 km, the vector difference between 5.5–6 and 0–0.5 km mean winds, and a 7.5 m/s rightward deviation. Calm/degenerated hodographs are marked missing. This numerical implementation is independent of MetPy's unit-handling; it is benchmarked with synthetic wind profiles.
- **SRH:** Signed discrete hodograph-integral (0–1 km and 0–3 km AGL), using Bunkers motion and piecewise-linear interpolation to exact endpoints.
- **0–6 km bulk shear:** Absolute vector difference between 6 km AGL and the effective surface wind.
- **Vertical coordinate:** GDEX pressure-level height **minus native invariant surface height**, yielding AGL. Remove pressure-level data below local surface pressure. Use 10-m winds as z=0 and, by default, native 12–500 m AGL wind data; pressure-level winds above 500 m. `--no-height-winds` tests the pressure-only alternative.
- **Quality:** When the sounding does not reach 6 km AGL, there are too few valid thermodynamic levels, or data are otherwise invalid, write NaN instead of inventing an extrapolated diagnostic. Invalid counts are recorded in the benchmark JSON.
- **Comparison:** The optional `cape_native` output is the GDEX archived 20CRv3 CAPE field. Differences are expected because that field is produced by the model diagnostic, while the recomputation uses ensemble-mean profiles and MetPy's parcel physics.
- **Historical caveats:** 20CRv3 predominantly assimilates surface pressure observations. 3-hourly environmental diagnostics do not resolve convective storms. The fixed bounding box should be clipped to a true CONUS geometry before scientific aggregation. No claim of actual archive performance is made until a Derecho run completes.

## Tests

```bash
python -m pytest -q
```

Unit tests exercise physical definitions and an artificial miniature GDEX archive; the MetPy test automatically skips when MetPy is absent. **Actual archive metadata/shape and numerical climate validity remain to be tested on Derecho.**

## References

- [NCAR GDEX d131003](https://gdex.ucar.edu/datasets/d131003/dataaccess/)
- [GDEX POSIX access on HPC](https://projectpythia.org/gdex-cookbook/notebooks/services/posix-intro/)
- [MetPy SBCAPE/CIN](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.surface_based_cape_cin.html)
- [MetPy Bunkers storm motion](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.bunkers_storm_motion.html)
- [NCAR Derecho PBS charging](https://ncar-hpc-docs.readthedocs.io/en/stable/pbs/charging/)
