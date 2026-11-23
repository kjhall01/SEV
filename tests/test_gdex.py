from pathlib import Path
import json
import numpy as np
import xarray as xr
from sev.gdex import GDEXReader, SOURCES
from sev.cli import main


def _archive(root: Path):
    (root / "anl").mkdir(parents=True)
    (root / "invariants").mkdir(parents=True)
    time = np.arange(np.datetime64("1900-05-01T00:00"),
                     np.datetime64("1900-05-02T00:00"), np.timedelta64(3, "h"))
    lat = np.array([20., 24., 38., 50., 60.])
    lon = np.array([225., 235., 265., 294., 300.])
    p = np.array([1000., 950., 900., 850., 700., 600., 500., 400., 300., 200., 100.])
    h = np.array([12., 20., 30., 50., 80., 100., 150., 200., 250., 300., 500.])
    coords = {"time": time, "latitude": lat, "longitude": lon}
    for key, tag in SOURCES.items():
        if key in ("t", "q", "u", "v", "z"):
            dims = ("time", "level", "latitude", "longitude")
            c = dict(coords, level=xr.DataArray(p, dims="level", attrs={"units": "hPa"}))
            if key == "z":
                values = np.tile((np.array([100., 550., 1000., 1500., 3100., 4300.,
                                           5700., 7300., 9200., 11500., 16000.]) + 100.)[None,:,None,None],
                                 (len(time), 1, len(lat), len(lon)))
            elif key == "u":
                values = np.tile(np.linspace(1., 40., len(p))[None,:,None,None],
                                 (len(time), 1, len(lat), len(lon)))
            elif key == "v":
                values = np.tile(np.linspace(1., 6., len(p))[None,:,None,None],
                                 (len(time), 1, len(lat), len(lon)))
            elif key == "t":
                values = np.tile(np.linspace(300., 196., len(p))[None,:,None,None],
                                 (len(time), 1, len(lat), len(lon)))
            else:
                values = np.tile(np.linspace(.018, 1.e-8, len(p))[None,:,None,None],
                                 (len(time), 1, len(lat), len(lon)))
        elif key in ("uh", "vh"):
            dims = ("time", "height", "latitude", "longitude")
            c = dict(coords, height=xr.DataArray(h, dims="height", attrs={"units":"m"}))
            base = np.linspace(0, 5, len(h)) if key == "uh" else np.linspace(0, 1., len(h))
            values = np.tile(base[None,:,None,None], (len(time),1,len(lat),len(lon)))
        else:
            dims = ("time", "latitude", "longitude")
            c = coords
            base = {"ps": 101000., "t2": 301., "q2": 0.018,
                    "u10": 0., "v10": 0., "cape_native": 800.}[key]
            values = np.full((len(time),len(lat),len(lon)), base)
        da = xr.DataArray(values.astype('f4'), dims=dims, coords=c, name=tag)
        xr.Dataset({tag: da}).to_netcdf(root/"anl"/f"anl_mean_1900_{tag}.nc", engine="scipy")
    for fn, value in (("surface_height.nc", 100.), ("land_mask.nc", 1.)):
        da = xr.DataArray(np.full((len(lat),len(lon)), value, dtype="f4"),
                          dims=("latitude", "longitude"),
                          coords={"latitude": lat, "longitude": lon}, name="value")
        da.to_dataset().to_netcdf(root/"invariants"/fn, engine="scipy")


def test_native_reader(tmp_path):
    _archive(tmp_path)
    with GDEXReader(tmp_path, 1900, 5, cape=True, wind=True,
                    height_winds=True, native_cape=True) as reader:
        d = reader.describe()
        assert d["ntimes"] == 8
        assert d["npoints"] == 9
        assert reader.p_wind[0] == 1000.
        assert reader.p_thermo[-1] == 100.
        assert len(reader.low_heights) == 11
        block = reader.load_block(0, 2)
        assert block["ps"].shape == (2, 9)
        assert block["ps"][0,0] == 1010.
        assert block["t"].shape == (2, 11, 9)
        assert block["uh"].shape == (2, 11, 9)


def test_end_to_end_kinematic_benchmark(tmp_path):
    root = tmp_path / "source"
    dest = tmp_path / "out"
    _archive(root)
    assert main(["--root", str(root), "--year", "1900", "--month", "5",
                 "--workers", "2", "--time-block", "2", "--tile-size", "3",
                 "--max-times", "3", "--max-points", "6",
                 "--skip-cape", "--native-cape", "--output-dir", str(dest)]) == 0
    output = next(dest.glob("*.nc"))
    with xr.open_dataset(output) as ds:
        assert ds.sizes == {"time": 3, "latitude": 3, "longitude": 3}
        assert "sbcape" not in ds
        assert "shear06" in ds and "srh01" in ds and "cape_native" in ds
        assert int(np.isfinite(ds.shear06.values).sum()) == 18
        assert float(ds.cape_native.max()) == 800.
    report = json.loads(next(dest.glob("*.json")).read_text())
    assert report["n_columns_attempted"] == 18
    assert report["counts"]["wind_valid"] == 18
    assert report["io_wall_seconds"] >= 0
