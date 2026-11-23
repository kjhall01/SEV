import numpy as np
import pytest
from sev.diagnostics import _srh, kinematic_diagnostics, wind_profile, surface_cape_cin


def test_srh_sign_and_endpoint_interpolation():
    z = np.array([0., 600., 1200.])
    u = z / 1000. * 10.  # linearly increases 10 m/s/km
    v = np.zeros_like(z)
    assert _srh(z, u, v, 1000., cu=0., cv=-5.) == pytest.approx(50.)
    assert _srh(z, u, v, 1000., cu=0., cv=5.) == pytest.approx(-50.)


def test_bunkers_right_mover_shear_and_srh_linear():
    z = np.arange(500., 7500., 500.)
    p = 1000. * np.exp(-z / 8000.)
    u = z * .005
    v = np.zeros_like(z)
    x = kinematic_diagnostics(1010., 100., p, z + 100., u, v, 0., 0.)
    assert x["shear06"] == pytest.approx(30., abs=1e-6)
    assert x["storm_u"] == pytest.approx(15., abs=1e-6)
    assert x["storm_v"] == pytest.approx(-7.5, abs=1e-6)
    assert x["srh01"] == pytest.approx(37.5, abs=1e-5)
    assert x["srh03"] == pytest.approx(112.5, abs=1e-5)


def test_height_winds_override_below_500m():
    p = np.array([990., 900., 700., 500., 300.])
    z = np.array([100., 1000., 3000., 6000., 9000.])
    u = np.array([80., 10., 20., 30., 40.])
    v = np.zeros_like(u)
    zz, uu, _ = wind_profile(1000., 0., p, z, u, v, 0., 0.,
                             np.array([12., 100., 500.]),
                             np.array([0.12, 1., 5.]), np.zeros(3))
    assert 100. in zz
    assert uu[np.flatnonzero(zz == 100.)[0]] == pytest.approx(1.)
    assert 6000. in zz


def test_fails_closed_if_no_6km_wind():
    z = np.array([500., 2000., 4000.])
    p = np.array([950., 800., 600.])
    with pytest.raises(ValueError, match="span"):
        kinematic_diagnostics(1000., 0., p, z, np.array([2., 5., 12.]),
                              np.zeros(3), 0., 0.)


def test_cape_unit_consistency_if_metpy_present():
    pytest.importorskip("metpy")
    p = np.array([1000., 950., 900., 850., 800., 700., 600., 500., 400., 300., 200., 100.])
    t = np.array([300., 297., 293., 289., 285., 277., 269., 258., 246., 231., 216., 200.])
    q = np.array([0.018, .014, .012, .010, .008, .005, .003, .0012, .0005, .0001, .00003, 0.])
    cape, cin = surface_cape_cin(1010., 301., .019, p, t, q)
    assert np.isfinite(cape) and cape >= 0
    assert np.isfinite(cin) and cin <= 0


def test_explicit_below_ground_thermodynamics():
    pytest.importorskip("metpy")
    p = np.array([1000., 925., 850., 700., 500., 300., 200., 100.])
    t = np.array([292., 290., 287., 277., 256., 231., 216., 195.])
    q = np.array([0.012, 0.010, .008, .004, .0015, .0002, .00001, 0.])
    cape, _ = surface_cape_cin(900., 295., .013, p, t, q)
    assert cape >= 0.
