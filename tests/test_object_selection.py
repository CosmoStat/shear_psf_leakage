"""Row selection and output tagging of the object-wise leakage."""

import numpy as np
import pytest
from astropy.io import fits

from shear_psf_leakage import run_object as run


@pytest.fixture
def obj(tmp_path):
    n = 20
    cols = [
        fits.Column(name="idx", format="K", array=np.arange(n)),
        fits.Column(name="e1", format="D", array=np.zeros(n)),
    ]
    path = tmp_path / "cat.fits"
    fits.BinTableHDU.from_columns(cols).writeto(path)
    o = run.LeakageObject()
    o._params["input_path_shear"] = str(path)
    o._params["output_dir"] = str(tmp_path)
    return o


def test_selection_keeps_selected_rows(obj):
    sel = np.arange(20) % 3 == 0
    with obj.temporarily_read_data(selection=sel) as dat:
        np.testing.assert_array_equal(dat["idx"], np.arange(20)[sel])
    assert obj._dat is None


def test_selection_length_mismatch_raises(obj):
    with pytest.raises(ValueError):
        obj.read_data(selection=np.ones(5, dtype=bool))


def test_preloaded_catalogue_is_kept(obj):
    obj.read_data()
    loaded = obj._dat
    with obj.temporarily_read_data() as dat:
        assert dat is loaded
    assert obj._dat is loaded


def test_suffix_tags_output_base(obj):
    base = obj.get_out_base(True, "lin")
    assert base.endswith("PSF_e_vs_e_gal_order-lin_mix-True")
    assert obj.get_out_base(True, "lin", suffix="tomo_bin_1") == (
        f"{base}-tomo_bin_1"
    )
