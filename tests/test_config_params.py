from __future__ import annotations

from src.config_params import Config


def test_config_defaults_when_env_not_set(monkeypatch) -> None:
    monkeypatch.delenv("PRESSUREPROCESS_ROOT_DIR", raising=False)
    cfg = Config()

    assert cfg.ROOT_DIR == "data/bump1"
    assert cfg.RAW_CAL_BASE == "data/bump1/raw_calib"
    assert cfg.RAW_BASE == "data/bump1/raw_wallp"
    assert cfg.TF_BASE == "data/bump1/calibration"
    assert cfg.PH_RAW_FILE == "data/bump1/pressure/G_wallp_SU_raw.hdf5"


def test_config_derives_paths_from_env_root(monkeypatch) -> None:
    monkeypatch.setenv("PRESSUREPROCESS_ROOT_DIR", "/tmp/custom/root/")
    cfg = Config()

    assert cfg.ROOT_DIR == "/tmp/custom/root/"
    assert cfg.RAW_CAL_BASE == "/tmp/custom/root/raw_calib"
    assert cfg.RAW_BASE == "/tmp/custom/root/raw_wallp"
    assert cfg.TF_BASE == "/tmp/custom/root/calibration"
    assert cfg.PH_PROCESSED_FILE == "/tmp/custom/root/pressure/G_wallp_SU_production.hdf5"
