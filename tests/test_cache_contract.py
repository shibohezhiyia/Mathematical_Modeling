import pytest

from core.cache_contract import CacheContractError, CacheIdentity, build_cache_key


def _kwargs():
    return dict(ir_digest="ir-1", contract_digest="contract-1", data_view_digest="data-1",
                preprocessing_digest="prep-1", solver_version="solver-1", tolerance=1e-6,
                seed=7)


def test_intermediate_key_changes_for_every_material_dimension():
    base = build_cache_key(**_kwargs())
    assert base != build_cache_key(**{**_kwargs(), "seed": 8})
    assert base != build_cache_key(**{**_kwargs(), "tolerance": 1e-5})
    assert base != build_cache_key(**{**_kwargs(), "data_view_digest": "data-2"})


def test_api_cache_requires_model_prompt_and_config():
    with pytest.raises(CacheContractError, match="api_cache_requires"):
        CacheIdentity(**_kwargs()).key(api_response=True)
    assert CacheIdentity(**_kwargs(), model="deepseek", prompt_digest="p", config_digest="c").key(api_response=True)


def test_invalid_nonfinite_tolerance_is_rejected():
    with pytest.raises(CacheContractError, match="invalid_tolerance"):
        CacheIdentity(**{**_kwargs(), "tolerance": float("nan")})
