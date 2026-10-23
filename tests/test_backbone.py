"""Light tests for backbone helpers that need no model download."""

from atom.codec.backbone import is_cached, mid_stack_layer


def test_mid_stack_layer_maps_depth_to_index():
    assert mid_stack_layer(43, 0.0) == 0
    assert mid_stack_layer(43, 1.0) == 42
    assert mid_stack_layer(43, 0.5) == 21
    assert mid_stack_layer(43, 0.71) == 30  # the L30 last-token prior sits at depth ~0.71


def test_is_cached_false_for_bogus_id_without_network():
    # local_files_only under the hood: no network, just a cache miss -> (False, None).
    ok, path = is_cached("definitely-not/a-real-model-xyz-000")
    assert ok is False and path is None
