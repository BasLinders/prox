from prox.components.process_explorer import _build_js


def test_composed_js_bundles_vendor_libraries_offline():
    js = _build_js()

    # Both libraries are inlined (the app runs offline) and registered before use.
    assert js.index("cytoscape.use(") < js.index("export default function")
    assert "export default function" in js


def test_composed_js_has_no_dangling_source_map_reference():
    assert "sourceMappingURL" not in _build_js()
