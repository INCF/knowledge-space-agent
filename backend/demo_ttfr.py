from ttfr_estimator import estimate_ttfr

print("=" * 60)
print("TTFR Estimator Demo")
print("=" * 60)

test_cases = [
    {
        "name": "OpenNeuro fMRI Dataset",
        "datasource_id": "scr_005031_openneuro",
        "metadata": {"description": "Large-scale fMRI study with 200 participants investigating cognitive control"},
        "content": "BIDS-formatted neuroimaging data"
    },
    {
        "name": "DANDI Electrophysiology",
        "datasource_id": "scr_017571_dandi",
        "metadata": {"description": "Multi-electrode array recordings from mouse visual cortex"},
        "content": "NWB format neural recordings"
    },
    {
        "name": "Multimodal Dataset",
        "datasource_id": None,
        "metadata": {"description": "Combined fMRI and electrophysiology study of sensory processing"},
        "content": "Multimodal neuroimaging and neural recordings"
    },
    {
        "name": "Simple Morphology Database",
        "datasource_id": "scr_002145_neuromorpho_modelimage",
        "metadata": {"description": "Curated collection of neuron morphology reconstructions"},
        "content": "SWC format morphology files"
    }
]

for case in test_cases:
    print(f"\n{'─' * 60}")
    print(f"Dataset: {case['name']}")
    print(f"{'─' * 60}")
    
    estimate = estimate_ttfr(
        datasource_id=case["datasource_id"],
        metadata=case["metadata"],
        content=case["content"]
    )
    
    print(f"\nTime to First Result: {estimate.total}")
    print(f"\nBreakdown:")
    print(f"  - Access & setup: {estimate.access_setup}")
    print(f"  - Preprocessing: {estimate.preprocessing}")
    print(f"  - First output: {estimate.first_output}")
    print(f"\nAssumptions:")
    for assumption in estimate.assumptions:
        print(f" - {assumption}")
    
    print(f"\nJSON format:")
    import json
    print(json.dumps(estimate.to_dict(), indent=2))

print(f"\n{'=' * 60}")
print("Demo Complete!")
print("=" * 60)
