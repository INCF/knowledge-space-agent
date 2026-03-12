import json
from ttfr_estimator import estimate_ttfr

EXAMPLES = [
    {"datasource_id": "scr_005031_openneuro"},
    {"datasource_id": "scr_017612_ebrains"},
    {"datasource_id": "scr_002145_neuromorpho_modelimage"},
    {"content": "fMRI BOLD neuroimaging dataset with multiple subjects"},
    {"datasource_id": "unknown_id", "content": "ion channel database"},
]

def main():
    for i, kwargs in enumerate(EXAMPLES, 1):
        est = estimate_ttfr(**kwargs)
        print(f"Example {i}: {kwargs}")
        print(f"  Summary: {est.summary}")
        print("  Assumptions:")
        for a in est.assumptions:
            print(f"    - {a}")
        print("\nJSON format:")
        out = {
            "summary": str(est.summary),
            "phases": {k: str(v) for k, v in est.phases.items()},
            "assumptions": est.assumptions,
        }
        print(json.dumps(out, indent=2))
        print()

if __name__ == "__main__":
    main()
