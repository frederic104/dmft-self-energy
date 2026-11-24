"""Build fail-closed DMFT provenance, target, replica, and split reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from dmft_data.research_audit import (
    audit_replicas,
    build_physical_state_manifest,
    build_target_registry,
    create_blocked_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/local/research_audit")
    )
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--convergence-registry", type=Path)
    parser.add_argument("--minimum-states", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = build_physical_state_manifest(
        args.source.resolve(),
        output / "physical_state_manifest.json",
        diagnostics_path=args.diagnostics.resolve() if args.diagnostics else None,
        metadata_path=args.metadata.resolve() if args.metadata else None,
    )
    targets = build_target_registry(
        output / "physical_state_manifest.json",
        output / "target_registry.json",
        convergence_registry_path=(
            args.convergence_registry.resolve() if args.convergence_registry else None
        ),
    )
    audit_replicas(
        output / "physical_state_manifest.json", output / "replica_audit.json"
    )
    splits = create_blocked_splits(
        output / "physical_state_manifest.json",
        output / "target_registry.json",
        output / "blocked_splits.json", minimum_states=args.minimum_states,
    )
    print(
        f"manifest_records={manifest['record_count']} "
        f"primary_targets={targets['primary_target_count']} "
        f"split_status={splits['status']}"
    )


if __name__ == "__main__":
    main()
