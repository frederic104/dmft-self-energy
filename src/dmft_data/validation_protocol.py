"""Evidence checklist for planning physics validation, never a readiness certificate."""

from __future__ import annotations


def validation_tasks(report: dict) -> list[dict]:
    """Describe what was observed and what still requires independent evidence."""
    raw = report["raw_runs"]
    observed = [r for r in raw if r.get("iteration_comparisons")]
    return [
        {
            "priority": 1,
            "check": "quantity_basis_energy_conventions",
            "status": "unresolved"
            if report["unresolved_definitions"]
            else "researcher_attested",
            "procedure": "Review input/log/publication evidence for G0 versus Weiss, Sigma/DC convention, U/J, filling and basis; map each source and target path.",
            "acceptance": "Traceable, mutually consistent evidence; do not infer from path names.",
        },
        {
            "priority": 2,
            "check": "stored_iteration_lineage",
            "status": "observed" if observed else "not_assessed",
            "procedure": "Compare final and all stored iterations; decompose adjacent changes per block; inspect stages around large changes with matching mu/DC/mixing/restart records.",
            "acceptance": "Document target selection and storage stage; equality and plateaus alone are not convergence.",
        },
        {
            "priority": 3,
            "check": "reference_uncertainty_and_symmetry",
            "status": "not_assessed",
            "procedure": "At fixed reviewed Hamiltonian, compare independent seed/bin statistics, occupancies and expected symmetries. Inspect block differences before any symmetry averaging.",
            "acceptance": "Use solver statistics and independent repeats; warmup/iteration replicas do not define standard errors.",
        },
        {
            "priority": 4,
            "check": "same_stage_impurity_Dyson",
            "status": "observed"
            if any(
                d.get("status") == "evaluated"
                for r in observed
                for d in r.get("dyson", [])
            )
            else "not_assessed",
            "procedure": "With reviewed same-stage Weiss, impurity G and solver Sigma, test matrix inverse(Weiss)-inverse(G)-Sigma and its sensitivity to inversion conditioning and postprocessing.",
            "acceptance": "Compare to numeric/solver uncertainty; residual agreement alone does not certify lattice self-consistency.",
        },
        {
            "priority": 5,
            "check": "moments_and_frequency_window",
            "status": "not_assessed",
            "procedure": "Obtain density-matrix/interaction-derived moments or wider stored meshes. Repeat tail fits across declared windows/orders, check stability and known limits; retain residual plots.",
            "acceptance": "Independent moment agreement and fit-window stability; first-window means are not Sigma-infinity.",
        },
        {
            "priority": 6,
            "check": "DMFT_self_consistency",
            "status": "not_assessed",
            "procedure": "Track G_imp versus G_loc in the same basis, mu, filling and applicable energy changes across consecutive complete iterations, with solver noise and mixing recorded.",
            "acceptance": "Predeclared observable-dependent criteria and supporting logs; a universal self-energy RMSE cutoff or fixed iteration count is insufficient.",
        },
        {
            "priority": 7,
            "check": "observable_error_budget",
            "status": "not_assessed",
            "procedure": "When H0/projectors/DC and reference uncertainty exist, perturb static and dynamic Sigma independently at declared energy scales; compare filling, orbital polarization and eligible metallic Z estimates. Validate any continuation independently before spectra/DOS claims.",
            "acceptance": "Set tolerances from target observable sensitivity and reference uncertainty, not DFT EDIFF or a global relative norm.",
        },
    ]
