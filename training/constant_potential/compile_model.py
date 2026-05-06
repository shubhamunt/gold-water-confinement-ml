#!/usr/bin/env python3
"""
Compile a Franken checkpoint to a LAMMPS-compatible TorchScript model.

This script applies a runtime patch to FrankenMACE.descriptors to fix a TorchScript
type inference bug in Franken 0.5.0 + mace-torch 0.3.15:
  sc: Optional[torch.Tensor] = None   <-- TorchScript infers NoneType, not Optional[Tensor]
  fixed with: sc = torch.jit.annotate(Optional[torch.Tensor], None)
"""

import argparse
import torch
from typing import Optional, Tuple


def apply_torchscript_patch():
    """Replace FrankenMACE.descriptors with a TorchScript-compatible version."""
    from franken.backbones.wrappers.mace_wrap import FrankenMACE
    from franken.data import Configuration
    from mace.modules.utils import get_edge_vectors_and_lengths

    def patched_descriptors(self, data: Configuration) -> torch.Tensor:
        edge_index = data.edge_index
        shifts = data.shifts
        node_attrs = data.node_attrs
        assert edge_index is not None
        assert shifts is not None
        assert node_attrs is not None
        node_feats = self.node_embedding(node_attrs)  # type: ignore
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.atom_pos,
            edge_index=edge_index,
            shifts=shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)  # type: ignore
        rad_emb = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )  # type: ignore
        if torch.jit.isinstance(rad_emb, torch.Tensor):
            edge_feats, cutoff = rad_emb, None
        elif torch.jit.isinstance(rad_emb, Tuple[torch.Tensor, torch.Tensor]):
            edge_feats, cutoff = rad_emb
        elif torch.jit.isinstance(rad_emb, Tuple[torch.Tensor, Optional[float]]):
            edge_feats, cutoff = rad_emb
        else:
            edge_feats, cutoff = rad_emb

        lammps_class = None
        lammps_natoms = (0, 0)

        node_feats_list = []
        # Fix: torch.jit.annotate instead of type annotation to avoid TorchScript NoneType inference
        sc = torch.jit.annotate(Optional[torch.Tensor], None)
        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            if self.is_mace_v3_14:
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                    cutoff=cutoff,
                    first_layer=(i == 0),
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                )  # type: ignore
            elif self.is_mace_v3_13:
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                    first_layer=(i == 0),
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                )  # type: ignore
            else:
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    edge_index=edge_index,
                )  # type: ignore

            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )  # type: ignore
            irreps = product.linear.irreps_out
            invariant_slices = slice(0, irreps[0][0] * (2 * irreps[0][1][0] + 1))
            node_feats_list.append(node_feats[..., invariant_slices])
        return torch.cat(node_feats_list, dim=-1)

    FrankenMACE.descriptors = patched_descriptors
    print("Patch applied: FrankenMACE.descriptors now uses torch.jit.annotate for 'sc'")


def main():
    parser = argparse.ArgumentParser(
        description="Compile Franken checkpoint to LAMMPS TorchScript model"
    )
    parser.add_argument("--model_path", required=True, help="Path to best_ckpt.pt")
    parser.add_argument(
        "--rf_weight_id", type=int, default=None, help="RF weight head ID (default: None)"
    )
    args = parser.parse_args()

    apply_torchscript_patch()

    from franken.calculators.lammps_calc import LammpsFrankenCalculator

    save_path = LammpsFrankenCalculator.create_lammps_model(
        args.model_path, args.rf_weight_id
    )
    print(f"Compiled LAMMPS model saved to: {save_path}")


if __name__ == "__main__":
    main()
