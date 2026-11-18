# Three specified longitudinal layouts

Exactly the A/B/C layouts recorded in `artifacts/grasp_longitudinal_0001/README.md` were evaluated. A and C use 40 mm AF; B uses 45 mm AF. Every rod retains the existing 120 mm axial length and the +90° relative x rotation that aligns its local y axis with palm-local z. This is an exploratory visual-alignment inference, not author calibration.

All three layouts passed fingertip IK reachability but failed the unchanged 9 mm initial contact-penetration limit. None was admitted to settling or the cold six-second four-finger hold. No grasp or training configuration was exported.

| Layout | IK residual, mm | Deepest contact, mm | Middle distal, mm | Thumb distal, mm |
| --- | ---: | ---: | ---: | ---: |
| A, 40 mm | 0.000713 | 33.750, middle medial | 11.137 | 16.165 |
| B, 45 mm | 0.000285 | 36.250, middle medial | 12.336 | 21.020 |
| C, 40 mm | 0.000000010 | 32.846, middle medial | 10.107 | 12.881 |

No candidate had palm/table contact. Full initial contacts, exact joint targets, target/measured fingertip positions, and source hashes are in [search_report.json](search_report.json). The artifact-local [driver](synthesize.py) rejects repeat execution after that report exists.

**Rod/link intersection is decisive; fingertip reachability is not the failure.** The middle fingertip can reach the far side of the rod while its medial link passes through the rod interior. Independent static inspection of layout C places the middle medial collision-box center at approximately palm coordinates `[.06367, .00252, .04900]` m, only 3.64 mm from the rod axis at `[.067, .004]`. The rod has a 20 mm cross-sectional apothem, and its axial span is `[-.03, .09]` m. That link lies well within the rod volume; the MuJoCo contacts above confirm substantial interpenetration. Thumb distal collisions independently violate the same gate.

Body length, collision geometry, stiffness, gates, and shared training sources were preserved. A different collision-free finger-chain approach would be needed before these longitudinal initializations could enter physical hold verification. This experiment does not establish the feasibility of the authors' geometry, whose axial dimensions and exact hand-frame placement remain unavailable.
