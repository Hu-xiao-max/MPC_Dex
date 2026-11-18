# Findings: object-shape diagnostic (circular object, archived 0.1 s period)

## Cold-reset MPC with the archived recipe on a cylinder (2026-10-08)

| diameter | seed 42 | controls to a strict force-qualified turn |
| --- | --- | --- |
| 40 mm | success | 8 (0.8 s; the hexagon needs 13) |
| 46 mm (the hexagon's corner diameter) | success | 13 |

The archived initial grasp (recorded on the hexagon) holds the cylinder without any change.

## Open-loop tolerance, 40 mm cylinder, seed 42 (fixed rays, 3 per control, 2 s hold; successes of 3)

| control j | r = .0003 / .001 / .003 / .01 / .03 / .1 |
| --- | --- | 
| 0-2 | 0 / 0 / 0 / 0 / 0 / 0 |
| 3 | 1 / 3 / 1 / 0 / 0 / 1 |
| 4 | 2 / 3 / 2 / 1 / 0 / 1 |
| 5 | 3 / 3 / 3 / 2 / 3 / 2 |
| 6-7 | 3 / 3 / 3 / 3 / 3 / 3 |

The structure is the hexagon's (an intolerant prefix down to 0.0003, then a tolerant tail of about five control decisions), but the turn is shorter, so the intolerant prefix is three controls instead of the hexagon's five to nine: the early contact-switching events of the grasp remain bifurcations, the later edge-rolling events of the hexagon are gone.

## Open-loop tolerance, 46 mm cylinder (the hexagon's corner diameter), seed 42, 13 controls

| control j | r = .0003 / .001 / .003 / .01 / .03 / .1 (successes of 3) |
| --- | --- |
| 0-3 | 1 / 0 / 0 / 0 / 0 / 0 (one ray survives 0.0003 at each) |
| 4-5 | 0 / 1 / 0 / 0 / 0-1 / 0 |
| 6-9 | 0-3 / 1-2 / 1-2 / 1-2 / 0-3 / 0-1 (mixed) |
| 10 | 3 / 3 / 3 / 3 / 2 / 1 |
| 11-12 | 3 / 3 / 3 / 3 / 3 / 3 |

The larger cylinder needs as many controls as the hexagon (13) and its intolerant prefix is as long (controls 0-5, with sporadic survivals), so the shorter intolerant prefix of the 40 mm cylinder comes with its shorter turn (8 controls), not from the round shape alone: what shrinks is the number of contact-switching decisions a turn needs. Of the two, the 40 mm cylinder (the archived diameter, so the archived grasp and object mass apply) is the candidate for a learning campaign.

## All four cylinder demonstrations (2026-10-08, `probe_P0100ms_circular*_seeds42_47.json`; 4 of 4 attempts strict successes)

| object | seed | controls | controls with zero tolerance up to r = .01 | controls fully tolerant (3/3 up to r = .01) |
| --- | --- | --- | --- | --- |
| 40 mm | 42 | 8 | 0, 1, 2 | 6, 7 |
| 40 mm | 43 | 21 | 0, 1, 4, 5, 7, 10 | 16-20 |
| 46 mm | 42 | 13 | none fully zero (one ray survives 0.0003 at 0-3) | 10-12 |
| 46 mm | 43 | 15 | 0, 1, 3, 4, 5, 8 | 13 |

The 8-control turn of the first 40 mm seed is not typical: the second seed takes 21 controls with six intolerant ones spread over the first eleven, and the 46 mm demonstrations have five to six. Across the four, the intolerant decisions are the contact-switching controls of the first half of the turn and the fully tolerant ones are the last two to five; a cylinder turn therefore still carries three to six bifurcating decisions, against five to nine for the hexagon. The MPC itself is more reliable on the cylinder (4/4 strict successes against about 2/3 on the hexagon), which would make demonstration collection cheaper, but the learnability obstacle (decisions with no open-loop slack) is reduced, not removed.
