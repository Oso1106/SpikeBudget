# Dense Low-LR Scratch Screen Summary

| Theory | Seeds | Mean Gain BPB | Mean Gain % | Pass Count | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `lr3e4` | 2 | 0.089184 | 2.657603 | 0 | `screen_only_or_rejected` |
| `steps1000` | 2 | 0.033083 | 0.987257 | 0 | `screen_only_or_rejected` |
| `steps600` | 2 | 0.016846 | 0.502135 | 0 | `screen_only_or_rejected` |
| `d512` | 2 | -0.031326 | -0.933844 | 0 | `screen_only_or_rejected` |
| `d768` | 2 | -0.063719 | -1.899985 | 0 | `screen_only_or_rejected` |
| `lr2e3` | 2 | -0.115043 | -3.430102 | 0 | `screen_only_or_rejected` |

| Run | Seed | Theory | Final BPB | Baseline BPB | Gain BPB | Gain % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `dense_low_lr_scratch_s1234_baseline300_d512_l4_lr1e3` | 1234 | `baseline300` | 3.361127 | 3.361127 | 0.000000 | 0.000000 |
| `dense_low_lr_scratch_s1234_d512_l6_steps300_lr1e3` | 1234 | `d512` | 3.392591 | 3.361127 | -0.031465 | -0.936139 |
| `dense_low_lr_scratch_s1234_d768_l4_steps300_lr1e3` | 1234 | `d768` | 3.416709 | 3.361127 | -0.055582 | -1.653672 |
| `dense_low_lr_scratch_s1234_lr2e3_d512_l4` | 1234 | `lr2e3` | 3.465986 | 3.361127 | -0.104860 | -3.119782 |
| `dense_low_lr_scratch_s1234_lr3e4_d512_l4` | 1234 | `lr3e4` | 3.254527 | 3.361127 | 0.106600 | 3.171544 |
| `dense_low_lr_scratch_s1234_steps1000_d512_l4_lr1e3` | 1234 | `steps1000` | 3.345432 | 3.361127 | 0.015694 | 0.466931 |
| `dense_low_lr_scratch_s1234_steps600_d512_l4_lr1e3` | 1234 | `steps600` | 3.343582 | 3.361127 | 0.017544 | 0.521972 |
| `dense_low_lr_scratch_s4321_baseline300_d512_l4_lr1e3` | 4321 | `baseline300` | 3.347916 | 3.347916 | 0.000000 | 0.000000 |
| `dense_low_lr_scratch_s4321_d512_l6_steps300_lr1e3` | 4321 | `d512` | 3.379103 | 3.347916 | -0.031187 | -0.931548 |
| `dense_low_lr_scratch_s4321_d768_l4_steps300_lr1e3` | 4321 | `d768` | 3.419772 | 3.347916 | -0.071856 | -2.146299 |
| `dense_low_lr_scratch_s4321_lr2e3_d512_l4` | 4321 | `lr2e3` | 3.473142 | 3.347916 | -0.125226 | -3.740421 |
| `dense_low_lr_scratch_s4321_lr3e4_d512_l4` | 4321 | `lr3e4` | 3.276148 | 3.347916 | 0.071768 | 2.143662 |
| `dense_low_lr_scratch_s4321_steps1000_d512_l4_lr1e3` | 4321 | `steps1000` | 3.297443 | 3.347916 | 0.050473 | 1.507584 |
| `dense_low_lr_scratch_s4321_steps600_d512_l4_lr1e3` | 4321 | `steps600` | 3.331769 | 3.347916 | 0.016147 | 0.482299 |

| Decision Field | Value |
| --- | --- |
| Direction | `no_positive_theory_yet` |
| Best theory | `lr3e4` |
| Reason | No hypothesis passed the two-seed, >=5%, >=2x-noise rule. |
