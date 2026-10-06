# JJAP paper evidence and parameter correspondence

Source: Gao and Ohsawa, [JJAP DOI](https://doi.org/10.35848/1347-4065/ad1895), Sections 4-6 and 8, Figs. 2-4, 7-13. Values and numbering were cross-checked against the author-submitted manuscript and author-held journal galley. Galley printed article pages 8-10 contain the MNIST experiment and Fig. 13; printed page 11 contains the conclusion. These are source locators, not a redistributed proof or a claim that proof pagination equals final pagination.

| Parameter or condition | Paper | Current public computation |
|---|---|---|
| Dataset / topology | MNIST / 784x256x128x10 (Fig. 11) | MNIST configs keep topology; smoke is synthetic 4x5x3 |
| Technology / supply | 32 nm / 1.0 V | Numerical model; no validated transistor run |
| Reference | V0=VDD/2=0.5 V | 0.5 V |
| Synapse range | 0.01-1 mS, plus zero state (Â§6) | Dimensionless g=.01-1; inspected generator uses 1e-5 S per unit |
| Load | 1 mS (Â§6) | Current profile dimensionless loads 1, 1/3, 1/6 |
| Offsets | Fixed per simulated chip; normal distributions derived from Monte Carlo (Fig. 7) | Seeded numerical offset fixtures; not recovered original chip vectors |
| Gain | Finite op-amp gains characterized in Fig. 8 | Current-source gains, not a certified paper parameter set |
| Final output | No ReLU or follower (Â§4.2) | Linear subtractor readout |
| Evaluation | Ten simulated chips; SPICE (Â§6, Fig. 13) | Separate bounded CPU subsets in validation |

Paper conductances correspond to 1-100 kÎ© for nonzero synapses. The current source generator's scale corresponds to 100 kÎ©-10 MÎ©. This difference and the layer load profile must not be hidden by labeling the current config a paper reproduction. Common rescaling can preserve some static ratios only when all related conductances are scaled consistently; it does not establish equivalent transistor loading or timing.

The paper reports 97.2% without Vos, 10-50% for offline weights under fixed Vos, and 93-97% after HCST (Â§8), evaluated under each same simulated chip's offsets. No exact per-chip values are transcribed from the plot or fabricated here. The weights are updated for the offset-bearing circuit; the offsets are not removed. This fixed-offset comparison is distinct from the additional resistance-fluctuation experiment in Fig. 14.

The 1 us example waveform (Fig. 12) has 600 ns active and 400 ns precharge. It is an illustration setting, not a minimum cycle-time benchmark. Hardware fabrication is explicitly absent (Â§5, Fig. 10); neither a fabricated-chip result nor fully on-chip training is claimed.
