# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | forward sanity | MLP u(x,t) | toy points | shape check | MUST | TODO | confirm output `[N,1]` |
| R002 | M0 | autograd sanity | compute u_t, u_xx | toy points | derivative exists | MUST | TODO | required for PDE residual |
| R003 | M1 | first PINN | full PINN | 1D heat equation | loss, MAE | MUST | TODO | first target implementation |
| R004 | M2 | visualize | full PINN | grid | heatmap + error | MUST | TODO | generate report figures |
| R005 | M3 | compare sparse data | MLP vs PINN | few supervised points | test MAE | SHOULD | TODO | proves physics constraint |
| R006 | M4 | ablation | remove loss terms | grid | residual, MAE | NICE | TODO | only after first success |

