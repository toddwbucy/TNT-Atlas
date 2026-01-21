# Atlas TNT Documentation

## Training Analysis

Detailed analysis documents for training runs:

| Run | Dataset | Status | Document |
|-----|---------|--------|----------|
| atlas_58m_fineweb_100M | fineweb-edu 100M | In Progress | [Analysis](analysis/run_atlas_58m_fineweb_100M.md) |

## Live Monitoring

Start the Streamlit dashboard for real-time training visualization:

```bash
./launch_dashboard.sh       # Default port 8501
./launch_dashboard.sh 8502  # Custom port
```

Or manually:
```bash
.venv/bin/streamlit run scripts/dashboard.py --server.port 8501 --theme.base dark
```

The dashboard shows:
- Live loss and PPL curves
- Smoothed curves with variance bands
- Train-Val PPL delta
- M_init layer norms from latest checkpoint
- Full checkpoint history table

## Figures

Generated figures are stored in `runs/figures/`:

- `ppl_chart.png` - Raw PPL curves
- `ppl_chart_smoothed.png` - Smoothed PPL with variance bands
