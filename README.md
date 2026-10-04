# CA-HL-STGNCDE-GGO
# Water Quality Time-Series Forecasting

This project uses Graph Neural Controlled Differential Equations (GCDEs) to forecast water quality across multiple monitoring sites. The model represents observations as continuous control paths, models temporal dynamics and relationships between sites with an adaptive graph structure, and predicts target measurements over future time steps.

The default configuration in this repository uses 10 nodes, 24 historical time steps, and a 12-step prediction horizon. Each input time step contains 6 raw features. The data loader appends one time channel, so `input_dim` is set to 7. The model predicts one target channel.

## Environment and Dependencies

The following versions are taken from the package list for the environment used with this project:

| Package | Version |
| --- | --- |
| PyTorch | `2.6.0+cu124` |
| NumPy | `2.1.2` |
| Pandas | `2.2.3` |
| SciPy | `1.15.3` |
| scikit-learn | `1.6.1` |
| Matplotlib | `3.10.3` |
| openpyxl | `3.1.5` |
| TensorBoard | `2.19.0` |
| tqdm | `4.67.1` |
| torchcde | `0.2.5` |
| torchdiffeq | `0.2.5` |
| protobuf | `6.31.1` |

## Configuration

Edit `config_file` to change the data, model, and training parameters. The main settings are:

| Parameter | Default | Description |
| --- | ---: | --- |
| `num_nodes` | 10 | Number of monitoring sites |
| `lag` | 24 | Number of historical input time steps |
| `horizon` | 12 | Number of future time steps to predict |
| `input_dim` | 7 | Number of raw input features plus the time channel |
| `output_dim` | 1 | Number of target channels to predict |
| `embed_dim` | 10 | Node embedding dimension |
| `hid_dim` | 32 | Hidden state dimension |
| `num_layers` | 4 | Number of vector-field network layers |
| `batch_size` | 64 | Batch size |
| `epochs` | 100 | Maximum number of training epochs |
