import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="AggNet Model")
parser.add_argument(
    "-bsz", "--batch", help="Batch_size", required=False, type=int, default=8
)
parser.add_argument(
    "-emb",
    "--emb_dim",
    help="Embedding Dimension Size",
    required=False,
    type=int,
    default=200,
)
parser.add_argument(
    "-ehd",
    "--enc_hid_dim",
    help="Encoder Hidden Dimension Size",
    required=False,
    type=int,
    default=100,
)
parser.add_argument(
    "-dhd",
    "--dec_hid_dim",
    help="Decoder Hidden Dimension Size",
    required=False,
    type=int,
    default=200,
)  # Should always be 2 * enc_hid_dim
parser.add_argument(
    "-ats", "--attn_size", help="Attention Size", required=False, type=int, default=200
)
parser.add_argument(
    "-e",
    "--num_epochs",
    help="Number of Epochs to Run",
    required=False,
    type=int,
    default=40,
)
parser.add_argument(
    "-d", "--dropout", help="Dropout Rate", required=False, type=float, default=0.05
)
parser.add_argument(
    "-gd",
    "--gru_drop",
    help="GRU Dropout Rate",
    required=False,
    type=float,
    default=0.2,
)
parser.add_argument(
    "-lr", "--lr", help="Learinng Rate", required=False, type=float, default=2.5e-4
)
parser.add_argument(
    "-ld", "--load", help="Load Model Checkpoint", required=False, default=None
)
parser.add_argument(
    "-cp",
    "--ckpt_path",
    help="Path to save Checkpoints",
    required=False,
    default="./../../../scratch/Aggnet_ckpts/ablations_cnn",
)
parser.add_argument(
    "-n", "--name", help="Name Your Model", required=False, default="default"
)
parser.add_argument(
    "-clip", "--clip", help="gradient clipping", required=False, type=float, default=10
)
parser.add_argument(
    "-gpu",
    "--gpu",
    help="Run in GPU or not",
    required=False,
    type=str2bool,
    default=True,
)
parser.add_argument(
    "-m", "--model", help="1 is MLM, 2 is MLM+GLMP", required=False, type=int, default=2
)
parser.add_argument(
    "-ds",
    "--dataset",
    help="1 is Incar, 2 is Camrest, 3 is MultiWoz",
    required=False,
    type=int,
    default=1,
)
parser.add_argument(
    "-dp",
    "--data",
    help="Dataset path",
    required=False,
    default="../Incar_sketch_standard/",
)
parser.add_argument(
    "-lg",
    "--logs",
    help="Print Logs or not",
    required=False,
    type=str2bool,
    default=False,
)
parser.add_argument(
    "-test",
    "--test",
    help="Test or Train",
    required=False,
    type=str2bool,
    default=False,
)
parser.add_argument(
    "-hp", "--hops", help="Number of Memory Hops", required=False, type=int, default=3
)
parser.add_argument(
    "-s", "--seed", help="Enter Manual Seed", required=False, type=int, default=None
)
parser.add_argument(
    "-v", "--vocab", help="Vocab Name", required=False, default="vocab.json"
)
parser.add_argument(
    "-tf",
    "--teacher_forcing",
    help="Teacher Forcing",
    type=float,
    required=False,
    default=0.9,
)
parser.add_argument(
    "-abl_g",
    "--abl_glove",
    help="Glove Embedding Use or not",
    required=False,
    type=str2bool,
    default=True,
)
parser.add_argument(
    "-abl_bs",
    "--abl_beta_supvis",
    help="Beta Supervision Loss False is disable, True is both labels",
    required=False,
    type=str2bool,
    default=True,
)
parser.add_argument(
    "-abl_gb",
    "--abl_global_beta",
    help="Global Beta enable/disable",
    required=False,
    type=str2bool,
    default=True,
)
parser.add_argument(
    "-abl_wd",
    "--abl_window",
    help="Window CNN enable/disable",
    required=False,
    type=int,
    default=1,
)
parser.add_argument(
    "-abl_sml",
    "--abl_similarity_loss",
    help="Similarity Loss enable/disable",
    required=False,
    type=str2bool,
    default=True,
)
parser.add_argument(
    "-abl_gle",
    "--abl_glove_rnd",
    help="0 no glove, 1 only non entities glove, 2 all glove",
    required=False,
    type=int,
    default=2,
)

args = vars(parser.parse_args())
print(str(args), flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not args["gpu"]:
    DEVICE = torch.device("cpu")
if DEVICE.type == "cuda":
    USE_CUDA = True
else:
    USE_CUDA = False
print("Using Device:", DEVICE, flush=True)

# print("USE_CUDA: "+str(USE_CUDA))
