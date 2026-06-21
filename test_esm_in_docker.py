import torch
from transformers import AutoTokenizer, EsmModel
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Current device index:", torch.cuda.current_device())

try:
    print("Loading tokenizer...")
    t = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    print("Tokenizer loaded.")
    
    print("Loading model on cuda...")
    m = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda")
    print("Model loaded successfully on GPU!")
    
    m.eval()
    seq = "TTYKLILNLKQAKEEAIKELVDAGTAEKYFKLIANAKTVEGVWTLKDEIKTFTVTE"
    inputs = t(seq, return_tensors="pt").to("cuda")
    print("Inputs prepared.")
    
    with torch.no_grad():
        out = m(**inputs)
    print("Forward pass completed successfully!")
    print("Output shape:", out.last_hidden_state.shape)
except Exception as e:
    print("Exception occurred:")
    import traceback
    traceback.print_exc()
