import torch
import time
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count
from tqdm import tqdm

from models.cnn3d import CNN3D


if __name__ == "__main__":
    n_iters_warmup = 500
    n_iters = 1000

    model = CNN3D().cuda()

    # FPS test
    data = torch.randn((1, 1, 64, 128, 128)).cuda()

    start = time.time()
    for i in tqdm(range(n_iters_warmup)):
        _ = model(data)
    t = time.time() - start
    print(
        f"Warm up finished - time (total): {t} seconds; fps: {n_iters_warmup/t}; time (per-sample): {t * 1000 / n_iters_warmup} ms"
    )

    start = time.time()
    for i in tqdm(range(n_iters)):
        _ = model(data)
    t = time.time() - start
    print(
        f"Test finished - time (total): {t} seconds; fps: {n_iters/t}; time (per-sample): {t * 1000 / n_iters} ms"
    )

    # MACs / #Params
    flops = FlopCountAnalysis(model, torch.randn((1, 1, 64, 128, 128)).cuda())
    print(f"MACs (G): {flops.total() / 1e+9}")
    print(f"#Params (M): {parameter_count(model)['']/1000000}")
