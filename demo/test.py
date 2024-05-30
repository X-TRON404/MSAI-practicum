import numpy as np


if __name__ == "__main__":
    # np.save("./lung1.npy", (np.random.random((512, 512, 128))>0.5).astype(np.int32))
    np.save("./dose1.npy", np.random.random((512, 512, 128)).astype(np.float32))