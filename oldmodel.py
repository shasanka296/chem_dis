
import torch as t
import numpy as np
from rd import MyFirstNN
from rd import rd_kit_descriptors



model= MyFirstNN()
model.load_state_dict(t.load("dicl.pth"))
model.eval()
print(model(rd_kit_descriptors("CC1(CCCC(N1[O])(C)C)C")))
