import torch
from kornia.feature.loftr import LoFTR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Inicializujeme model s danou konfiguráciou
model = LoFTR(pretrained=None).to(device)

# Načítame uložené váhy
state_dict = torch.load("loftr_outdoor.pth", map_location=device)
model.load_state_dict(state_dict)

# Teraz je model pripravený na tréning s načítanými váhami
model.train()
