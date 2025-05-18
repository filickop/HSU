import torch
import kornia.feature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Načítaj model s predtrénovanými váhami
model = kornia.feature.LoFTR(pretrained='outdoor').to(device)

# Získaj stav modelu (váhy)
state_dict = model.state_dict()

# Ulož váhy do súboru (napr. "loftr_outdoor.pth")
torch.save(state_dict, "loftr_outdoor.pth")

print("Váhy boli uložené do súboru 'loftr_outdoor.pth'")
