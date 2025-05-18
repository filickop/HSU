import kornia.feature as KF
model = KF.LoFTR(pretrained='outdoor')
model.train()
help(model.train())
