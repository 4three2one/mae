import models_mae_origin
import models_mae

from thop import profile,clever_format
import  torch

img=torch.randn(1,3,224,224)
model = models_mae_origin.__dict__["mae_vit_base_patch16"](norm_pix_loss=False)
macs,params=profile(model,inputs=(img,0.75))
macs,params=clever_format([macs,params],"%.3f")

print(macs,params)

model = models_mae.__dict__["mae_vit_base_patch16"](norm_pix_loss=False)
macs,params=profile(model,inputs=(img,0.75))
macs,params=clever_format([macs,params],"%.3f")
print(macs,params)

model = models_mae_origin.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
macs,params=profile(model,inputs=(img,0.75))
macs,params=clever_format([macs,params],"%.3f")

print(macs,params)

model = models_mae.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
macs,params=profile(model,inputs=(img,0.75))
macs,params=clever_format([macs,params],"%.3f")
print(macs,params)
