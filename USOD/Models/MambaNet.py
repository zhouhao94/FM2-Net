from .Transformer_depth import Transformer
from .Transformer import token_Transformer
from .DAM_module import *
from .decoders.MambaDecoder import MambaDecoder
from .encoders.dual_vmamba import vssm_tiny as backbone

class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        # Encoder
        self.backbone = backbone()

        # Decoder
        self.channels = [96, 192, 384, 768] # for small and tiny vmamba, and [128, 256, 512, 1024] for base vmamba
        self.deep_supervision = True
        self.decoder = MambaDecoder(img_size=[224, 224], in_channels=self.channels, num_classes=1, embed_dim=self.channels[0], deep_supervision=self.deep_supervision)

    def forward(self, image_Input, depth_Input):
        B, _, _, _ = image_Input.shape

        fused_feat = self.backbone(image_Input, depth_Input) #[16, 96, 56, 56], [16, 192, 28, 28], [16, 384, 14, 14], [16, 768, 7, 7]
        outputs = self.decoder.forward(fused_feat) # [16, 1, 224, 224] * 1 or 4

        return outputs
