from .modules.caps import CAPS
from .modules.superpoint import SuperPoint
from .modules.superglue import SuperGlue
from .modules.d2net import D2Net
from .modules.r2d2 import R2D2
from .modules.patch2pix import Patch2Pix, NCNet, Patch2PixRefined
from .modules.loftr import LoFTR  # Cause warnings
from .modules.sift import SIFT
from .modules.dogaffnethardnet import DogAffNetHardNet
from .modules.cotr import COTR

try:
    from .modules.aspanformer import ASpanFormer
except ImportError as e:
    print(f"Can not import ASpanFormer: {e}")
    pass

try:
    import MinkowskiEngine
    import sys
    from pathlib import Path
    
    # To prevent naming conflict as D2Net also has module called lib   
    d2net_path = Path(__file__).parent / 'modules/../../third_party/d2net'
    sys.path.remove(str(d2net_path))
    
    from .modules.sparsencnet import SparseNCNet     
    use_sparsencnet = True
except ImportError as e:
    print(f"Can not import sparsencnet: {e}")
    pass    
