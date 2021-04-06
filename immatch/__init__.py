from .modules.caps import CAPS
from .modules.superpoint import SuperPoint
from .modules.superglue import SuperGlue
from .modules.d2net import D2Net
from .modules.r2d2 import R2D2
from .modules.patch2pix import Patch2Pix, NCNet, Patch2PixRefined

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
    print('Can not imoprt sparsencnet')
    pass    