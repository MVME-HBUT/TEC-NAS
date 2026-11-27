from .blocks_basic import *
from .SuperResConvK1KXK1 import *
from .SuperResK1DWK1 import *
from .SuperResConvK1KX import *
from .SuperGhostv1 import *
from .SuperResConvKXKX import *
# from .SuperConvnextDWK1K1 import *

__all_blocks__ = {
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
    'ResConvK1KXK1': ResConvK1KXK1,
    'SuperResConvK1KXK1': SuperResConvK1KXK1,
    'ResK1DWK1': ResK1DWK1,
    'SuperResK1DWK1': SuperResK1DWK1,
    'ResConvK1KX': ResConvK1KX,
    'SuperResConvK1KX': SuperResConvK1KX,
    'SuperGhostv1': SuperGhostv1,
    'Ghostv1': Ghostv1,
    'SuperResConvKXKX': SuperResConvKXKX,
    'ResConvKXKX': ResConvKXKX,
    # 'SuperConvnextDWK1K1': SuperConvnextDWK1K1,
    # 'ConvnextDWK1K1': ConvnextDWK1K1,
}

