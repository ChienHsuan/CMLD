from .triplet import TripletLoss, SoftTripletLoss, SoftConsistencyLoss
from .multi_similarity import MultiSimilarityLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .triplet_xbm import TripletLossXBM
from .contrastive import ContrastiveLoss, PNContrastiveLoss
from .crd_loss import CRDLoss
from .kd import DistillKL
from .moco import MoCo
from .cam_aware_memory import CAPMemory
