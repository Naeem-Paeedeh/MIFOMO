from enum import Enum


class InitializationType(Enum):
    Normal = 1
    Zeros = 2
    Uniform = 3
    Kaiming = 4
    AlmostDiagonal = 5
    Diagonal = 6
    

class Phase(Enum):
    Source = 1
    Intermediate = 2
    ClassificationMap = 3
    t_SNE = 4
    

class DistanceFunction(Enum):
    Euclidean = 1
    CosineDistance = 2
    CosineSimilarity = 3
    L1 = 4
    

# class PatchEmbeddingType(Enum):
#     Spatial = 1
#     Spectral = 2
#     Pixels = 3
