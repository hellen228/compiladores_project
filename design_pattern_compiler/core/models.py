from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

#Contenedores/estructuras que TODOS usan

class RelationType(Enum):
    INHERITANCE = "inheritance"
    COMPOSITION = "composition"
    ASSOCIATION = "association"
    DEPENDENCY = "dependency"
    DECORATOR = "decorator"

@dataclass
class Token:
    type: any #(TokenType)
    value: str
    line: int
    column: int
    confidence: float = 1.0

@dataclass
class ClassRelation:
    source: str
    target: str
    relation_type: RelationType
    description: str
    line: int
    confidence: float = 1.0

@dataclass
class ASTNode:
    node_type: str
    name: str
    line: int
    children: List['ASTNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticInfo:
    class_name: str
    pattern_type: str
    confidence: float
    evidences: Dict[str, Any]
    violations: List[str] = field(default_factory=list)

@dataclass
class IntermediateRepresentation:
    classes: Dict[str, Dict[str, Any]]
    patterns: Dict[str, SemanticInfo]
    relations: List[ClassRelation]
    global_metadata: Dict[str, Any]

@dataclass
class OptimizedData:
    patterns: Dict[str, SemanticInfo]
    relations: List[ClassRelation]
    false_positives_removed: List[str]
    confidence_adjustments: Dict[str, float]