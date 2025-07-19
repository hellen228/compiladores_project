from typing import Any, Dict, List
from core.models import IntermediateRepresentation, OptimizedData
from core.models import SemanticInfo, ClassRelation, RelationType

class Optimizer:
    """Optimizador que mejora la detección y filtra falsos positivos"""
    
    def __init__(self):
        self.optimized_data = None
    
    def optimize(self, ir: IntermediateRepresentation) -> OptimizedData:
        """Ejecuta las optimizaciones sobre la representación intermedia"""
        print("\nFASE 5: OPTIMIZACION")
        print("-" * 40)
        
        optimized_patterns = {}
        optimized_relations = []
        false_positives_removed = []
        confidence_adjustments = {}
        
        for class_name, pattern_info in ir.patterns.items():
            optimized_pattern = self._optimize_pattern(pattern_info, ir.classes[class_name], ir.relations)
            
            if self._is_false_positive(optimized_pattern, ir.classes[class_name]):
                false_positives_removed.append(class_name)
                optimized_pattern.confidence *= 0.1
                optimized_pattern.pattern_type = "false_positive"
            
            original_confidence = pattern_info.confidence
            adjustment = optimized_pattern.confidence - original_confidence
            if abs(adjustment) > 0.05:
                confidence_adjustments[class_name] = adjustment
            
            optimized_patterns[class_name] = optimized_pattern
        
        optimized_relations = self._optimize_relations(ir.relations, optimized_patterns)
        
        self.optimized_data = OptimizedData(
            patterns=optimized_patterns,
            relations=optimized_relations,
            false_positives_removed=false_positives_removed,
            confidence_adjustments=confidence_adjustments
        )
        
        self._print_optimization_results()
        return self.optimized_data
    
    def _optimize_pattern(self, pattern_info: SemanticInfo, class_info: Dict[str, Any], 
                         relations: List[ClassRelation]) -> SemanticInfo:
        """Optimiza la información de un patrón específico"""
        optimized = SemanticInfo(
            class_name=pattern_info.class_name,
            pattern_type=pattern_info.pattern_type,
            confidence=pattern_info.confidence,
            evidences=pattern_info.evidences.copy(),
            violations=pattern_info.violations.copy()
        )
        
        # Bonificar si otras clases usan el singleton
        usage_relations = [r for r in relations if r.target == pattern_info.class_name 
                          and r.relation_type in [RelationType.ASSOCIATION, RelationType.DEPENDENCY]]
        if len(usage_relations) >= 2:
            optimized.confidence = min(optimized.confidence * 1.15, 1.0)
        elif len(usage_relations) == 1:
            optimized.confidence = min(optimized.confidence * 1.05, 1.0)
        
        return optimized
    
    def _optimize_relations(self, relations: List[ClassRelation], 
                          patterns: Dict[str, SemanticInfo]) -> List[ClassRelation]:
        """Optimiza y filtra las relaciones detectadas"""
        optimized_relations = []
        
        for relation in relations:
            source_pattern = patterns.get(relation.source)
            target_pattern = patterns.get(relation.target)
            
            valid_source = (source_pattern and 
                           source_pattern.pattern_type != "no_singleton" and 
                           source_pattern.pattern_type != "false_positive" and
                           source_pattern.confidence >= 0.5)
            
            valid_target = (target_pattern and 
                           target_pattern.pattern_type != "no_singleton" and 
                           target_pattern.pattern_type != "false_positive" and
                           target_pattern.confidence >= 0.5)
            
            if valid_source or valid_target:
                optimized_relations.append(relation)
        
        return optimized_relations
    
    def _is_false_positive(self, pattern_info: SemanticInfo, class_info: Dict[str, Any]) -> bool:
        """Detecta falsos positivos en la detección de patrones"""
        if pattern_info.confidence < 0.3:
            return True
        
        if len(pattern_info.violations) >= 2:
            return True
        
        return False
    
    def _print_optimization_results(self):
        print(f"Optimización completada:")
        print(f"   Falsos positivos removidos: {len(self.optimized_data.false_positives_removed)}")
        if self.optimized_data.confidence_adjustments:
            print(f"   Ajustes de confianza aplicados: {len(self.optimized_data.confidence_adjustments)}")
