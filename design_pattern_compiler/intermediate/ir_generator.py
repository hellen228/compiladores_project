from typing import Any, Dict, List
from datetime import datetime
from core.models import ASTNode, ClassRelation, SemanticInfo, IntermediateRepresentation

class IntermediateCodeGenerator:
    """Generador de representación intermedia con información de clases y relaciones"""
    
    def __init__(self):
        self.ir = None
    
    def generate(self, ast_nodes: List[ASTNode], semantic_info: Dict[str, SemanticInfo], 
                relations: List[ClassRelation]) -> IntermediateRepresentation:
        """Genera la representación intermedia del análisis"""
        print("\nFASE 4: GENERACION DE CODIGO INTERMEDIO")
        print("-" * 40)
        
        classes = {}
        patterns = {}
        
        for class_node in [node for node in ast_nodes if node.node_type == "class"]:
            class_name = class_node.name
            
            classes[class_name] = {
                'name': class_name,
                'line': class_node.line,
                'bases': class_node.metadata.get('bases', []),
                'methods': self._normalize_methods(class_node),
                'variables': self._normalize_variables(class_node),
                'complexity': self._calculate_complexity(class_node)
            }
            
            if class_name in semantic_info:
                patterns[class_name] = semantic_info[class_name]
        
        filtered_relations = self._filter_relevant_relations(relations, patterns)
        
        global_metadata = {
            'total_classes': len(classes),
            'singleton_patterns': len([p for p in patterns.values() if p.pattern_type != "no_singleton"]),
            'total_relations': len(filtered_relations),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.ir = IntermediateRepresentation(
            classes=classes,
            patterns=patterns,
            relations=filtered_relations,
            global_metadata=global_metadata
        )
        
        self._print_ir_results()
        return self.ir
    
    def _filter_relevant_relations(self, relations: List[ClassRelation], 
                                 patterns: Dict[str, SemanticInfo]) -> List[ClassRelation]:
        """Filtra relaciones relevantes para el análisis de patrones"""
        singleton_classes = set()
        
        for class_name, pattern_info in patterns.items():
            if pattern_info.pattern_type != "no_singleton" and pattern_info.confidence >= 0.5:
                singleton_classes.add(class_name)
        
        relevant_relations = []
        
        for relation in relations:
            if (relation.target in singleton_classes or 
                relation.source in singleton_classes):
                relevant_relations.append(relation)
        
        return relevant_relations
    
    def _normalize_methods(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Normaliza información de métodos de la clase"""
        methods = []
        for child in class_node.children:
            if child.node_type == "method":
                method_info = {
                    'name': child.name,
                    'line': child.line,
                    'decorators': child.metadata.get('decorators', []),
                    'args': child.metadata.get('args', []),
                    'is_special': child.name.startswith('__') and child.name.endswith('__'),
                    'is_static': child.metadata.get('is_static', False),
                    'is_classmethod': child.metadata.get('is_classmethod', False),
                    'visibility': 'private' if child.name.startswith('_') else 'public'
                }
                methods.append(method_info)
        return methods
    
    def _normalize_variables(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Normaliza información de variables de la clase"""
        variables = []
        
        for child in class_node.children:
            if child.node_type == "class_variable":
                variables.append({
                    'name': child.name,
                    'line': child.line,
                    'value': child.metadata.get('value'),
                    'is_private': child.name.startswith('_'),
                    'is_class_var': True,
                    'type': self._infer_variable_type(child.metadata.get('value'))
                })
        
        for child in class_node.children:
            if child.node_type == "instance_variable":
                variables.append({
                    'name': child.name,
                    'line': child.line,
                    'value': child.metadata.get('value'),
                    'is_private': child.name.startswith('_'),
                    'is_class_var': False,
                    'type': child.metadata.get('inferred_type', 'Object')
                })
        
        return variables
    
    def _infer_variable_type(self, value) -> str:
        """Infiere el tipo de una variable"""
        if value is None:
            return "Object"
        elif isinstance(value, str):
            if value.isdigit():
                return "int"
            elif value.lower() in ['true', 'false']:
                return "bool"
            else:
                return "str"
        elif isinstance(value, (int, float, bool)):
            return type(value).__name__
        else:
            return "Object"
    
    def _calculate_complexity(self, class_node: ASTNode) -> int:
        """Calcula la complejidad de la clase"""
        complexity = 0
        complexity += len([c for c in class_node.children if c.node_type == "method"])
        complexity += len([c for c in class_node.children if c.node_type in ["class_variable", "instance_variable"]])
        return complexity
    
    def _print_ir_results(self):
        print(f"Representación intermedia generada:")
        print(f"   Clases: {len(self.ir.classes)}")
        print(f"   Patrones Singleton: {self.ir.global_metadata['singleton_patterns']}")
        print(f"   Relaciones: {len(self.ir.relations)}")