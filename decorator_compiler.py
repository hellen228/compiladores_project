import ast
import sys
import os
import re
import subprocess
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from data_structure import ASTNode, ClassRelation, RelationType, IntermediateRepresentation
from data_structure import OptimizedData, PatternInfo

# =============================================================================
# DEFINICIÓN DE ESTRUCTURAS DE DATOS
# =============================================================================

class TokenType(Enum):
    COMPONENT_REF = "component_reference"
    DELEGATION_CALL = "delegation_call"
    WRAPPER_FUNCTION = "wrapper_function"
    DECORATOR_SYNTAX = "decorator_syntax"
    FUNCTOOLS_WRAPS = "functools_wraps"
    RETURN_WRAPPER = "return_wrapper"
    ARGS_KWARGS = "args_kwargs"
    CLASS_DEF = "class_definition"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    confidence: float = 1.0

# =============================================================================
# FASE 1: ANALIZADOR LÉXICO
# =============================================================================

class LexicalAnalyzer:
    def __init__(self):
        self.tokens = []
        self.symbol_table = {}
        
        # Patrones de tokens específicos para el patrón Decorator
        self.patterns = {
            TokenType.COMPONENT_REF: [
                r'self\._component\b', r'self\.component\b',
                r'self\._wrapped\b', r'self\.wrapped\b'
            ],
            TokenType.DELEGATION_CALL: [
                r'self\._component\.\w+\(', r'self\.component\.\w+\(',
                r'self\._wrapped\.\w+\(', r'self\.wrapped\.\w+\('
            ],
            TokenType.WRAPPER_FUNCTION: [
                r'def\s+wrapper\s*\(', r'def\s+inner\s*\(',
                r'def\s+decorated\s*\('
            ],
            TokenType.DECORATOR_SYNTAX: [
                r'@\w+\b', r'@\w+\.\w+\b'
            ],
            TokenType.FUNCTOOLS_WRAPS: [
                r'@wraps\s*\(', r'@functools\.wraps\s*\(',
                r'from\s+functools\s+import\s+wraps'
            ],
            TokenType.RETURN_WRAPPER: [
                r'return\s+wrapper\b', r'return\s+inner\b',
                r'return\s+decorated\b'
            ],
            TokenType.ARGS_KWARGS: [
                r'\*args', r'\*\*kwargs'
            ],
            TokenType.CLASS_DEF: [r'class\s+\w+']
        }
    
    def analyze(self, source_code: str) -> List[Token]:
        print("Iniciando análisis léxico...")
        
        lines = source_code.split('\n')
        self.tokens = []
        
        for line_num, line in enumerate(lines, 1):
            for token_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        token = Token(
                            type=token_type,
                            value=match.group(),
                            line=line_num,
                            column=match.start(),
                            confidence=self._calculate_confidence(token_type, match.group())
                        )
                        self.tokens.append(token)
                        self._update_symbol_table(token)
        
        print(f"Tokens encontrados: {len(self.tokens)}")
        return self.tokens
    
    def _calculate_confidence(self, token_type: TokenType, value: str) -> float:
        confidence_table = {
            TokenType.COMPONENT_REF: {
                '_component': 0.95, 'component': 0.85, 
                '_wrapped': 0.90, 'wrapped': 0.80
            },
            TokenType.DELEGATION_CALL: {
                '_component.': 0.95, 'component.': 0.85,
                '_wrapped.': 0.90, 'wrapped.': 0.80
            },
            TokenType.WRAPPER_FUNCTION: {
                'wrapper': 0.95, 'inner': 0.85, 'decorated': 0.90
            },
            TokenType.FUNCTOOLS_WRAPS: {
                '@wraps': 0.98, 'functools.wraps': 0.98
            }
        }
        
        for key, conf in confidence_table.get(token_type, {}).items():
            if key.lower() in value.lower():
                return conf
        return 0.60
    
    def _update_symbol_table(self, token: Token):
        if token.value not in self.symbol_table:
            self.symbol_table[token.value] = {
                'type': token.type,
                'count': 0,
                'lines': []
            }
        self.symbol_table[token.value]['count'] += 1
        self.symbol_table[token.value]['lines'].append(token.line)

# =============================================================================
# FASE 2: ANALIZADOR SINTÁCTICO
# =============================================================================

class SyntaxAnalyzer:
    def __init__(self):
        self.ast_nodes = []
        self.syntax_errors = []
        self.relations = []
        
    def parse(self, source_code: str, tokens: List[Token]) -> Tuple[List[ASTNode], List[ClassRelation]]:
        print("Iniciando análisis sintáctico...")
        
        try:
            tree = ast.parse(source_code)
            classes = self._extract_classes(tree)
            functions = self._extract_functions(tree)
            
            self.ast_nodes = self._build_ast(tree, tokens, classes, functions)
            self.relations = self._detect_relations(tree, classes)
            
            print(f"Clases encontradas: {len(classes)}")
            print(f"Funciones encontradas: {len(functions)}")
            print(f"Relaciones detectadas: {len(self.relations)}")
            
            return self.ast_nodes, self.relations
            
        except SyntaxError as e:
            self.syntax_errors.append(f"Error de sintaxis: {e}")
            print(f"ERROR: {e}")
            return [], []
    
    def _extract_classes(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                constructor_analysis = self._analyze_constructor(node)
                composition_attrs = self._find_composition_attributes(node)
                
                classes[node.name] = {
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [self._get_base_name(base) for base in node.bases],
                    'methods': [method.name for method in node.body 
                               if isinstance(method, ast.FunctionDef)],
                    'node': node,
                    'composition_attrs': composition_attrs,
                    'constructor_analysis': constructor_analysis,
                    'accepts_external_component': constructor_analysis.get('accepts_external', False)
                }
        
        return classes
    
    def _analyze_constructor(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        analysis = {
            'accepts_external': False,
            'external_params': [],
            'internal_compositions': [],
            'param_names': []
        }
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                params = [arg.arg for arg in item.args.args if arg.arg != 'self']
                analysis['param_names'] = params
                
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                
                                attr_name = target.attr
                                
                                # Componente externo
                                if (isinstance(stmt.value, ast.Name) and 
                                    stmt.value.id in params):
                                    if self._is_component_attribute(attr_name):
                                        analysis['accepts_external'] = True
                                        analysis['external_params'].append(stmt.value.id)
                                
                                # Composición interna
                                elif isinstance(stmt.value, ast.Call):
                                    if self._is_component_attribute(attr_name):
                                        analysis['internal_compositions'].append(attr_name)
                break
        
        return analysis
    
    def _is_component_attribute(self, attr_name: str) -> bool:
        component_keywords = ['component', 'wrapped', 'decorated']
        return any(keyword in attr_name.lower() for keyword in component_keywords)
    
    def _find_composition_attributes(self, class_node: ast.ClassDef) -> List[str]:
        attrs = []
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for node in ast.walk(item):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                if self._is_component_attribute(target.attr):
                                    attrs.append(target.attr)
        
        return attrs
    
    def _extract_functions(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_class_method(node, tree):
                has_wrapper = self._has_wrapper_pattern(node)
                returns_function = self._returns_function(node)
                
                functions[node.name] = {
                    'name': node.name,
                    'line': node.lineno,
                    'node': node,
                    'has_wrapper': has_wrapper,
                    'returns_function': returns_function
                }
        
        return functions
    
    def _has_wrapper_pattern(self, func_node: ast.FunctionDef) -> bool:
        wrapper_names = ['wrapper', 'decorated', 'inner']
        for item in ast.walk(func_node):
            if isinstance(item, ast.FunctionDef) and item != func_node:
                if any(name in item.name.lower() for name in wrapper_names):
                    return True
        return False
    
    def _returns_function(self, func_node: ast.FunctionDef) -> bool:
        wrapper_names = ['wrapper', 'decorated', 'inner']
        for item in ast.walk(func_node):
            if isinstance(item, ast.Return) and isinstance(item.value, ast.Name):
                if any(name in item.value.id.lower() for name in wrapper_names):
                    return True
        return False
    
    def _is_class_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def _get_base_name(self, base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr
        return str(base)
    
    def _detect_relations(self, tree: ast.AST, classes: Dict[str, Dict[str, Any]]) -> List[ClassRelation]:
        relations = []
        
        # Herencia
        for class_name, class_info in classes.items():
            for base in class_info['bases']:
                if base in classes:
                    relations.append(ClassRelation(
                        source=class_name,
                        target=base,
                        relation_type=RelationType.INHERITANCE,
                        description=f"{class_name} hereda de {base}",
                        line=class_info['line'],
                        confidence=0.95
                    ))
        
        # Composición para decorators
        for class_name, class_info in classes.items():
            if class_info['accepts_external_component']:
                relations.append(ClassRelation(
                    source=class_name,
                    target="Component",
                    relation_type=RelationType.COMPOSITION,
                    description=f"{class_name} decora componente",
                    line=class_info['line'],
                    confidence=0.90
                ))
        
        return relations
    
    def _build_ast(self, tree: ast.AST, tokens: List[Token], 
                  classes: Dict[str, Dict[str, Any]], 
                  functions: Dict[str, Dict[str, Any]]) -> List[ASTNode]:
        nodes = []
        
        # Procesar clases
        for class_name, class_info in classes.items():
            class_node = class_info['node']
            
            ast_node = ASTNode(
                node_type="class",
                name=class_name,
                line=class_info['line'],
                metadata={
                    'bases': class_info['bases'],
                    'methods': class_info['methods'],
                    'composition_attrs': class_info['composition_attrs'],
                    'constructor_analysis': class_info['constructor_analysis'],
                    'accepts_external_component': class_info['accepts_external_component']
                }
            )
            
            # Analizar métodos
            for item in class_node.body:
                if isinstance(item, ast.FunctionDef):
                    method_node = ASTNode(
                        node_type="method",
                        name=item.name,
                        line=item.lineno,
                        metadata={
                            'args': [arg.arg for arg in item.args.args],
                            'has_delegation': self._check_delegation(item, class_info['composition_attrs'])
                        }
                    )
                    ast_node.children.append(method_node)
            
            nodes.append(ast_node)
        
        # Procesar funciones
        for func_name, func_info in functions.items():
            func_node = ASTNode(
                node_type="function",
                name=func_name,
                line=func_info['line'],
                metadata={
                    'has_wrapper': func_info['has_wrapper'],
                    'returns_function': func_info['returns_function']
                }
            )
            nodes.append(func_node)
        
        return nodes
    
    def _check_delegation(self, method_node: ast.FunctionDef, composition_attrs: List[str]) -> bool:
        for node in ast.walk(method_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value, ast.Attribute) and
                    isinstance(node.func.value.value, ast.Name) and
                    node.func.value.value.id == 'self' and
                    node.func.value.attr in composition_attrs):
                    return True
        return False

# =============================================================================
# FASE 3: ANALIZADOR SEMÁNTICO
# =============================================================================

class SemanticAnalyzer:
    def __init__(self):
        self.semantic_errors = []
        self.pattern_info = {}
    
    def analyze(self, ast_nodes: List[ASTNode], tokens: List[Token]) -> Dict[str, PatternInfo]:
        print("Iniciando análisis semántico...")
        
        results = {}
        
        # Analizar clases
        for class_node in [node for node in ast_nodes if node.node_type == "class"]:
            analysis = self._analyze_class(class_node, tokens)
            results[class_node.name] = analysis
        
        # Analizar funciones
        for func_node in [node for node in ast_nodes if node.node_type == "function"]:
            analysis = self._analyze_function(func_node, tokens)
            results[func_node.name] = analysis
        
        patterns_found = len([p for p in results.values() 
                             if p.pattern_type not in ["no_decorator", "decorator_client"]])
        print(f"Patrones encontrados: {patterns_found}")
        
        return results
    
    def _analyze_class(self, class_node: ASTNode, tokens: List[Token]) -> PatternInfo:
        evidences = {}
        violations = []
        
        # Verificar si acepta componente externo
        accepts_external = class_node.metadata.get('accepts_external_component', False)
        constructor_info = class_node.metadata.get('constructor_analysis', {})
        
        # Verificar composición
        composition_attrs = class_node.metadata.get('composition_attrs', [])
        if composition_attrs:
            evidences['has_composition'] = {
                'found': True,
                'attributes': composition_attrs,
                'confidence': 0.90
            }
        
        # Verificar delegación
        delegation_methods = []
        for method in class_node.children:
            if method.node_type == "method" and method.metadata.get('has_delegation', False):
                delegation_methods.append(method.name)
        
        if delegation_methods:
            evidences['has_delegation'] = {
                'found': True,
                'methods': delegation_methods,
                'confidence': 0.85
            }
        
        evidences['accepts_external_component'] = accepts_external
        evidences['constructor_info'] = constructor_info
        
        pattern_type = self._determine_class_pattern(evidences)
        confidence = self._calculate_class_confidence(evidences, pattern_type)
        violations = self._check_class_violations(evidences)
        
        return PatternInfo(
            class_name=class_node.name,
            pattern_type=pattern_type,
            confidence=confidence,
            evidences=evidences,
            violations=violations
        )
    
    def _analyze_function(self, func_node: ASTNode, tokens: List[Token]) -> PatternInfo:
        evidences = {}
        
        has_wrapper = func_node.metadata.get('has_wrapper', False)
        returns_function = func_node.metadata.get('returns_function', False)
        
        if has_wrapper:
            evidences['has_wrapper'] = True
        
        if returns_function:
            evidences['returns_function'] = True
        
        uses_functools = self._check_functools_usage(func_node, tokens)
        if uses_functools:
            evidences['uses_functools'] = True
        
        pattern_type = self._determine_function_pattern(evidences)
        confidence = self._calculate_function_confidence(evidences)
        violations = self._check_function_violations(evidences)
        
        return PatternInfo(
            class_name=func_node.name,
            pattern_type=pattern_type,
            confidence=confidence,
            evidences=evidences,
            violations=violations
        )
    
    def _check_functools_usage(self, func_node: ASTNode, tokens: List[Token]) -> bool:
        func_line = func_node.line
        for token in tokens:
            if (token.type == TokenType.FUNCTOOLS_WRAPS and
                abs(token.line - func_line) <= 10):
                return True
        return False
    
    def _determine_class_pattern(self, evidences: Dict[str, Any]) -> str:
        accepts_external = evidences.get('accepts_external_component', False)
        
        if not accepts_external:
            constructor_info = evidences.get('constructor_info', {})
            if constructor_info.get('internal_compositions'):
                return "decorator_client"
            else:
                return "no_decorator"
        
        has_composition = evidences.get('has_composition', {}).get('found', False)
        has_delegation = evidences.get('has_delegation', {}).get('found', False)
        
        if has_composition and has_delegation:
            return "object_decorator"
        elif has_composition:
            return "incomplete_object_decorator"
        else:
            return "no_decorator"
    
    def _determine_function_pattern(self, evidences: Dict[str, Any]) -> str:
        if (evidences.get('has_wrapper', False) and evidences.get('returns_function', False)):
            if evidences.get('uses_functools', False):
                return "advanced_function_decorator"
            else:
                return "basic_function_decorator"
        elif evidences.get('has_wrapper', False):
            return "incomplete_function_decorator"
        else:
            return "no_decorator"
    
    def _calculate_class_confidence(self, evidences: Dict[str, Any], pattern_type: str) -> float:
        if pattern_type in ["decorator_client", "no_decorator"]:
            return 0.1
        
        weights = {
            'has_composition': 0.40,
            'has_delegation': 0.40,
            'accepts_external_component': 0.20
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for evidence_type, weight in weights.items():
            evidence = evidences.get(evidence_type, {})
            if isinstance(evidence, dict) and evidence.get('found', False):
                confidence = evidence.get('confidence', 0.0)
                total_confidence += confidence * weight
                total_weight += weight
            elif evidence is True:
                total_confidence += 0.95 * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_function_confidence(self, evidences: Dict[str, Any]) -> float:
        weights = {
            'has_wrapper': 0.40,
            'returns_function': 0.40,
            'uses_functools': 0.20
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for evidence_type, weight in weights.items():
            if evidences.get(evidence_type, False):
                total_confidence += 0.85 * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _check_class_violations(self, evidences: Dict[str, Any]) -> List[str]:
        violations = []
        
        if not evidences.get('accepts_external_component', False):
            constructor_info = evidences.get('constructor_info', {})
            if constructor_info.get('internal_compositions'):
                violations.append("Crea componentes internos en lugar de recibirlos")
        
        if (evidences.get('has_composition', {}).get('found', False) and
            not evidences.get('has_delegation', {}).get('found', False)):
            violations.append("Composición sin delegación apropiada")
        
        return violations
    
    def _check_function_violations(self, evidences: Dict[str, Any]) -> List[str]:
        violations = []
        
        if (evidences.get('has_wrapper', False) and
            not evidences.get('returns_function', False)):
            violations.append("Función wrapper sin retorno de función")
        
        return violations

# =============================================================================
# FASE 4: GENERADOR DE CÓDIGO INTERMEDIO
# =============================================================================

class IntermediateCodeGenerator:
    def __init__(self):
        self.ir = None
    
    def generate(self, ast_nodes: List[ASTNode], patterns: Dict[str, PatternInfo], 
                relations: List[ClassRelation]) -> IntermediateRepresentation:
        print("Generando código intermedio...")
        
        classes = {}
        
        for node in ast_nodes:
            classes[node.name] = {
                'name': node.name,
                'line': node.line,
                'type': node.node_type,
                'bases': node.metadata.get('bases', []),
                'methods': self._extract_methods(node),
                'composition_attrs': node.metadata.get('composition_attrs', []),
                'accepts_external': node.metadata.get('accepts_external_component', False),
                'complexity': self._calculate_complexity(node)
            }
        
        filtered_relations = self._filter_relations(relations, patterns)
        
        metadata = {
            'total_elements': len(classes),
            'decorator_patterns': len([p for p in patterns.values() 
                                     if p.pattern_type not in ["no_decorator", "decorator_client"]]),
            'decorator_clients': len([p for p in patterns.values() 
                                    if p.pattern_type == "decorator_client"]),
            'total_relations': len(filtered_relations),
            'timestamp': datetime.now().isoformat()
        }
        
        self.ir = IntermediateRepresentation(
            classes=classes,
            patterns=patterns,
            relations=filtered_relations,
            global_metadata=metadata
        )
        
        print(f"Elementos procesados: {len(classes)}")
        print(f"Relaciones válidas: {len(filtered_relations)}")
        
        return self.ir
    
    def _extract_methods(self, node: ASTNode) -> List[Dict[str, Any]]:
        methods = []
        for child in node.children:
            if child.node_type == "method":
                methods.append({
                    'name': child.name,
                    'line': child.line,
                    'has_delegation': child.metadata.get('has_delegation', False),
                    'visibility': 'private' if child.name.startswith('_') else 'public'
                })
        return methods
    
    def _calculate_complexity(self, node: ASTNode) -> int:
        if node.node_type == "class":
            method_count = len([c for c in node.children if c.node_type == "method"])
            attr_count = len(node.metadata.get('composition_attrs', []))
            return method_count + attr_count
        return 1
    
    def _filter_relations(self, relations: List[ClassRelation], 
                         patterns: Dict[str, PatternInfo]) -> List[ClassRelation]:
        decorator_elements = set()
        
        for name, pattern_info in patterns.items():
            if (pattern_info.pattern_type not in ["no_decorator", "decorator_client"] and 
                pattern_info.confidence >= 0.5):
                decorator_elements.add(name)
        
        filtered = []
        for relation in relations:
            if (relation.target in decorator_elements or 
                relation.source in decorator_elements):
                filtered.append(relation)
        
        return filtered

# =============================================================================
# FASE 5: OPTIMIZADOR
# =============================================================================

class Optimizer:
    def __init__(self):
        self.optimized_data = None
    
    def optimize(self, ir: IntermediateRepresentation) -> OptimizedData:
        print("Optimizando resultados...")
        
        optimized_patterns = {}
        false_positives = []
        confidence_adjustments = {}
        
        for name, pattern_info in ir.patterns.items():
            optimized_pattern = self._optimize_pattern(pattern_info, ir.classes[name], ir.relations)
            
            if optimized_pattern.pattern_type == "decorator_client":
                false_positives.append(name)
                optimized_pattern.confidence = 0.1
            elif self._is_false_positive(optimized_pattern, ir.classes[name]):
                false_positives.append(name)
                optimized_pattern.confidence *= 0.1
                optimized_pattern.pattern_type = "false_positive"
            
            original_confidence = pattern_info.confidence
            adjustment = optimized_pattern.confidence - original_confidence
            if abs(adjustment) > 0.05:
                confidence_adjustments[name] = adjustment
            
            optimized_patterns[name] = optimized_pattern
        
        optimized_relations = self._optimize_relations(ir.relations, optimized_patterns)
        
        self.optimized_data = OptimizedData(
            patterns=optimized_patterns,
            relations=optimized_relations,
            false_positives_removed=false_positives,
            confidence_adjustments=confidence_adjustments
        )
        
        print(f"Falsos positivos removidos: {len(false_positives)}")
        return self.optimized_data
    
    def _optimize_pattern(self, pattern_info: PatternInfo, element_info: Dict[str, Any], 
                         relations: List[ClassRelation]) -> PatternInfo:
        optimized = PatternInfo(
            class_name=pattern_info.class_name,
            pattern_type=pattern_info.pattern_type,
            confidence=pattern_info.confidence,
            evidences=pattern_info.evidences.copy(),
            violations=pattern_info.violations.copy()
        )
        
        # Bonificar patrones completos
        if pattern_info.pattern_type in ["object_decorator", "advanced_function_decorator"]:
            optimized.confidence = min(optimized.confidence * 1.10, 1.0)
        
        # Penalizar clientes
        elif pattern_info.pattern_type == "decorator_client":
            optimized.confidence = 0.1
        
        # Bonificar si hay relaciones válidas
        related_count = len([r for r in relations 
                           if r.source == pattern_info.class_name or r.target == pattern_info.class_name])
        if related_count >= 1 and pattern_info.pattern_type not in ["decorator_client", "no_decorator"]:
            optimized.confidence = min(optimized.confidence * 1.05, 1.0)
        
        return optimized
    
    def _optimize_relations(self, relations: List[ClassRelation], 
                          patterns: Dict[str, PatternInfo]) -> List[ClassRelation]:
        optimized_relations = []
        
        for relation in relations:
            source_pattern = patterns.get(relation.source)
            target_pattern = patterns.get(relation.target)
            
            valid_source = (source_pattern and 
                           source_pattern.pattern_type not in ["no_decorator", "false_positive", "decorator_client"] and
                           source_pattern.confidence >= 0.5)
            
            valid_target = (target_pattern and 
                           target_pattern.pattern_type not in ["no_decorator", "false_positive", "decorator_client"] and
                           target_pattern.confidence >= 0.5)
            
            if valid_source or valid_target:
                optimized_relations.append(relation)
        
        return optimized_relations
    
    def _is_false_positive(self, pattern_info: PatternInfo, element_info: Dict[str, Any]) -> bool:
        # Umbral de confianza mínimo
        if pattern_info.confidence < 0.4:
            return True
        
        # Demasiadas violaciones
        if len(pattern_info.violations) >= 2:
            return True
        
        return False

# =============================================================================
# FASE 6: GENERADOR UML
# =============================================================================

class UMLGenerator:
    def __init__(self):
        self.templates = {
            'object_decorator': self._object_decorator_template,
            'incomplete_object_decorator': self._incomplete_object_decorator_template,
            'basic_function_decorator': self._basic_function_decorator_template,
            'advanced_function_decorator': self._advanced_function_decorator_template,
            'incomplete_function_decorator': self._incomplete_function_decorator_template,
            'decorator_client': self._decorator_client_template,
            'no_decorator': self._no_pattern_template,
            'false_positive': self._false_positive_template
        }
        
        self.relation_styles = {
            RelationType.INHERITANCE: "--|>",
            RelationType.COMPOSITION: "*--",
            RelationType.ASSOCIATION: "--",
            RelationType.DEPENDENCY: "..>",
            RelationType.DECORATOR: "..>"
        }
    
    def generate(self, optimized_data: OptimizedData, ir_data: IntermediateRepresentation = None) -> str:
        print("Generando código UML...")
        
        self._ir_data = ir_data
        
        uml_content = self._create_header()
        
        # Filtrar patrones válidos
        decorator_patterns = [p for p in optimized_data.patterns.values() 
                            if p.pattern_type not in ["no_decorator", "false_positive", "decorator_client"]]
        
        decorator_clients = [p for p in optimized_data.patterns.values() 
                           if p.pattern_type == "decorator_client"]
        
        if not decorator_patterns:
            uml_content += self._no_patterns_section()
        else:
            # Generar clases principales
            for pattern in decorator_patterns:
                template_func = self.templates.get(pattern.pattern_type, self._unknown_template)
                uml_content += template_func(pattern)
            
            # Generar relaciones
            uml_content += self._generate_relations(optimized_data.relations)
        
        # Sección de clientes
        if decorator_clients:
            uml_content += self._clients_section(decorator_clients)
        
        # Sección de falsos positivos
        real_false_positives = [fp for fp in optimized_data.false_positives_removed
                              if not any(p.class_name == fp and p.pattern_type == "decorator_client" 
                                       for p in optimized_data.patterns.values())]
        if real_false_positives:
            uml_content += self._false_positives_section(real_false_positives)
        
        uml_content += "\n@enduml"
        
        print(f"Patrones generados: {len(decorator_patterns)}")
        print(f"Clientes identificados: {len(decorator_clients)}")
        print(f"Relaciones incluidas: {len(optimized_data.relations)}")
        
        return uml_content
    
    def _create_header(self) -> str:
        return f"""@startuml
!theme cerulean-outline

title Análisis de Patrones Decorator
' Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    def _object_decorator_template(self, pattern: PatternInfo) -> str:
        element_info = self._get_element_info(pattern.class_name)
        attributes = self._format_composition_attributes(element_info)
        methods = self._format_methods(element_info)
        
        return f"""
class {pattern.class_name} <<Decorator>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Object Decorator Pattern
  Confianza: {int(pattern.confidence * 100)}%
  Recibe componente externo
end note

"""
    
    def _incomplete_object_decorator_template(self, pattern: PatternInfo) -> str:
        element_info = self._get_element_info(pattern.class_name)
        attributes = self._format_composition_attributes(element_info)
        methods = self._format_methods(element_info)
        
        return f"""
class {pattern.class_name} <<IncompleteDecorator>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Decorator Incompleto
  Confianza: {int(pattern.confidence * 100)}%
  Requiere implementar delegación
end note

"""
    
    def _basic_function_decorator_template(self, pattern: PatternInfo) -> str:
        return f"""
class {pattern.class_name} <<FunctionDecorator>> {{
  + wrapper(*args, **kwargs)
  + __call__(func)
}}

note top of {pattern.class_name}
  Function Decorator Básico
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _advanced_function_decorator_template(self, pattern: PatternInfo) -> str:
        return f"""
class {pattern.class_name} <<AdvancedFunctionDecorator>> {{
  + wrapper(*args, **kwargs)
  + __call__(func)
  + preserve_metadata()
}}

note top of {pattern.class_name}
  Function Decorator Avanzado
  Confianza: {int(pattern.confidence * 100)}%
  Utiliza functools.wraps
end note

"""
    
    def _incomplete_function_decorator_template(self, pattern: PatternInfo) -> str:
        return f"""
class {pattern.class_name} <<IncompleteFunctionDecorator>> {{
  + wrapper(*args, **kwargs)
}}

note top of {pattern.class_name}
  Function Decorator Incompleto
  Confianza: {int(pattern.confidence * 100)}%
  Falta retorno de función
end note

"""
    
    def _decorator_client_template(self, pattern: PatternInfo) -> str:
        return f"""
class {pattern.class_name} <<Client>> {{
  - decorated_components: Component[]
  --
  + use_decorators()
}}

note top of {pattern.class_name}
  Cliente de Decorators
  Utiliza decorators pero no implementa el patrón
end note

"""
    
    def _no_pattern_template(self, pattern: PatternInfo) -> str:
        return ""
    
    def _false_positive_template(self, pattern: PatternInfo) -> str:
        return f"""
class {pattern.class_name} <<FalsoPositivo>> {{
}}

"""
    
    def _unknown_template(self, pattern: PatternInfo) -> str:
        return f"""
class {pattern.class_name} {{
}}

"""
    
    def _get_element_info(self, name: str) -> Dict[str, Any]:
        if hasattr(self, '_ir_data') and self._ir_data and name in self._ir_data.classes:
            return self._ir_data.classes[name]
        return {'methods': [], 'composition_attrs': []}
    
    def _format_composition_attributes(self, element_info: Dict[str, Any]) -> str:
        attrs = element_info.get('composition_attrs', [])
        if not attrs:
            return "  ' Sin atributos de composición"
        
        formatted_attrs = []
        for attr in attrs[:3]:
            formatted_attrs.append(f"  - {attr}: Component")
        
        return "\n".join(formatted_attrs)
    
    def _format_methods(self, element_info: Dict[str, Any]) -> str:
        methods = element_info.get('methods', [])
        if not methods:
            if element_info.get('type') == 'function':
                return "  + __call__(func)"
            else:
                return "  + operation()"
        
        formatted_methods = []
        excluded = ['__str__', '__repr__', '__del__', '__hash__', '__eq__']
        
        for method in methods[:6]:
            method_name = method.get('name', 'unknown')
            
            if method_name in excluded:
                continue
            
            if method_name.startswith('__') and method_name.endswith('__'):
                visibility = "+"
            elif method_name.startswith('_'):
                visibility = "-"
            else:
                visibility = "+"
            
            formatted_methods.append(f"  {visibility} {method_name}()")
        
        if not formatted_methods:
            return "  + operation()"
        
        return "\n".join(formatted_methods)
    
    def _generate_relations(self, relations: List[ClassRelation]) -> str:
        if not relations:
            return ""
        
        relations_uml = "\n' Relaciones\n"
        
        for relation in relations:
            style = self._get_relation_style(relation.relation_type)
            relations_uml += f"{relation.source} {style} {relation.target}\n"
        
        return relations_uml + "\n"
    
    def _get_relation_style(self, relation_type: RelationType) -> str:
        return self.relation_styles.get(relation_type, "-->")
    
    def _no_patterns_section(self) -> str:
        return """
note as NoPatterns
  No se encontraron implementaciones
  del patrón Decorator en el código.
end note

"""
    
    def _clients_section(self, decorator_clients: List[PatternInfo]) -> str:
        if not decorator_clients:
            return ""
        
        clients_list = ', '.join([client.class_name for client in decorator_clients])
        return f"""
note as DecoratorClients
  Clientes de Decorators: {len(decorator_clients)}
  Clases: {clients_list}
  (Utilizan decorators sin implementar el patrón)
end note

"""
    
    def _false_positives_section(self, false_positives: List[str]) -> str:
        return f"""
note as FalsePositives
  Falsos Positivos: {len(false_positives)}
  Elementos: {', '.join(false_positives)}
end note

"""

# =============================================================================
# COMPILADOR PRINCIPAL
# =============================================================================

class DecoratorCompiler:
    def __init__(self):
        self.lexer = LexicalAnalyzer()
        self.parser = SyntaxAnalyzer()
        self.semantic = SemanticAnalyzer()
        self.ir_generator = IntermediateCodeGenerator()
        self.optimizer = Optimizer()
        self.uml_generator = UMLGenerator()
    
    def compile(self, input_file: str, output_file: str) -> bool:
        print("COMPILADOR DE PATRONES DECORATOR")
        print("=" * 50)
        print(f"Archivo entrada: {input_file}")
        print(f"Archivo salida: {output_file}")
        print("=" * 50)
        
        try:
            # Leer código fuente
            with open(input_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Ejecutar pipeline de 6 fases
            tokens = self.lexer.analyze(source_code)
            
            ast_nodes, relations = self.parser.parse(source_code, tokens)
            if not ast_nodes and self.parser.syntax_errors:
                print("ERROR: Problemas de sintaxis detectados.")
                return False
            
            patterns = self.semantic.analyze(ast_nodes, tokens)
            
            ir = self.ir_generator.generate(ast_nodes, patterns, relations)
            
            optimized_data = self.optimizer.optimize(ir)
            
            uml_code = self.uml_generator.generate(optimized_data, ir)
            
            # Escribir resultado
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(uml_code)
            
            print("\n" + "=" * 50)
            print("COMPILACIÓN COMPLETADA")
            print(f"Diagrama UML: {output_file}")
            
            # Intentar generar imagen
            if self._generate_image(output_file):
                print(f"Imagen: {output_file.replace('.puml', '.png')}")
            else:
                print("Para imagen: plantuml " + output_file)
            
            print("=" * 50)
            return True
            
        except FileNotFoundError:
            print(f"ERROR: Archivo {input_file} no encontrado")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def _generate_image(self, puml_file: str) -> bool:
        try:
            commands = ['plantuml', 'java -jar plantuml.jar']
            
            for cmd in commands:
                try:
                    result = subprocess.run(
                        f"{cmd} {puml_file}",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        return True
                        
                except (subprocess.TimeoutExpired, Exception):
                    continue
            
            return False
            
        except Exception:
            return False

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

#def main():
#    if len(sys.argv) != 3:
#        print("Uso: python3 decorator_compiler.py codigo.py diagrama.puml")
#        print()
#        print("Patrones detectados:")
#        print("  - object_decorator: Decorador de objeto completo")
#        print("  - incomplete_object_decorator: Decorador de objeto incompleto")
#        print("  - basic_function_decorator: Decorador de función básico")
#        print("  - advanced_function_decorator: Decorador con functools")
#        print("  - incomplete_function_decorator: Decorador de función incompleto")
#        print("  - decorator_client: Cliente que usa decorators")
#        sys.exit(1)
#    
#    input_file = sys.argv[1]
#    output_file = sys.argv[2]
#    
#    if not os.path.exists(input_file):
#        print(f"ERROR: El archivo {input_file} no existe")
#        sys.exit(1)
#    
#    if not output_file.endswith('.puml'):
#        print("RECOMENDACIÓN: Use extensión .puml para el archivo de salida")
#    
#    compiler = DecoratorCompiler()
#    success = compiler.compile(input_file, output_file)
#    
#    sys.exit(0 if success else 1)
#
#if __name__ == "__main__":
#    main()