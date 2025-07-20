#!/usr/bin/env python3
"""
Compilador de Patrones Singleton con Relaciones - 6 Fases

Uso: python3 singleton_compiler.py input.py output.puml
"""

import ast
import re
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from data_structure import ASTNode, SemanticInfo, ClassRelation, RelationType
from data_structure import OptimizedData, IntermediateRepresentation

# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

class TokenType(Enum):
    INSTANCE_VAR = "instance_variable"
    ACCESS_METHOD = "access_method"
    NEW_OVERRIDE = "new_override"
    CONTROL_FLOW = "control_flow"
    STATIC_DECORATOR = "static_decorator"
    CLASS_DEF = "class_definition"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    confidence: float = 1.0


# =============================================================================
# FASE 1: ANÁLISIS LÉXICO
# =============================================================================

class LexicalAnalyzer:
    """Analizador léxico para detectar tokens relacionados con patrones Singleton"""
    
    def __init__(self):
        self.tokens = []
        self.symbol_table = {}
        
        self.token_patterns = {
            TokenType.INSTANCE_VAR: [
                r'_instance\b', r'__instance\b', r'_obj\b', 
                r'_singleton\b', r'__singleton\b'
            ],
            TokenType.ACCESS_METHOD: [
                r'get_instance\b', r'getInstance\b', r'instance\b',
                r'get_singleton\b', r'create\b'
            ],
            TokenType.NEW_OVERRIDE: [r'__new__\b'],
            TokenType.CONTROL_FLOW: [
                r'hasattr\b', r'is None\b', r'== None\b', r'if\b'
            ],
            TokenType.STATIC_DECORATOR: [r'@staticmethod\b', r'@classmethod\b'],
            TokenType.CLASS_DEF: [r'class\s+\w+']
        }
    
    def tokenize(self, source_code: str) -> List[Token]:
        """Ejecuta el análisis léxico del código fuente"""
        print("FASE 1: ANALISIS LEXICO")
        print("-" * 40)
        
        lines = source_code.split('\n')
        tokens = []
        
        for line_num, line in enumerate(lines, 1):
            for token_type, patterns in self.token_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        token = Token(
                            type=token_type,
                            value=match.group(),
                            line=line_num,
                            column=match.start(),
                            confidence=self._calculate_token_confidence(token_type, match.group())
                        )
                        tokens.append(token)
                        
                        if token.value not in self.symbol_table:
                            self.symbol_table[token.value] = {
                                'type': token_type,
                                'occurrences': 0,
                                'lines': []
                            }
                        self.symbol_table[token.value]['occurrences'] += 1
                        self.symbol_table[token.value]['lines'].append(line_num)
        
        self.tokens = tokens
        self._print_lexical_results()
        return tokens
    
    def _calculate_token_confidence(self, token_type: TokenType, value: str) -> float:
        """Calcula el nivel de confianza de un token basado en su tipo y valor"""
        confidence_map = {
            TokenType.INSTANCE_VAR: {
                '_instance': 0.95, '__instance': 0.90, '_obj': 0.70,
                '_singleton': 0.85, '__singleton': 0.80
            },
            TokenType.ACCESS_METHOD: {
                'get_instance': 0.95, 'getInstance': 0.90, 'instance': 0.75,
                'get_singleton': 0.85, 'create': 0.60
            },
            TokenType.NEW_OVERRIDE: {'__new__': 0.95},
            TokenType.CONTROL_FLOW: {
                'hasattr': 0.85, 'is None': 0.80, '== None': 0.75, 'if': 0.50
            },
            TokenType.STATIC_DECORATOR: {
                '@staticmethod': 0.90, '@classmethod': 0.85
            }
        }
        return confidence_map.get(token_type, {}).get(value, 0.50)
    
    def _print_lexical_results(self):
        print(f"Tokens encontrados: {len(self.tokens)}")
        for token_type in TokenType:
            count = len([t for t in self.tokens if t.type == token_type])
            if count > 0:
                print(f"   {token_type.value}: {count}")

# =============================================================================
# FASE 2: ANÁLISIS SINTÁCTICO CON DETECCIÓN DE RELACIONES
# =============================================================================

class SyntaxAnalyzer:
    """Analizador sintáctico que construye el AST y detecta relaciones entre clases"""
    
    def __init__(self):
        self.ast_nodes = []
        self.syntax_errors = []
        self.relations = []
        
    def parse(self, source_code: str, tokens: List[Token]) -> Tuple[List[ASTNode], List[ClassRelation]]:
        """Ejecuta el análisis sintáctico y detección de relaciones"""
        print("\nFASE 2: ANALISIS SINTACTICO CON RELACIONES")
        print("-" * 40)
        
        try:
            tree = ast.parse(source_code)
            all_classes = self._extract_all_classes(tree)
            ast_nodes = self._build_custom_ast(tree, tokens, all_classes)
            relations = self._detect_relations(tree, all_classes)
            
            self.ast_nodes = ast_nodes
            self.relations = relations
            
            self._print_syntax_results()
            return ast_nodes, relations
            
        except SyntaxError as e:
            self.syntax_errors.append(f"Error de sintaxis: {e}")
            print(f"ERROR: {e}")
            return [], []
    
    def _extract_all_classes(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        """Extrae información básica de todas las clases del código"""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [self._extract_base_name(base) for base in node.bases],
                    'methods': [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
                    'node': node
                }
        
        return classes
    
    def _extract_base_name(self, base: ast.expr) -> str:
        """Extrae el nombre de la clase base de herencia"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr
        else:
            return str(base)
    
    def _detect_relations(self, tree: ast.AST, all_classes: Dict[str, Dict[str, Any]]) -> List[ClassRelation]:
        """Detecta diferentes tipos de relaciones entre clases"""
        relations = []
        
        for class_name, class_info in all_classes.items():
            class_node = class_info['node']
            
            # Detectar herencia
            for base in class_info['bases']:
                if base in all_classes:
                    relations.append(ClassRelation(
                        source=class_name,
                        target=base,
                        relation_type=RelationType.INHERITANCE,
                        description=f"{class_name} hereda de {base}",
                        line=class_info['line'],
                        confidence=0.95
                    ))
            
            # Detectar composición, asociación y dependencia
            relations.extend(self._detect_usage_relations(class_node, class_name, all_classes))
            
            # Detectar decoradores
            relations.extend(self._detect_decorator_relations(class_node, class_name, all_classes))
        
        return relations
    
    def _detect_usage_relations(self, class_node: ast.ClassDef, class_name: str, all_classes: Dict[str, Dict[str, Any]]) -> List[ClassRelation]:
        """Detecta relaciones de uso entre clases con clasificación CORREGIDA"""
        relations = []
        used_classes = {}  # Para evitar relaciones duplicadas
        
        # PASO 1: Analizar constructor (__init__) para detectar COMPOSICIÓN
        init_method = None
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break
        
        if init_method:
            for node in ast.walk(init_method):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        # Buscar asignaciones a self.atributo
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            
                            # CASO 1: self.atributo = OtraClase() -> COMPOSICIÓN DIRECTA
                            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                                if node.value.func.id in all_classes:
                                    used_classes[node.value.func.id] = RelationType.COMPOSITION
                                    relations.append(ClassRelation(
                                        source=class_name,
                                        target=node.value.func.id,
                                        relation_type=RelationType.COMPOSITION,
                                        description=f"{class_name} compone {node.value.func.id}",
                                        line=node.lineno,
                                        confidence=0.95
                                    ))
                            
                            # CASO 2: self.atributo = OtraClase.get_instance() -> ASOCIACIÓN (pero se guarda)
                            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                                if isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in all_classes:
                                    if node.value.func.attr in ['get_instance', 'getInstance', 'instance']:
                                        # Aunque es singleton, al guardarse en self es COMPOSICIÓN
                                        used_classes[node.value.func.value.id] = RelationType.COMPOSITION
                                        relations.append(ClassRelation(
                                            source=class_name,
                                            target=node.value.func.value.id,
                                            relation_type=RelationType.COMPOSITION,
                                            description=f"{class_name} compone singleton {node.value.func.value.id}",
                                            line=node.lineno,
                                            confidence=0.95
                                        ))
        
        # PASO 2: Analizar métodos (NO __init__) para detectar DEPENDENCIA y ASOCIACIÓN
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                
                # Analizar el cuerpo del método
                for node in ast.walk(item):
                    
                    # CASO 3: Detectar var = OtraClase.get_instance() -> DEPENDENCIA (uso temporal)
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            # Si es una variable local (no self.algo)
                            if isinstance(target, ast.Name):
                                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                                    if isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in all_classes:
                                        if node.value.func.attr in ['get_instance', 'getInstance', 'instance']:
                                            target_class = node.value.func.value.id
                                            # Solo si no ya está usado como composición
                                            if target_class not in used_classes:
                                                used_classes[target_class] = RelationType.DEPENDENCY
                                                relations.append(ClassRelation(
                                                    source=class_name,
                                                    target=target_class,
                                                    relation_type=RelationType.DEPENDENCY,
                                                    description=f"{class_name} depende de singleton {target_class}",
                                                    line=node.lineno,
                                                    confidence=0.90
                                                ))
                    
                    # CASO 4: Detectar llamadas directas OtraClase.metodo() -> DEPENDENCIA
                    elif isinstance(node, ast.Attribute):
                        if isinstance(node.value, ast.Name) and node.value.id in all_classes:
                            target_class = node.value.id
                            
                            # Solo si no ya está usado
                            if target_class not in used_classes:
                                # Si es get_instance, es uso de singleton -> DEPENDENCIA
                                if node.attr in ['get_instance', 'getInstance', 'instance']:
                                    used_classes[target_class] = RelationType.DEPENDENCY
                                    relations.append(ClassRelation(
                                        source=class_name,
                                        target=target_class,
                                        relation_type=RelationType.DEPENDENCY,
                                        description=f"{class_name} usa singleton {target_class}",
                                        line=node.lineno,
                                        confidence=0.85
                                    ))
                                # Si es otro método, también es dependencia
                                else:
                                    used_classes[target_class] = RelationType.DEPENDENCY
                                    relations.append(ClassRelation(
                                        source=class_name,
                                        target=target_class,
                                        relation_type=RelationType.DEPENDENCY,
                                        description=f"{class_name} depende de {target_class}",
                                        line=node.lineno,
                                        confidence=0.80
                                    ))
                    
                    # CASO 5: Detectar instanciación temporal OtraClase() -> DEPENDENCIA
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id in all_classes:
                            target_class = node.func.id
                            if target_class not in used_classes:
                                used_classes[target_class] = RelationType.DEPENDENCY
                                relations.append(ClassRelation(
                                    source=class_name,
                                    target=target_class,
                                    relation_type=RelationType.DEPENDENCY,
                                    description=f"{class_name} instancia temporalmente {target_class}",
                                    line=node.lineno,
                                    confidence=0.85
                                ))
        return relations
     
    def _detect_decorator_relations(self, class_node: ast.ClassDef, class_name: str, all_classes: Dict[str, Dict[str, Any]]) -> List[ClassRelation]:
        """Detecta relaciones de decoración entre clases"""
        relations = []
        
        # Decoradores de clase
        for decorator in class_node.decorator_list:
            decorator_name = None
            if isinstance(decorator, ast.Name):
                decorator_name = decorator.id
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorator_name = decorator.func.id
            
            if decorator_name and decorator_name in all_classes:
                relations.append(ClassRelation(
                    source=decorator_name,
                    target=class_name,
                    relation_type=RelationType.DECORATOR,
                    description=f"{decorator_name} decora {class_name}",
                    line=class_node.lineno,
                    confidence=0.95
                ))
        
        return relations

    def _extract_decorators(self, decorator_list):
        """Extrae información de decoradores del AST"""
        decorators = []
        for decorator in decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(f"@{decorator.id}")
            elif isinstance(decorator, ast.Attribute):
                decorators.append(f"@{decorator.attr}")
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(f"@{decorator.func.id}")
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(f"@{decorator.func.attr}")
        return decorators

    def _build_custom_ast(self, tree: ast.AST, tokens: List[Token], all_classes: Dict[str, Dict[str, Any]]) -> List[ASTNode]:
        """Construye un AST personalizado para el análisis de patrones"""
        nodes = []
        
        for class_name, class_info in all_classes.items():
            class_node = class_info['node']
            
            ast_class_node = ASTNode(
                node_type="class",
                name=class_name,
                line=class_info['line'],
                metadata={
                    'bases': class_info['bases'],
                    'methods': class_info['methods']
                }
            )
            
            instance_attributes = self._extract_instance_attributes(class_node)
            
            # Analizar métodos y variables de clase
            for item in class_node.body:
                if isinstance(item, ast.FunctionDef):
                    method_node = ASTNode(
                        node_type="method",
                        name=item.name,
                        line=item.lineno,
                        metadata={
                            'decorators': self._extract_decorators(item.decorator_list),
                            'args': [arg.arg for arg in item.args.args],
                            'body_complexity': len(item.body),
                            'is_static': any('@staticmethod' in str(d) for d in self._extract_decorators(item.decorator_list)),
                            'is_classmethod': any('@classmethod' in str(d) for d in self._extract_decorators(item.decorator_list))
                        }
                    )
                    
                    method_node.children = self._analyze_method_body(item)
                    ast_class_node.children.append(method_node)
                
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            var_node = ASTNode(
                                node_type="class_variable",
                                name=target.id,
                                line=item.lineno,
                                metadata={
                                    'value': self._extract_value(item.value),
                                    'is_class_var': True
                                }
                            )
                            ast_class_node.children.append(var_node)
            
            # Añadir atributos de instancia
            for attr_name, attr_info in instance_attributes.items():
                attr_node = ASTNode(
                    node_type="instance_variable",
                    name=attr_name,
                    line=attr_info['line'],
                    metadata={
                        'value': attr_info['value'],
                        'is_class_var': False,
                        'inferred_type': self._infer_attribute_type(attr_info['value'])
                    }
                )
                ast_class_node.children.append(attr_node)
            
            nodes.append(ast_class_node)
        
        return nodes
    
    def _extract_instance_attributes(self, class_node: ast.ClassDef) -> Dict[str, Dict[str, Any]]:
        """Extrae atributos de instancia desde métodos de la clase"""
        attributes = {}
        
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                for node in ast.walk(method):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                
                                attr_name = target.attr
                                attr_value = self._extract_value(node.value)
                                
                                if attr_name not in attributes or method.name == '__init__':
                                    attributes[attr_name] = {
                                        'value': attr_value,
                                        'line': node.lineno,
                                        'method': method.name
                                    }
        
        return attributes
    
    def _infer_attribute_type(self, value) -> str:
        """Infiere el tipo de datos de un atributo"""
        if value is None:
            return "Object"
        elif isinstance(value, str):
            if value.isdigit():
                return "int"
            elif value.lower() in ['true', 'false']:
                return "bool"
            elif '"' in value or "'" in value:
                return "str"
            else:
                return "str"
        elif isinstance(value, (int, float, bool)):
            return type(value).__name__
        else:
            return "Object"
    
    def _analyze_method_body(self, method_node: ast.FunctionDef) -> List[ASTNode]:
        """Analiza el contenido de un método"""
        body_nodes = []
        
        for stmt in ast.walk(method_node):
            if isinstance(stmt, ast.If):
                condition_node = ASTNode(
                    node_type="conditional",
                    name="if_statement",
                    line=stmt.lineno,
                    metadata={'condition_type': self._analyze_condition(stmt.test)}
                )
                body_nodes.append(condition_node)
            
            elif isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Name):
                    call_node = ASTNode(
                        node_type="function_call",
                        name=stmt.func.id,
                        line=stmt.lineno,
                        metadata={'args_count': len(stmt.args)}
                    )
                    body_nodes.append(call_node)
        
        return body_nodes
    
    def _analyze_condition(self, test_node: ast.expr) -> str:
        """Analiza el tipo de condición en estructuras de control"""
        if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            return "not_condition"
        elif isinstance(test_node, ast.Compare):
            return "comparison"
        elif isinstance(test_node, ast.Call):
            if isinstance(test_node.func, ast.Name) and test_node.func.id == 'hasattr':
                return "hasattr_check"
        return "unknown"
    
    def _extract_value(self, node: ast.expr) -> Any:
        """Extrae el valor de un nodo AST"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        return "complex_expression"
    
    def _print_syntax_results(self):
        classes = [node for node in self.ast_nodes if node.node_type == "class"]
        print(f"Clases encontradas: {len(classes)}")
        for class_node in classes:
            methods = [child for child in class_node.children if child.node_type == "method"]
            variables = [child for child in class_node.children if child.node_type in ["class_variable", "instance_variable"]]
            print(f"   {class_node.name}: {len(methods)} métodos, {len(variables)} atributos")
        
        print(f"Relaciones encontradas: {len(self.relations)}")

# =============================================================================
# FASE 3: ANÁLISIS SEMÁNTICO
# =============================================================================

class SemanticAnalyzer:
    """Analizador semántico que identifica patrones Singleton"""
    
    def __init__(self):
        self.semantic_errors = []
        self.pattern_info = {}
    
    def analyze(self, ast_nodes: List[ASTNode], tokens: List[Token]) -> Dict[str, SemanticInfo]:
        """Ejecuta el análisis semántico para identificar patrones"""
        print("\nFASE 3: ANALISIS SEMANTICO")
        print("-" * 40)
        
        semantic_results = {}
        
        for class_node in [node for node in ast_nodes if node.node_type == "class"]:
            analysis = self._analyze_class_semantics(class_node, tokens)
            semantic_results[class_node.name] = analysis
        
        self._print_semantic_results(semantic_results)
        return semantic_results
    
    def _analyze_class_semantics(self, class_node: ASTNode, tokens: List[Token]) -> SemanticInfo:
        """Analiza la semántica específica de una clase"""
        evidences = {}
        violations = []
        
        instance_vars = self._find_instance_variables(class_node)
        if instance_vars:
            evidences['instance_variable'] = {
                'found': True,
                'names': instance_vars,
                'confidence': self._calculate_var_confidence(instance_vars)
            }
        
        access_methods = self._find_access_methods(class_node)
        if access_methods:
            evidences['access_method'] = {
                'found': True,
                'methods': access_methods,
                'confidence': self._calculate_method_confidence(access_methods)
            }
        
        new_override = self._find_new_override(class_node)
        if new_override:
            evidences['new_override'] = {
                'found': True,
                'method': new_override,
                'confidence': 0.95
            }
        
        pattern_type = self._determine_pattern_type(evidences)
        overall_confidence = self._calculate_overall_confidence(evidences)
        violations = self._check_semantic_violations(class_node, evidences)
        
        return SemanticInfo(
            class_name=class_node.name,
            pattern_type=pattern_type,
            confidence=overall_confidence,
            evidences=evidences,
            violations=violations
        )
    
    def _find_instance_variables(self, class_node: ASTNode) -> List[str]:
        """Busca variables que puedan ser instancias singleton"""
        instance_vars = []
        for child in class_node.children:
            if child.node_type in ["class_variable", "instance_variable"]:
                if any(keyword in child.name.lower() for keyword in 
                      ['instance', 'singleton', 'obj', 'current']):
                    instance_vars.append(child.name)
        return instance_vars
    
    def _find_access_methods(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Busca métodos de acceso al singleton"""
        access_methods = []
        for child in class_node.children:
            if child.node_type == "method":
                if any(keyword in child.name.lower() for keyword in 
                      ['get_instance', 'getinstance', 'instance', 'create']):
                    access_methods.append({
                        'name': child.name,
                        'decorators': child.metadata.get('decorators', []),
                        'line': child.line
                    })
        return access_methods
    
    def _find_new_override(self, class_node: ASTNode) -> Optional[Dict[str, Any]]:
        """Busca override del método __new__"""
        for child in class_node.children:
            if child.node_type == "method" and child.name == "__new__":
                return {
                    'name': child.name,
                    'args': child.metadata.get('args', []),
                    'line': child.line
                }
        return None
    
    def _determine_pattern_type(self, evidences: Dict[str, Any]) -> str:
        """Determina el tipo específico de patrón Singleton"""
        if evidences.get('new_override', {}).get('found'):
            return "new_override_singleton"
        elif (evidences.get('instance_variable', {}).get('found') and 
              evidences.get('access_method', {}).get('found')):
            methods = evidences['access_method']['methods']
            if any('@staticmethod' in str(method.get('decorators', [])) for method in methods):
                return "classic_static_singleton"
            elif any('@classmethod' in str(method.get('decorators', [])) for method in methods):
                return "classic_class_singleton"
            else:
                return "basic_singleton"
        elif evidences.get('instance_variable', {}).get('found'):
            return "incomplete_singleton"
        else:
            return "no_singleton"
    
    def _calculate_var_confidence(self, variables: List[str]) -> float:
        """Calcula confianza basada en nombres de variables"""
        if not variables:
            return 0.0
        
        confidence_map = {
            '_instance': 0.95, '__instance': 0.90, '_singleton': 0.85,
            '__singleton': 0.80, '_obj': 0.70, '_current': 0.65
        }
        
        return max(confidence_map.get(var, 0.50) for var in variables)
    
    def _calculate_method_confidence(self, methods: List[Dict[str, Any]]) -> float:
        """Calcula confianza basada en métodos de acceso"""
        if not methods:
            return 0.0
        
        base_confidence = 0.60
        
        for method in methods:
            name = method['name']
            decorators = method.get('decorators', [])
            
            if name in ['get_instance', 'getInstance']:
                base_confidence = max(base_confidence, 0.90)
            elif 'instance' in name:
                base_confidence = max(base_confidence, 0.75)
            
            if '@staticmethod' in str(decorators):
                base_confidence = min(base_confidence + 0.10, 1.0)
            elif '@classmethod' in str(decorators):
                base_confidence = min(base_confidence + 0.05, 1.0)
        
        return base_confidence
    
    def _calculate_overall_confidence(self, evidences: Dict[str, Any]) -> float:
        """Calcula la confianza general del patrón detectado"""
        weights = {
            'instance_variable': 0.30,
            'access_method': 0.30,
            'new_override': 0.40
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for evidence_type, weight in weights.items():
            evidence = evidences.get(evidence_type, {})
            if evidence.get('found', False):
                confidence = evidence.get('confidence', 0.0)
                total_confidence += confidence * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _check_semantic_violations(self, class_node: ASTNode, evidences: Dict[str, Any]) -> List[str]:
        """Verifica violaciones en la implementación del patrón"""
        violations = []
        
        instance_vars = evidences.get('instance_variable', {}).get('names', [])
        if len(instance_vars) > 1:
            violations.append(f"Múltiples variables de instancia: {instance_vars}")
        
        access_methods = evidences.get('access_method', {}).get('methods', [])
        if len(access_methods) > 1:
            method_names = [m['name'] for m in access_methods]
            violations.append(f"Múltiples métodos de acceso: {method_names}")
        
        return violations
    
    def _print_semantic_results(self, results: Dict[str, SemanticInfo]):
        print(f"Clases analizadas: {len(results)}")
        for class_name, info in results.items():
            if info.pattern_type != "no_singleton":
                print(f"   {class_name}: {info.pattern_type} (confianza: {info.confidence:.2f})")

# =============================================================================
# FASE 4: GENERACIÓN DE REPRESENTACIÓN INTERMEDIA
# =============================================================================

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

# =============================================================================
# FASE 5: OPTIMIZACIÓN
# =============================================================================

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

# =============================================================================
# FASE 6: GENERACIÓN DE CÓDIGO UML
# =============================================================================

class UMLCodeGenerator:
    """Generador de código UML a partir del análisis de patrones"""
    
    def __init__(self):
        self.templates = {
            'new_override_singleton': self._template_new_override,
            'classic_static_singleton': self._template_classic_static,
            'classic_class_singleton': self._template_classic_class,
            'basic_singleton': self._template_basic,
            'incomplete_singleton': self._template_incomplete,
            'no_singleton': self._template_no_pattern,
            'false_positive': self._template_false_positive
        }
        
        self.relation_styles = {
            RelationType.INHERITANCE: "--|>",
            RelationType.COMPOSITION: "*--",
            RelationType.ASSOCIATION: "--",
            RelationType.DEPENDENCY: "..>",
            RelationType.DECORATOR: "..>"
        }
    
    def generate_uml(self, optimized_data: OptimizedData, ir_data: IntermediateRepresentation = None) -> str:
        """Genera el código UML final"""
        print("\nFASE 6: GENERACION DE CODIGO UML")
        print("-" * 40)
        
        self._ir_data = ir_data
        
        uml_content = self._generate_header()
        
        singleton_patterns = [p for p in optimized_data.patterns.values() 
                            if p.pattern_type != "no_singleton" and p.pattern_type != "false_positive"]
        
        all_involved_classes = set()
        for pattern in singleton_patterns:
            all_involved_classes.add(pattern.class_name)
        
        for relation in optimized_data.relations:
            all_involved_classes.add(relation.source)
            all_involved_classes.add(relation.target)
        
        if not singleton_patterns:
            uml_content += self._generate_no_patterns_found()
        else:
            for pattern in singleton_patterns:
                template_func = self.templates.get(pattern.pattern_type, self._template_unknown)
                uml_content += template_func(pattern)
            
            for class_name in all_involved_classes:
                if class_name not in [p.class_name for p in singleton_patterns]:
                    uml_content += self._template_related_class(class_name)
            
            uml_content += self._generate_relations(optimized_data.relations)
        
        if optimized_data.false_positives_removed:
            uml_content += self._generate_false_positives_section(optimized_data.false_positives_removed)
        
        uml_content += "\n@enduml"
        
        self._print_generation_results(len(singleton_patterns), len(optimized_data.relations))
        return uml_content
    
    def _generate_header(self) -> str:
        """Genera el encabezado del archivo UML"""
        return f"""@startuml
!theme cerulean-outline

title Análisis de Patrones Singleton
' Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    def _template_classic_static(self, pattern: SemanticInfo) -> str:
        """Template para Singleton con método estático"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<Singleton>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Patrón Singleton
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_new_override(self, pattern: SemanticInfo) -> str:
        """Template para Singleton con override de __new__"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<Singleton>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Singleton (__new__ override)
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_classic_class(self, pattern: SemanticInfo) -> str:
        """Template para Singleton con método de clase"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<Singleton>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Singleton (class method)
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_basic(self, pattern: SemanticInfo) -> str:
        """Template para Singleton básico"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<Singleton>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Singleton Básico
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_incomplete(self, pattern: SemanticInfo) -> str:
        """Template para Singleton incompleto"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<Incompleto>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Implementación incompleta
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_no_pattern(self, pattern: SemanticInfo) -> str:
        """Template para clases sin patrón"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} {{
{attributes}
  --
{methods}
}}

"""
    
    def _template_related_class(self, class_name: str) -> str:
        """Template para clases relacionadas"""
        class_info = self._get_class_info(class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {class_name} {{
{attributes}
  --
{methods}
}}

"""
    
    def _template_false_positive(self, pattern: SemanticInfo) -> str:
        """Template para falsos positivos"""
        return f"""
class {pattern.class_name} <<FalsoPositivo>> {{
}}

"""
    
    def _template_unknown(self, pattern: SemanticInfo) -> str:
        """Template para patrones desconocidos"""
        return f"""
class {pattern.class_name} {{
}}

"""
    
    def _get_class_info(self, class_name: str) -> Dict[str, Any]:
        """Obtiene información de una clase desde la representación intermedia"""
        if hasattr(self, '_ir_data') and self._ir_data and class_name in self._ir_data.classes:
            return self._ir_data.classes[class_name]
        return {'variables': [], 'methods': []}
    
    def _format_attributes(self, class_info: Dict[str, Any]) -> str:
        """Formatea los atributos para mostrar en UML"""
        variables = class_info.get('variables', [])
        if not variables:
            return "  ' Sin atributos"
        
        formatted_attrs = []
        
        class_vars = [v for v in variables if v.get('is_class_var', False)]
        instance_vars = [v for v in variables if not v.get('is_class_var', False)]
        
        for var in class_vars[:3]:
            visibility = "-" if var.get('is_private', False) else "+"
            var_name = var.get('name', 'unknown')
            var_type = var.get('type', 'Object')
            formatted_attrs.append(f"  {visibility} {var_name}: {var_type} {{static}}")
        
        for var in instance_vars[:5]:
            visibility = "-" if var.get('is_private', False) else "+"
            var_name = var.get('name', 'unknown')
            var_type = var.get('type', 'Object')
            formatted_attrs.append(f"  {visibility} {var_name}: {var_type}")
        
        return "\n".join(formatted_attrs) if formatted_attrs else "  ' Sin atributos"
    
    def _format_methods(self, class_info: Dict[str, Any]) -> str:
        """Formatea los métodos para mostrar en UML"""
        methods = class_info.get('methods', [])
        if not methods:
            return "  + __init__()"
        
        formatted_methods = []
        excluded_methods = ['__str__', '__repr__', '__del__', '__hash__', '__eq__']
        
        for method in methods:
            method_name = method.get('name', 'unknown')
            
            if method_name in excluded_methods:
                continue
            
            if method_name.startswith('__') and method_name.endswith('__'):
                visibility = "+"
            elif method_name.startswith('_'):
                visibility = "-"
            else:
                visibility = "+"
            
            if method.get('is_static', False):
                method_display = f"{method_name}() {{static}}"
            elif method.get('is_classmethod', False):
                method_display = f"{method_name}() {{class}}"
            else:
                method_display = f"{method_name}()"
            
            formatted_methods.append(f"  {visibility} {method_display}")
            
            if len(formatted_methods) >= 6:
                break
        
        if not formatted_methods:
            return "  + __init__()"
        
        return "\n".join(formatted_methods)
    
    def _generate_relations(self, relations: List[ClassRelation]) -> str:
        """Genera las relaciones UML"""
        if not relations:
            return ""
        
        relations_uml = "\n' Relaciones entre clases\n"
        
        for relation in relations:
            style = self._get_relation_style(relation.relation_type)
            relations_uml += f"{relation.source} {style} {relation.target}\n"
        
        return relations_uml + "\n"
    
    def _get_relation_style(self, relation_type: RelationType) -> str:
        """Obtiene el estilo UML para el tipo de relación"""
        return self.relation_styles.get(relation_type, "-->")
    
    def _generate_no_patterns_found(self) -> str:
        """Genera nota cuando no se encuentran patrones"""
        return """
note as NoPatterns
  No se encontraron patrones Singleton
  en el código analizado.
end note

"""
    
    def _generate_false_positives_section(self, false_positives: List[str]) -> str:
        """Genera sección de falsos positivos"""
        return f"""
note as FalsePositives
  Falsos Positivos: {len(false_positives)}
  Clases: {', '.join(false_positives)}
end note

"""
    
    def _print_generation_results(self, patterns_count: int, relations_count: int):
        print(f"Código UML generado:")
        print(f"   Patrones detectados: {patterns_count}")
        print(f"   Relaciones incluidas: {relations_count}")

# =============================================================================
# COMPILADOR PRINCIPAL
# =============================================================================

class SingletonCompiler:
    """Compilador principal que coordina todas las fases del análisis"""
    
    def __init__(self):
        self.lexer = LexicalAnalyzer()
        self.parser = SyntaxAnalyzer()
        self.semantic = SemanticAnalyzer()
        self.ir_generator = IntermediateCodeGenerator()
        self.optimizer = Optimizer()
        self.code_generator = UMLCodeGenerator()
    
    def compile(self, input_file: str, output_file: str) -> bool:
        """Ejecuta el proceso completo de compilación"""
        print("ANALISIS DE PATRONES SINGLETON")
        print("=" * 60)
        #print(f"Archivo de entrada: {input_file}")
        #print(f"Archivo de salida: {output_file}")
        #print("=" * 60)
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Ejecutar las 6 fases del compilador
            tokens = self.lexer.tokenize(source_code)
            
            ast_nodes, relations = self.parser.parse(source_code, tokens)
            if not ast_nodes and self.parser.syntax_errors:
                print("ERROR: Errores de sintaxis detectados.")
                return False
            
            semantic_info = self.semantic.analyze(ast_nodes, tokens)
            
            ir = self.ir_generator.generate(ast_nodes, semantic_info, relations)
            
            optimized_data = self.optimizer.optimize(ir)
            
            uml_code = self.code_generator.generate_uml(optimized_data, ir)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(uml_code)
            
            print("\n" + "=" * 60)
            print("COMPILACION COMPLETADA")
            print(f"Diagrama UML generado: {output_file}")
            
            # Intentar generar imagen PNG
            png_generated = self._generate_png(output_file)
            
            if png_generated:
                print(f"Imagen generada: {output_file.replace('.puml', '.png')}")
            else:
                print("Para generar imagen: plantuml " + output_file)
            
            print("=" * 60)
            return True
            
        except FileNotFoundError:
            print(f"ERROR: No se encontró el archivo {input_file}")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def _generate_png(self, puml_file: str) -> bool:
        """Intenta generar imagen PNG del diagrama UML"""
        try:
            plantuml_commands = [
                'plantuml',
                'java -jar plantuml.jar'
            ]
            
            for cmd in plantuml_commands:
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
