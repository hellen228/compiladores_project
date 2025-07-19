import re
import ast
from typing import List, Tuple, Optional, Any, Dict
from tokens import SingletonTokenType as TokenType
from core.models import Token, ASTNode, ClassRelation, RelationType, SemanticInfo

class SingletonLexicalAnalyzer:
    """Detecta tokens relacionados con patrones Singleton"""
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
    
    def get_token_stats(self) -> dict:
        """Retorna estadísticas para usar donde quieras"""
        return {tt.value: len([t for t in self.tokens if t.type == tt]) 
                for tt in TokenType if len([t for t in self.tokens if t.type == tt]) > 0}

    def _print_lexical_results(self):
        """Ahora usa get_token_stats()"""
        print(f"Tokens encontrados: {len(self.tokens)}")
        stats = self.get_token_stats()
        for token_type, count in stats.items():
            print(f" {token_type}: {count}")

class SingletonSyntaxAnalyzer:
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

class SingletonSemanticAnalyzer:
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
