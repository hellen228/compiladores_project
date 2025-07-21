import re
import ast
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from datetime import datetime
from data_structure import RelationType, ClassRelation, ASTNode, SemanticInfo
from data_structure import OptimizedData, IntermediateRepresentation

class TokenType(Enum):
    FACTORY_METHOD = "factory_method"
    FACTORY_CLASS = "factory_class"
    CREATOR_METHOD = "creator_method"
    PRODUCT_RETURN = "product_return"
    CONDITIONAL_CREATION = "conditional_creation"
    FACTORY_PARAMS = "factory_params"
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
    """Analizador léxico para detectar tokens relacionados con patrones Factory Method"""
    
    def __init__(self):
        self.tokens = []
        self.symbol_table = {}
        self.token_patterns = {
            TokenType.FACTORY_METHOD: [
                r'factory_method\b', r'create_\w+\b', r'make_\w+\b', 
                r'build_\w+\b', r'get_\w+\b', r'new_\w+\b',
                r'create\b', r'make\b', r'build\b', r'factory\b',

                r'crear_\w+\b', r'hacer_\w+\b', r'construir_\w+\b',
                r'obtener_\w+\b', r'nuevo_\w+\b', r'fabrica\b',
                r'crear\b', r'hacer\b', r'construir\b', r'obtener\b',
                
                # Patrones más generales para métodos factory
                r'def\s+\w*crear\w*\b', r'def\s+\w*create\w*\b',
                r'def\s+\w*make\w*\b', r'def\s+\w*build\w*\b'
            ],
            TokenType.FACTORY_CLASS: [
                r'\w*Factory\b', r'\w*Creator\b', r'\w*Builder\b',
                r'\w*Maker\b', r'Abstract\w+\b', r'Base\w+\b',

                r'\w*Fabrica\b', r'\w*Factory\b', r'\w*Creador\b',
                r'\w*Constructor\b', r'\w*Fabricante\b',
                r'Abstracto\w+\b', r'Base\w+\b'
            ],
            TokenType.CREATOR_METHOD: [
                r'@abstractmethod\b', r'@abc\.abstractmethod\b',
                r'def\s+create\w*\b', r'def\s+make\w*\b',
                r'def\s+build\w*\b', r'def\s+factory\w*\b',

                r'def\s+crear\w*\b', r'def\s+hacer\w*\b',
                r'def\s+construir\w*\b', r'def\s+obtener\w*\b'
            ],
            TokenType.PRODUCT_RETURN: [
                r'return\s+\w+\(\)', r'return\s+\w+\.\w+\(\)',
                r'return\s+self\.\w+\(\)', r'return\s+\w+Product\b',
                r'return\s+Concrete\w+\b',

                r'return\s+[A-Z]\w+\(\)',
                r'return\s+\w+\(\s*\)',
                r'return\s+\w+Instance\b'
            ],
            TokenType.CONDITIONAL_CREATION: [
                r'if\s+\w+\s*==', r'elif\s+\w+\s*==', r'else\s*:',
                r'if\s+isinstance\b', r'if\s+type\b', r'match\s+\w+',
                r'case\s+\w+', r'switch\b',

                r'if\s+tipo\s*==', r'elif\s+tipo\s*==',
                r'if\s+\w+\s*==\s*["\']', r'elif\s+\w+\s*==\s*["\']' 
            ],
            TokenType.FACTORY_PARAMS: [
                r'product_type\b', r'type_\w+\b', r'kind\b',
                r'variant\b', r'category\b', r'class_type\b',
                r'factory_type\b', r'creation_type\b',

                r'tipo\b', r'tipo_\w+\b', r'clase\b', r'categoria\b',
                r'variante\b', r'especie\b', r'modelo\b'
            ],
            TokenType.CLASS_DEF: [r'class\s+\w+']
        }
    
    def tokenize(self, source_code: str) -> List[Token]:
        """Ejecuta el análisis léxico del código fuente"""
        print("FASE 1: ANALISIS LEXICO - FACTORY METHOD")
        print("-" * 45)
        
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
            TokenType.FACTORY_METHOD: {
                'factory_method': 0.95, 'create': 0.85, 'make': 0.80,
                'build': 0.75, 'factory': 0.90, 'create_product': 0.95,
                'make_object': 0.90, 'build_instance': 0.85,
                'get_product': 0.80, 'new_instance': 0.85,

                'crear': 0.85, 'crear_': 0.90, 'hacer': 0.75,
                'construir': 0.80, 'obtener': 0.70, 'fabrica': 0.90,
                'crear_animal': 0.95, 'crear_producto': 0.95
            },
            TokenType.FACTORY_CLASS: {
                'Factory': 0.95, 'Creator': 0.90, 'Builder': 0.85,
                'Maker': 0.80, 'AbstractFactory': 0.95, 'BaseCreator': 0.90,
                'ProductFactory': 0.95, 'ObjectCreator': 0.90,

                'Fabrica': 0.95, 'Creador': 0.90, 'Constructor': 0.85,
                'Fabricante': 0.80, 'AnimalFactory': 0.95, 'Factory': 0.95
            },
            TokenType.CREATOR_METHOD: {
                '@abstractmethod': 0.95, '@abc.abstractmethod': 0.95,
                'def create': 0.90, 'def make': 0.85, 'def build': 0.80,
                'def factory': 0.85
            },
            TokenType.PRODUCT_RETURN: {
                'return': 0.70, 'return Product': 0.90, 'return Concrete': 0.85,
                'return self.': 0.75
            },
            TokenType.CONDITIONAL_CREATION: {
                'if': 0.60, 'elif': 0.65, 'else': 0.50,
                'if isinstance': 0.85, 'if type': 0.80,
                'match': 0.85, 'case': 0.80, 'switch': 0.75
            },
            TokenType.FACTORY_PARAMS: {
                'product_type': 0.95, 'type': 0.70, 'kind': 0.75,
                'variant': 0.80, 'category': 0.75, 'class_type': 0.85,
                'factory_type': 0.90, 'creation_type': 0.85
            }
        }
        
        # Buscar coincidencia parcial en el valor
        for key, conf in confidence_map.get(token_type, {}).items():
            if key.lower() in value.lower():
                return conf
        
        return 0.50
    
    def _print_lexical_results(self):
        print(f"Tokens encontrados: {len(self.tokens)}")
        for token_type in TokenType:
            count = len([t for t in self.tokens if t.type == token_type])
            if count > 0:
                print(f"  {token_type.value}: {count}")
        
        print("\nTabla de símbolos más relevantes:")
        # Mostrar solo los tokens con mayor confianza
        high_confidence_tokens = [
            (symbol, data) for symbol, data in self.symbol_table.items()
            if any(t.confidence > 0.8 for t in self.tokens if t.value == symbol)
        ]
        
        for symbol, data in sorted(high_confidence_tokens, 
                                 key=lambda x: x[1]['occurrences'], reverse=True)[:10]:
            print(f"  '{symbol}': {data['occurrences']} ocurrencias en líneas {data['lines']}")
    
    def get_factory_indicators(self) -> Dict[str, int]:
        """Retorna indicadores clave para la detección del patrón Factory Method"""
        indicators = {}
        
        # Contar tokens por tipo
        for token_type in TokenType:
            count = len([t for t in self.tokens if t.type == token_type])
            indicators[token_type.value] = count
        
        # Calcular métricas adicionales
        indicators['total_tokens'] = len(self.tokens)
        indicators['high_confidence_tokens'] = len([t for t in self.tokens if t.confidence > 0.8])
        indicators['factory_method_ratio'] = (
            indicators.get('factory_method', 0) / max(indicators['total_tokens'], 1)
        )
        
        return indicators

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
            
            # Detectar herencia (especialmente importante para Factory Method)
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
        """Detecta relaciones de uso entre clases específicas para Factory Method"""
        relations = []
        used_classes = {}  # Para evitar relaciones duplicadas
        
        # PASO 1: Analizar métodos factory para detectar COMPOSICIÓN (productos creados)
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                method_name = item.name
                
                # Detectar métodos factory típicos
                is_factory_method = any([
                    method_name.startswith('create_'),
                    method_name.startswith('make_'),
                    method_name.startswith('build_'),
                    method_name.startswith('get_'),
                    method_name in ['factory_method', 'create', 'make', 'build'],
                    
                    # Español (NUEVO)
                    method_name.startswith('crear_'),
                    method_name.startswith('hacer_'),
                    method_name.startswith('construir_'),
                    method_name.startswith('obtener_'),
                    method_name in ['crear', 'hacer', 'construir', 'obtener', 'fabrica'],
                    
                    # Patrones más generales
                    'crear' in method_name.lower(),
                    'create' in method_name.lower(),
                    'factory' in method_name.lower(),
                    'fabrica' in method_name.lower()
                ])
                
                for node in ast.walk(item):
                    # CASO 1: return OtraClase() -> COMPOSICIÓN (Factory crea productos)
                    if isinstance(node, ast.Return):
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                            if node.value.func.id in all_classes:
                                target_class = node.value.func.id
                                
                                # Si es un método factory, es composición fuerte
                                if is_factory_method:
                                    used_classes[target_class] = RelationType.COMPOSITION
                                    relations.append(ClassRelation(
                                        source=class_name,
                                        target=target_class,
                                        relation_type=RelationType.COMPOSITION,
                                        description=f"{class_name} factory crea {target_class}",
                                        line=node.lineno,
                                        confidence=0.95
                                    ))
                                else:
                                    # Otro tipo de método que retorna instancias
                                    used_classes[target_class] = RelationType.DEPENDENCY
                                    relations.append(ClassRelation(
                                        source=class_name,
                                        target=target_class,
                                        relation_type=RelationType.DEPENDENCY,
                                        description=f"{class_name} retorna {target_class}",
                                        line=node.lineno,
                                        confidence=0.85
                                    ))
                    
                    # CASO 2: Asignaciones a variables locales en métodos factory
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            # Variable local = OtraClase()
                            if isinstance(target, ast.Name):
                                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                                    if node.value.func.id in all_classes:
                                        target_class = node.value.func.id
                                        
                                        if target_class not in used_classes:
                                            if is_factory_method:
                                                used_classes[target_class] = RelationType.COMPOSITION
                                                relations.append(ClassRelation(
                                                    source=class_name,
                                                    target=target_class,
                                                    relation_type=RelationType.COMPOSITION,
                                                    description=f"{class_name} factory instancia {target_class}",
                                                    line=node.lineno,
                                                    confidence=0.90
                                                ))
                                            else:
                                                used_classes[target_class] = RelationType.DEPENDENCY
                                                relations.append(ClassRelation(
                                                    source=class_name,
                                                    target=target_class,
                                                    relation_type=RelationType.DEPENDENCY,
                                                    description=f"{class_name} instancia temporalmente {target_class}",
                                                    line=node.lineno,
                                                    confidence=0.80
                                                ))
                    
                    # CASO 3: Detectar uso de otras clases en condiciones (if product_type == 'A': return ClaseA())
                    elif isinstance(node, ast.If) or isinstance(node, ast.Match):
                        for child_node in ast.walk(node):
                            if isinstance(child_node, ast.Return):
                                if isinstance(child_node.value, ast.Call) and isinstance(child_node.value.func, ast.Name):
                                    if child_node.value.func.id in all_classes:
                                        target_class = child_node.value.func.id
                                        
                                        if target_class not in used_classes and is_factory_method:
                                            used_classes[target_class] = RelationType.COMPOSITION
                                            relations.append(ClassRelation(
                                                source=class_name,
                                                target=target_class,
                                                relation_type=RelationType.COMPOSITION,
                                                description=f"{class_name} factory crea condicionalmente {target_class}",
                                                line=child_node.lineno,
                                                confidence=0.95
                                            ))
        
        # PASO 2: Analizar atributos de instancia para detectar asociaciones
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
                            
                            # self.product_registry = {...} o self.products = []
                            if target.attr in ['products', 'product_registry', 'created_products', 'instances']:
                                # Es probablemente un registro de productos -> ASOCIACIÓN
                                pass  # No creamos relación aquí, es estructura interna
                            
                            # self.dependency = OtraClase() -> COMPOSICIÓN
                            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                                if node.value.func.id in all_classes:
                                    target_class = node.value.func.id
                                    if target_class not in used_classes:
                                        used_classes[target_class] = RelationType.COMPOSITION
                                        relations.append(ClassRelation(
                                            source=class_name,
                                            target=target_class,
                                            relation_type=RelationType.COMPOSITION,
                                            description=f"{class_name} compone {target_class}",
                                            line=node.lineno,
                                            confidence=0.90
                                        ))
        
        # PASO 3: Detectar dependencias por uso de métodos estáticos o de clase
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                for node in ast.walk(item):
                    # OtraClase.metodo_estatico()
                    if isinstance(node, ast.Attribute):
                        if isinstance(node.value, ast.Name) and node.value.id in all_classes:
                            target_class = node.value.id
                            
                            if target_class not in used_classes:
                                used_classes[target_class] = RelationType.DEPENDENCY
                                relations.append(ClassRelation(
                                    source=class_name,
                                    target=target_class,
                                    relation_type=RelationType.DEPENDENCY,
                                    description=f"{class_name} usa método de {target_class}",
                                    line=node.lineno,
                                    confidence=0.75
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
        """Construye un AST personalizado para el análisis de patrones Factory Method"""
        nodes = []
        
        for class_name, class_info in all_classes.items():
            class_node = class_info['node']
            
            ast_class_node = ASTNode(
                node_type="class",
                name=class_name,
                line=class_info['line'],
                metadata={
                    'bases': class_info['bases'],
                    'methods': class_info['methods'],
                    'is_abstract': self._is_abstract_class(class_node),
                    'has_factory_methods': self._has_factory_methods(class_node),
                    'factory_method_count': self._count_factory_methods(class_node)
                }
            )
            
            instance_attributes = self._extract_instance_attributes(class_node)
            
            # Analizar métodos con enfoque en factory methods
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
                            'is_classmethod': any('@classmethod' in str(d) for d in self._extract_decorators(item.decorator_list)),
                            'is_abstract': any('@abstractmethod' in str(d) for d in self._extract_decorators(item.decorator_list)),
                            'is_factory_method': self._is_factory_method(item),
                            'returns_object': self._method_returns_object(item),
                            'has_conditional_creation': self._has_conditional_creation(item)
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
                                    'is_class_var': True,
                                    'is_factory_registry': target.id in ['registry', 'products', 'factories', 'types']
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
                        'inferred_type': self._infer_attribute_type(attr_info['value']),
                        'is_product_registry': attr_name in ['products', 'created_products', 'instances', 'registry']
                    }
                )
                ast_class_node.children.append(attr_node)
            
            nodes.append(ast_class_node)
        
        return nodes
    
    def _is_abstract_class(self, class_node: ast.ClassDef) -> bool:
        """Determina si una clase es abstracta"""
        # Buscar import de abc
        has_abc_import = False
        # Buscar métodos abstractos
        has_abstract_methods = False
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                decorators = self._extract_decorators(item.decorator_list)
                if any('@abstractmethod' in d for d in decorators):
                    has_abstract_methods = True
                    break
        
        return has_abstract_methods or 'ABC' in [base for base in [self._extract_base_name(b) for b in class_node.bases]]
    
    def _has_factory_methods(self, class_node: ast.ClassDef) -> bool:
        """Determina si la clase tiene métodos factory"""
        return self._count_factory_methods(class_node) > 0
    
    def _count_factory_methods(self, class_node: ast.ClassDef) -> int:
        """Cuenta el número de métodos factory en la clase"""
        count = 0
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                if self._is_factory_method(item):
                    count += 1
        return count
    
    def _is_factory_method(self, method_node: ast.FunctionDef) -> bool:
        """Determina si un método es un factory method"""
        method_name = method_node.name
        
        # Nombres típicos de factory methods
        factory_names = [
            'create', 'make', 'build', 'factory_method', 'get_product',
            'new_instance', 'produce',
            'crear', 'hacer', 'construir', 'obtener', 'producir',
            'instanciar', 'generar', 'fabrica'
        ]
        
        factory_prefixes = ['create_', 'make_', 'build_', 'get_', 'new_', 'crear_', 'hacer_', 'construir_', 'obtener_', 'nuevo_']
        
        is_factory_name = (
            method_name in factory_names or
            any(method_name.startswith(prefix) for prefix in factory_prefixes)
        )
        
        # Verificar si retorna objetos
        returns_objects = self._method_returns_object(method_node)
        
        # Es abstracto
        decorators = self._extract_decorators(method_node.decorator_list)
        is_abstract = any('@abstractmethod' in d for d in decorators)
        
        return is_factory_name or returns_objects or is_abstract
    
    def _method_returns_object(self, method_node: ast.FunctionDef) -> bool:
        """Verifica si un método retorna instancias de objetos"""
        for node in ast.walk(method_node):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Call):
                    # Retorna una llamada a constructor
                    return True
        return False
    
    def _has_conditional_creation(self, method_node: ast.FunctionDef) -> bool:
        """Verifica si un método tiene creación condicional de objetos"""
        has_conditions = False
        has_object_creation = False
        
        for node in ast.walk(method_node):
            if isinstance(node, (ast.If, ast.Match)):
                has_conditions = True
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                has_object_creation = True
        
        return has_conditions and has_object_creation
    
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
            elif value in ['[]', 'list()', '{}', 'dict()']:
                return "collection"
            else:
                return "str"
        elif isinstance(value, (int, float, bool)):
            return type(value).__name__
        else:
            return "Object"
    
    def _analyze_method_body(self, method_node: ast.FunctionDef) -> List[ASTNode]:
        """Analiza el contenido de un método con enfoque en factory patterns"""
        body_nodes = []
        
        for stmt in ast.walk(method_node):
            if isinstance(stmt, ast.If):
                condition_node = ASTNode(
                    node_type="conditional",
                    name="if_statement",
                    line=stmt.lineno,
                    metadata={
                        'condition_type': self._analyze_condition(stmt.test),
                        'is_factory_condition': self._is_factory_condition(stmt.test)
                    }
                )
                body_nodes.append(condition_node)
            
            elif isinstance(stmt, ast.Match):  # Para Python 3.10+
                match_node = ASTNode(
                    node_type="match_statement",
                    name="match_case",
                    line=stmt.lineno,
                    metadata={'is_factory_match': True}
                )
                body_nodes.append(match_node)
            
            elif isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Name):
                    call_node = ASTNode(
                        node_type="function_call",
                        name=stmt.func.id,
                        line=stmt.lineno,
                        metadata={
                            'args_count': len(stmt.args),
                            'is_constructor_call': stmt.func.id[0].isupper()
                        }
                    )
                    body_nodes.append(call_node)
            
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Call):
                    return_node = ASTNode(
                        node_type="return_statement",
                        name="object_creation",
                        line=stmt.lineno,
                        metadata={
                            'returns_object': True,
                            'constructor_name': stmt.value.func.id if isinstance(stmt.value.func, ast.Name) else 'unknown'
                        }
                    )
                    body_nodes.append(return_node)
        
        return body_nodes
    
    def _analyze_condition(self, test_node: ast.expr) -> str:
        """Analiza el tipo de condición en estructuras de control"""
        if isinstance(test_node, ast.UnaryOp) and isinstance(test_node.op, ast.Not):
            return "not_condition"
        elif isinstance(test_node, ast.Compare):
            return "comparison"
        elif isinstance(test_node, ast.Call):
            if isinstance(test_node.func, ast.Name) and test_node.func.id == 'isinstance':
                return "isinstance_check"
        return "unknown"
    
    def _is_factory_condition(self, test_node: ast.expr) -> bool:
        """Determina si una condición es típica de factory method"""
        if isinstance(test_node, ast.Compare):
            # Buscar comparaciones como product_type == 'A'
            if isinstance(test_node.left, ast.Name):
                var_name = test_node.left.id

                factory_var_names = ['type', 'product_type', 'kind', 'variant', 'category',
                                      'class_type', 'object_type', 'instance_type',
                                      'tipo', 'tipo_producto', 'clase', 'variante', 'categoria',
                                      'tipo_clase', 'tipo_objeto', 'especie', 'modelo']
                
                return var_name in factory_var_names or any(name in var_name.lower() for name in ['type', 'kind', 'variant'])
        
        elif isinstance(test_node, ast.Call):
            # isinstance checks
            if isinstance(test_node.func, ast.Name) and test_node.func.id == 'isinstance':
                return True
        
        return False
    
    def _extract_value(self, node: ast.expr) -> Any:
        """Extrae el valor de un nodo AST"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[]"
        elif isinstance(node, ast.Dict):
            return "{}"
        return "complex_expression"
    
    def _print_syntax_results(self):
        classes = [node for node in self.ast_nodes if node.node_type == "class"]
        print(f"Clases encontradas: {len(classes)}")
        
        factory_classes = 0
        abstract_classes = 0
        
        for class_node in classes:
            methods = [child for child in class_node.children if child.node_type == "method"]
            variables = [child for child in class_node.children if child.node_type in ["class_variable", "instance_variable"]]
            factory_methods = [m for m in methods if m.metadata.get('is_factory_method', False)]
            
            print(f"   {class_node.name}: {len(methods)} métodos, {len(variables)} atributos, {len(factory_methods)} factory methods")
            
            if class_node.metadata.get('is_abstract', False):
                abstract_classes += 1
            if class_node.metadata.get('has_factory_methods', False):
                factory_classes += 1
        
        print(f"Clases abstractas: {abstract_classes}")
        print(f"Clases con métodos factory: {factory_classes}")
        print(f"Relaciones encontradas: {len(self.relations)}")


# =============================================================================
# FASE 3: ANÁLISIS SEMÁNTICO 
# =============================================================================

class SemanticAnalyzer:
    """Analizador semántico que identifica patrones Factory Method"""
    
    def __init__(self):
        self.semantic_errors = []
        self.pattern_info = {}
    
    def analyze(self, ast_nodes: List[ASTNode], tokens: List[Token]) -> Dict[str, SemanticInfo]:
        """Ejecuta el análisis semántico para identificar patrones Factory Method"""
        print("\nFASE 3: ANALISIS SEMANTICO - FACTORY METHOD")
        print("-" * 50)
        
        semantic_results = {}
        
        for class_node in [node for node in ast_nodes if node.node_type == "class"]:
            analysis = self._analyze_class_semantics(class_node, tokens)
            semantic_results[class_node.name] = analysis
        
        self._print_semantic_results(semantic_results)
        return semantic_results
    
    def _analyze_class_semantics(self, class_node: ASTNode, tokens: List[Token]) -> SemanticInfo:
        """Analiza la semántica específica de una clase para patrones Factory Method"""
        evidences = {}
        violations = []
        
        # Buscar métodos factory
        factory_methods = self._find_factory_methods(class_node)
        if factory_methods:
            evidences['factory_method'] = {
                'found': True,
                'methods': factory_methods,
                'confidence': self._calculate_factory_method_confidence(factory_methods)
            }
        
        # Buscar métodos creator (abstractos)
        creator_methods = self._find_creator_methods(class_node)
        if creator_methods:
            evidences['creator_method'] = {
                'found': True,
                'methods': creator_methods,
                'confidence': self._calculate_creator_method_confidence(creator_methods)
            }
        
        # Buscar creación condicional de productos
        conditional_creation = self._find_conditional_creation(class_node)
        if conditional_creation:
            evidences['conditional_creation'] = {
                'found': True,
                'patterns': conditional_creation,
                'confidence': self._calculate_conditional_confidence(conditional_creation)
            }
        
        # Buscar retorno de productos
        product_returns = self._find_product_returns(class_node)
        if product_returns:
            evidences['product_return'] = {
                'found': True,
                'returns': product_returns,
                'confidence': self._calculate_product_return_confidence(product_returns)
            }
        
        # Buscar parámetros de factory
        factory_params = self._find_factory_params(class_node)
        if factory_params:
            evidences['factory_params'] = {
                'found': True,
                'params': factory_params,
                'confidence': self._calculate_factory_params_confidence(factory_params)
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
    
    def _find_factory_methods(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Busca métodos factory en la clase"""
        factory_methods = []
        for child in class_node.children:
            if child.node_type == "method":
                method_name = child.name.lower()
                if any(keyword in method_name for keyword in 
                      ['create', 'make', 'build', 'factory', 'new', 'get', 'produce',
                       'crear', 'hacer', 'generar', 'construir', 'fabricar', 'obtener', 'nuevo', 'nueva']):
                    factory_methods.append({
                        'name': child.name,
                        'decorators': child.metadata.get('decorators', []),
                        'args': child.metadata.get('args', []),
                        'line': child.line,
                        'is_abstract': self._is_abstract_method(child)
                    })
        return factory_methods
    
    def _find_creator_methods(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Busca métodos creator (típicamente abstractos)"""
        creator_methods = []
        for child in class_node.children:
            if child.node_type == "method":
                if (self._is_abstract_method(child) or 
                    'NotImplementedError' in str(child.metadata.get('body', ''))):
                    creator_methods.append({
                        'name': child.name,
                        'is_abstract': True,
                        'line': child.line,
                        'args': child.metadata.get('args', [])
                    })
        return creator_methods
    
    def _find_conditional_creation(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Busca patrones de creación condicional"""
        conditional_patterns = []
        for child in class_node.children:
            if child.node_type == "method":
                body = str(child.metadata.get('body', ''))
                if ('if' in body and any(keyword in body.lower() for keyword in 
                    ['create', 'make', 'build', 'new', 'return', 
                     'crear', 'hacer', 'construir', 'fabricar', 'generar', 'nuevo', 'nueva'])):
                    conditional_patterns.append({
                        'method': child.name,
                        'line': child.line,
                        'conditions': self._extract_conditions(body)
                    })
        return conditional_patterns
    
    def _find_product_returns(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Busca métodos que retornan productos"""
        product_returns = []
        for child in class_node.children:
            if child.node_type == "method":
                return_type = child.metadata.get('return_type', '')
                body = str(child.metadata.get('body', ''))
                if ('return' in body and 
                    any(pattern in body.lower() for pattern in 
                    ['()', '.create', '.make', '.build'])):
                    product_returns.append({
                        'method': child.name,
                        'return_type': return_type,
                        'line': child.line,
                        'creates_object': True
                    })
        return product_returns
    
    def _find_factory_params(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Busca parámetros típicos de factory methods"""
        factory_params = []
        for child in class_node.children:
            if child.node_type == "method":
                args = child.metadata.get('args', [])
                for arg in args:
                    if any(keyword in arg.lower() for keyword in 
                          ['type', 'kind', 'class', 'category', 'variant', 'mode'
                           'tipo', 'clase', 'categoria', 'variante', 'modo']):
                        factory_params.append({
                            'method': child.name,
                            'param': arg,
                            'line': child.line
                        })
        return factory_params
    
    def _is_abstract_method(self, method_node: ASTNode) -> bool:
        """Verifica si un método es abstracto"""
        decorators = method_node.metadata.get('decorators', [])
        body = str(method_node.metadata.get('body', ''))
        return ('@abstractmethod' in str(decorators) or 
                'NotImplementedError' in body or
                'raise NotImplementedError' in body)
    
    def _extract_conditions(self, body: str) -> List[str]:
        """Extrae condiciones de creación del cuerpo del método"""
        conditions = []
        lines = body.split('\n')
        for line in lines:
            if 'if' in line and ('==' in line or 'in' in line):
                conditions.append(line.strip())
        return conditions
    
    def _determine_pattern_type(self, evidences: Dict[str, Any]) -> str:
        """Determina el tipo específico de patrón Factory Method"""
        has_factory_method = evidences.get('factory_method', {}).get('found', False)
        has_creator_method = evidences.get('creator_method', {}).get('found', False)
        has_conditional = evidences.get('conditional_creation', {}).get('found', False)
        has_product_return = evidences.get('product_return', {}).get('found', False)
        
        if has_creator_method and has_factory_method:
            return "abstract_factory_method"
        elif has_factory_method and has_conditional and has_product_return:
            return "concrete_factory_method"
        elif has_factory_method and has_product_return:
            return "simple_factory_method"
        elif has_factory_method:
            return "basic_factory_method"
        elif has_creator_method:
            return "incomplete_factory_method"
        else:
            return "no_factory_method"
    
    def _calculate_factory_method_confidence(self, methods: List[Dict[str, Any]]) -> float:
        """Calcula confianza basada en métodos factory"""
        if not methods:
            return 0.0
        
        confidence_map = {
            'create': 0.95, 'make': 0.90, 'build': 0.85, 'factory': 0.90,
            'new': 0.80, 'get': 0.70, 'produce': 0.85,

            'crear': 0.95, 'hacer':0.90, 'construir': 0.85, 'fabricar': 0.90, 
            'obtener':0.70, 'nuevo':0.80, 'nueva':0.80, 'producir': 0.85
        }
        
        max_confidence = 0.0
        for method in methods:
            name = method['name'].lower()
            for keyword, conf in confidence_map.items():
                if keyword in name:
                    max_confidence = max(max_confidence, conf)
                    break
            else:
                max_confidence = max(max_confidence, 0.50)
        
        return max_confidence
    
    def _calculate_creator_method_confidence(self, methods: List[Dict[str, Any]]) -> float:
        """Calcula confianza basada en métodos creator"""
        if not methods:
            return 0.0
        
        abstract_methods = sum(1 for m in methods if m.get('is_abstract', False))
        base_confidence = 0.60
        
        if abstract_methods > 0:
            base_confidence = 0.85
        
        return min(base_confidence + (len(methods) * 0.05), 1.0)
    
    def _calculate_conditional_confidence(self, patterns: List[Dict[str, Any]]) -> float:
        """Calcula confianza basada en creación condicional"""
        if not patterns:
            return 0.0
        
        base_confidence = 0.70
        for pattern in patterns:
            conditions = pattern.get('conditions', [])
            if len(conditions) > 1:
                base_confidence = min(base_confidence + 0.10, 1.0)
        
        return base_confidence
    
    def _calculate_product_return_confidence(self, returns: List[Dict[str, Any]]) -> float:
        """Calcula confianza basada en retorno de productos"""
        if not returns:
            return 0.0
        
        creating_methods = sum(1 for r in returns if r.get('creates_object', False))
        if creating_methods > 0:
            return 0.80
        
        return 0.60
    
    def _calculate_factory_params_confidence(self, params: List[Dict[str, Any]]) -> float:
        """Calcula confianza basada en parámetros de factory"""
        if not params:
            return 0.0
        
        confidence_map = {
            'type': 0.90, 'kind': 0.85, 'class': 0.80,
            'category': 0.75, 'variant': 0.70, 'mode': 0.65,
            'tipo':0.90, 'clase':0.80, 'categoria':0.75, 
            'variante':0.70, 'modo':0.65
        }
        
        max_confidence = 0.0
        for param in params:
            param_name = param['param'].lower()
            for keyword, conf in confidence_map.items():
                if keyword in param_name:
                    max_confidence = max(max_confidence, conf)
                    break
        
        return max_confidence if max_confidence > 0 else 0.50
    
    def _calculate_overall_confidence(self, evidences: Dict[str, Any]) -> float:
        """Calcula la confianza general del patrón detectado"""
        weights = {
            'factory_method': 0.30,
            'creator_method': 0.25,
            'conditional_creation': 0.20,
            'product_return': 0.15,
            'factory_params': 0.10
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
        """Verifica violaciones en la implementación del patrón Factory Method"""
        violations = []
        
        # Verificar múltiples métodos factory sin jerarquía clara
        factory_methods = evidences.get('factory_method', {}).get('methods', [])
        if len(factory_methods) > 3:
            method_names = [m['name'] for m in factory_methods]
            violations.append(f"Demasiados métodos factory potenciales: {method_names}")
        
        # Verificar métodos creator sin implementación
        creator_methods = evidences.get('creator_method', {}).get('methods', [])
        abstract_count = sum(1 for m in creator_methods if m.get('is_abstract', False))
        if abstract_count > 0 and len(factory_methods) == 0:
            violations.append("Métodos creator abstractos sin métodos factory concretos")
        
        # Verificar falta de creación condicional en factory methods complejos
        has_conditional = evidences.get('conditional_creation', {}).get('found', False)
        if len(factory_methods) > 1 and not has_conditional:
            violations.append("Múltiples factory methods sin lógica condicional clara")
        
        return violations
    
    def _print_semantic_results(self, results: Dict[str, SemanticInfo]):
        print(f"Clases analizadas: {len(results)}")
        for class_name, info in results.items():
            if info.pattern_type != "no_factory_method":
                print(f"   {class_name}: {info.pattern_type} (confianza: {info.confidence:.2f})")
                if info.violations:
                    print(f"      Violaciones: {', '.join(info.violations)}")

# =============================================================================
# FASE 4: GENERACIÓN DE REPRESENTACIÓN INTERMEDIA - FACTORY METHOD
# =============================================================================

class IntermediateCodeGenerator:
    """Generador de representación intermedia con información de clases y relaciones Factory Method"""
    
    def __init__(self):
        self.ir = None
    
    def generate(self, ast_nodes: List[ASTNode], semantic_info: Dict[str, SemanticInfo], 
                relations: List[ClassRelation]) -> IntermediateRepresentation:
        """Genera la representación intermedia del análisis Factory Method"""
        print("\nFASE 4: GENERACION DE CODIGO INTERMEDIO - FACTORY METHOD")
        print("-" * 55)
        
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
                'complexity': self._calculate_complexity(class_node),
                'factory_characteristics': self._extract_factory_characteristics(class_node)
            }
            
            if class_name in semantic_info:
                patterns[class_name] = semantic_info[class_name]
        
        filtered_relations = self._filter_relevant_relations(relations, patterns)
        
        global_metadata = {
            'total_classes': len(classes),
            'factory_patterns': len([p for p in patterns.values() if p.pattern_type != "no_factory_method"]),
            'creator_classes': len([p for p in patterns.values() if 'abstract_factory' in p.pattern_type]),
            'concrete_factories': len([p for p in patterns.values() if 'concrete_factory' in p.pattern_type]),
            'total_relations': len(filtered_relations),
            'inheritance_depth': self._calculate_inheritance_depth(filtered_relations),
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
        """Filtra relaciones relevantes para el análisis de patrones Factory Method"""
        factory_classes = set()
        
        for class_name, pattern_info in patterns.items():
            if pattern_info.pattern_type != "no_factory_method" and pattern_info.confidence >= 0.5:
                factory_classes.add(class_name)
        
        relevant_relations = []
        
        for relation in relations:
            # Incluir relaciones de herencia (muy importantes para Factory Method)
            if relation.relation_type == RelationType.INHERITANCE:
                if (relation.target in factory_classes or 
                    relation.source in factory_classes):
                    relevant_relations.append(relation)
            
            # Incluir relaciones de composición y asociación para productos
            elif relation.relation_type in [RelationType.COMPOSITION, RelationType.ASSOCIATION]:
                if (relation.target in factory_classes or 
                    relation.source in factory_classes):
                    relevant_relations.append(relation)
            
            # Incluir dependencias para creación de productos
            elif relation.relation_type == RelationType.DEPENDENCY:
                if relation.source in factory_classes:
                    relevant_relations.append(relation)
        
        return relevant_relations
    
    def _extract_factory_characteristics(self, class_node: ASTNode) -> Dict[str, Any]:
        """Extrae características específicas de Factory Method de la clase"""
        characteristics = {
            'factory_methods': [],
            'creator_methods': [],
            'product_types': [],
            'creation_parameters': [],
            'has_abstract_methods': False,
            'inheritance_level': 0
        }
        
        for child in class_node.children:
            if child.node_type == "method":
                method_name = child.name.lower()
                
                # Identificar métodos factory
                if any(keyword in method_name for keyword in 
                      ['create', 'make', 'build', 'factory', 'new', 'get', 'produce',
                       'crear', 'hacer', 'construir', 'nuevo', 'nueva', 'conseguir', 'producir']):
                    characteristics['factory_methods'].append({
                        'name': child.name,
                        'args': child.metadata.get('args', []),
                        'is_abstract': self._is_abstract_method(child),
                        'line': child.line
                    })
                
                # Identificar métodos creator
                if self._is_abstract_method(child):
                    characteristics['creator_methods'].append({
                        'name': child.name,
                        'args': child.metadata.get('args', []),
                        'line': child.line
                    })
                    characteristics['has_abstract_methods'] = True
                
                # Extraer parámetros de creación
                args = child.metadata.get('args', [])
                for arg in args:
                    if any(keyword in arg.lower() for keyword in 
                          ['type', 'kind', 'class', 'category', 'variant', 'mode',
                           'tipo', 'clase', 'categoria', 'variante', 'modo']):
                        characteristics['creation_parameters'].append(arg)
        
        # Inferir tipos de productos basado en métodos y retornos
        characteristics['product_types'] = self._infer_product_types(class_node)
        
        return characteristics
    
    def _infer_product_types(self, class_node: ASTNode) -> List[str]:
        """Infiere los tipos de productos que puede crear la factory"""
        product_types = []
        
        for child in class_node.children:
            if child.node_type == "method":
                body = str(child.metadata.get('body', ''))
                return_type = child.metadata.get('return_type', '')
                
                # Buscar instanciaciones en el cuerpo del método
                import re
                instantiation_pattern = r'(\w+)\s*\('
                matches = re.findall(instantiation_pattern, body)
                
                for match in matches:
                    if match[0].isupper():  # Probable clase
                        product_types.append(match)
                
                # Considerar tipo de retorno si está especificado
                if return_type and return_type != 'None':
                    product_types.append(return_type)
        
        return list(set(product_types))  # Eliminar duplicados
    
    def _is_abstract_method(self, method_node: ASTNode) -> bool:
        """Verifica si un método es abstracto"""
        decorators = method_node.metadata.get('decorators', [])
        body = str(method_node.metadata.get('body', ''))
        return ('@abstractmethod' in str(decorators) or 
                'NotImplementedError' in body or
                'raise NotImplementedError' in body)
    
    def _calculate_inheritance_depth(self, relations: List[ClassRelation]) -> int:
        """Calcula la profundidad máxima de herencia en las relaciones"""
        inheritance_relations = [r for r in relations if r.relation_type == RelationType.INHERITANCE]
        
        if not inheritance_relations:
            return 0
        
        # Construir grafo de herencia
        inheritance_graph = {}
        for relation in inheritance_relations:
            if relation.source not in inheritance_graph:
                inheritance_graph[relation.source] = []
            inheritance_graph[relation.source].append(relation.target)
        
        # Calcular profundidad máxima
        max_depth = 0
        visited = set()
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            if node in visited:
                return current_depth
            visited.add(node)
            
            max_depth = max(max_depth, current_depth)
            
            if node in inheritance_graph:
                for child in inheritance_graph[node]:
                    calculate_depth(child, current_depth + 1)
            
            return current_depth
        
        for node in inheritance_graph:
            calculate_depth(node)
        
        return max_depth
    
    def _normalize_methods(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        """Normaliza información de métodos de la clase con enfoque en Factory Method"""
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
                    'is_abstract': self._is_abstract_method(child),
                    'visibility': 'private' if child.name.startswith('_') else 'public',
                    'is_factory_method': self._is_factory_method_name(child.name),
                    'is_creator_method': self._is_abstract_method(child),
                    'return_type': child.metadata.get('return_type', 'Unknown')
                }
                methods.append(method_info)
        return methods
    
    def _is_factory_method_name(self, method_name: str) -> bool:
        """Verifica si el nombre del método sugiere que es un factory method"""
        name_lower = method_name.lower()
        return any(keyword in name_lower for keyword in 
                  ['create', 'make', 'build', 'factory', 'new', 'get', 'produce',
                   'crear', 'hacer', 'construir', 'nuevo', 'nueva', 'conseguir', 'producir'])
    
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
                    'type': self._infer_variable_type(child.metadata.get('value')),
                    'is_product_registry': self._is_product_registry(child.name)
                })
        
        for child in class_node.children:
            if child.node_type == "instance_variable":
                variables.append({
                    'name': child.name,
                    'line': child.line,
                    'value': child.metadata.get('value'),
                    'is_private': child.name.startswith('_'),
                    'is_class_var': False,
                    'type': child.metadata.get('inferred_type', 'Object'),
                    'is_product_registry': self._is_product_registry(child.name)
                })
        
        return variables
    
    def _is_product_registry(self, var_name: str) -> bool:
        """Verifica si la variable podría ser un registro de productos"""
        name_lower = var_name.lower()
        return any(keyword in name_lower for keyword in 
                  ['registry', 'types', 'products', 'classes', 'factories', 'creators',
                   'registro', 'tipos', 'productos', 'clases', 'constructores', 'creadores'])
    
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
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, list):
            return "list"
        else:
            return "Object"
    
    def _calculate_complexity(self, class_node: ASTNode) -> int:
        """Calcula la complejidad de la clase considerando Factory Method"""
        complexity = 0
        
        # Métodos base
        methods = [c for c in class_node.children if c.node_type == "method"]
        complexity += len(methods)
        
        # Variables
        variables = [c for c in class_node.children if c.node_type in ["class_variable", "instance_variable"]]
        complexity += len(variables)
        
        # Bonificación por métodos abstractos (aumenta complejidad)
        abstract_methods = [m for m in methods if self._is_abstract_method(m)]
        complexity += len(abstract_methods) * 2
        
        # Bonificación por métodos factory
        factory_methods = [m for m in methods if self._is_factory_method_name(m.name)]
        complexity += len(factory_methods)
        
        return complexity
    
    def _print_ir_results(self):
        print(f"Representación intermedia generada:")
        print(f"   Clases: {len(self.ir.classes)}")
        print(f"   Patrones Factory Method: {self.ir.global_metadata['factory_patterns']}")
        print(f"   Clases Creator: {self.ir.global_metadata['creator_classes']}")
        print(f"   Factories Concretas: {self.ir.global_metadata['concrete_factories']}")
        print(f"   Relaciones: {len(self.ir.relations)}")
        print(f"   Profundidad de herencia: {self.ir.global_metadata['inheritance_depth']}")
        
        # Mostrar detalles de las clases factory más relevantes
        factory_classes = [name for name, pattern in self.ir.patterns.items() 
                          if pattern.pattern_type != "no_factory_method" and pattern.confidence >= 0.7]
        
        if factory_classes:
            print(f"   Clases Factory detectadas: {', '.join(factory_classes)}")

# =============================================================================
# FASE 5: OPTIMIZACIÓN 
# =============================================================================

class Optimizer:
    """Optimizador que mejora la detección de Factory Method y filtra falsos positivos"""
    
    def __init__(self):
        self.optimized_data = None
    
    def optimize(self, ir: IntermediateRepresentation) -> OptimizedData:
        """Ejecuta las optimizaciones sobre la representación intermedia Factory Method"""
        print("\nFASE 5: OPTIMIZACION - FACTORY METHOD")
        print("-" * 45)
        
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
        """Optimiza la información de un patrón Factory Method específico"""
        optimized = SemanticInfo(
            class_name=pattern_info.class_name,
            pattern_type=pattern_info.pattern_type,
            confidence=pattern_info.confidence,
            evidences=pattern_info.evidences.copy(),
            violations=pattern_info.violations.copy()
        )
        
        # Bonificar por herencia (crítico en Factory Method)
        inheritance_relations = [r for r in relations if 
                               (r.source == pattern_info.class_name or r.target == pattern_info.class_name) 
                               and r.relation_type == RelationType.INHERITANCE]
        
        if len(inheritance_relations) >= 2:
            # Múltiples niveles de herencia indican jerarquía factory bien estructurada
            optimized.confidence = min(optimized.confidence * 1.20, 1.0)
        elif len(inheritance_relations) == 1:
            # Herencia simple, típica en Factory Method
            optimized.confidence = min(optimized.confidence * 1.10, 1.0)
        
        # Bonificar por relaciones de dependencia (factory crea productos)
        dependency_relations = [r for r in relations if 
                              r.source == pattern_info.class_name 
                              and r.relation_type == RelationType.DEPENDENCY]
        
        if len(dependency_relations) >= 3:
            # Factory que crea múltiples tipos de productos
            optimized.confidence = min(optimized.confidence * 1.15, 1.0)
        elif len(dependency_relations) >= 1:
            # Factory que crea al menos un producto
            optimized.confidence = min(optimized.confidence * 1.05, 1.0)
        
        # Bonificar por métodos abstractos en clases base
        factory_characteristics = class_info.get('factory_characteristics', {})
        if factory_characteristics.get('has_abstract_methods', False):
            creator_methods = factory_characteristics.get('creator_methods', [])
            if len(creator_methods) >= 1:
                # Clase creator con métodos abstractos
                optimized.confidence = min(optimized.confidence * 1.10, 1.0)
        
        # Bonificar por múltiples factory methods con parámetros de tipo
        factory_methods = factory_characteristics.get('factory_methods', [])
        creation_parameters = factory_characteristics.get('creation_parameters', [])
        
        if len(factory_methods) >= 2 and len(creation_parameters) >= 1:
            # Factory compleja con múltiples métodos y parámetros de tipo
            optimized.confidence = min(optimized.confidence * 1.08, 1.0)
        
        # Penalizar ausencia de productos identificados
        product_types = factory_characteristics.get('product_types', [])
        if not product_types and optimized.pattern_type not in ["incomplete_factory_method", "basic_factory_method"]:
            # Factory sin productos identificados es sospechoso
            optimized.confidence *= 0.85
        
        # Bonificar consistencia en nomenclatura
        if self._has_consistent_naming(factory_methods):
            optimized.confidence = min(optimized.confidence * 1.05, 1.0)
        
        return optimized
    
    def _has_consistent_naming(self, factory_methods: List[Dict[str, Any]]) -> bool:
        """Verifica si los métodos factory tienen nomenclatura consistente"""
        if len(factory_methods) <= 1:
            return True
        
        # Verificar si todos usan el mismo patrón de nomenclatura
        prefixes = set()
        suffixes = set()
        
        for method in factory_methods:
            name = method['name'].lower()
            if name.startswith('create'):
                prefixes.add('create')
            elif name.startswith('crear'):
                prefixes.add('crear')
            elif name.startswith('make'):
                prefixes.add('make')
            elif name.startswith('hacer'):
                prefixes.add('hacer')
            elif name.startswith('build'):
                prefixes.add('build')
            elif name.startswith('construir'):
                prefixes.add('construir')
            
            if name.endswith('factory'):
                suffixes.add('factory')
            elif name.endswith('method'):
                suffixes.add('method')
        
        # Consistente si todos usan el mismo prefijo o sufijo
        return len(prefixes) <= 1 or len(suffixes) <= 1
    
    def _optimize_relations(self, relations: List[ClassRelation], 
                          patterns: Dict[str, SemanticInfo]) -> List[ClassRelation]:
        """Optimiza y filtra las relaciones detectadas para Factory Method"""
        optimized_relations = []
        
        for relation in relations:
            source_pattern = patterns.get(relation.source)
            target_pattern = patterns.get(relation.target)
            
            valid_source = (source_pattern and 
                           source_pattern.pattern_type != "no_factory_method" and 
                           source_pattern.pattern_type != "false_positive" and
                           source_pattern.confidence >= 0.5)
            
            valid_target = (target_pattern and 
                           target_pattern.pattern_type != "no_factory_method" and 
                           target_pattern.pattern_type != "false_positive" and
                           target_pattern.confidence >= 0.5)
            
            # Incluir relaciones importantes para Factory Method
            if valid_source or valid_target:
                # Siempre incluir herencia (crítica para Factory Method)
                if relation.relation_type == RelationType.INHERITANCE:
                    optimized_relations.append(relation)
                
                # Incluir dependencias de factories a productos
                elif (relation.relation_type == RelationType.DEPENDENCY and 
                      valid_source):
                    optimized_relations.append(relation)
                
                # Incluir composición y asociación relevantes
                elif (relation.relation_type in [RelationType.COMPOSITION, RelationType.ASSOCIATION] and
                      (valid_source or valid_target)):
                    optimized_relations.append(relation)
        
        # Filtrar relaciones duplicadas o redundantes
        optimized_relations = self._filter_redundant_relations(optimized_relations)
        
        return optimized_relations
    
    def _filter_redundant_relations(self, relations: List[ClassRelation]) -> List[ClassRelation]:
        """Filtra relaciones redundantes o duplicadas"""
        seen_relations = set()
        filtered_relations = []
        
        for relation in relations:
            # Crear clave única para la relación
            relation_key = (relation.source, relation.target, relation.relation_type.value)
            
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                filtered_relations.append(relation)
        
        return filtered_relations
    
    def _is_false_positive(self, pattern_info: SemanticInfo, class_info: Dict[str, Any]) -> bool:
        """Detecta falsos positivos en la detección de patrones Factory Method"""
        # Confianza muy baja
        if pattern_info.confidence < 0.3:
            return True
        
        # Múltiples violaciones serias
        if len(pattern_info.violations) >= 2:
            return True
        
        # Verificaciones específicas de Factory Method
        factory_characteristics = class_info.get('factory_characteristics', {})
        
        # Clase sin métodos factory ni creator
        factory_methods = factory_characteristics.get('factory_methods', [])
        creator_methods = factory_characteristics.get('creator_methods', [])
        
        if not factory_methods and not creator_methods:
            return True
        
        # Factory Method "abstracto" sin subclases
        if (pattern_info.pattern_type == "abstract_factory_method" and 
            creator_methods and 
            not self._has_concrete_subclasses(pattern_info.class_name, class_info)):
            return True
        
        # Métodos factory que no crean nada identificable
        if (factory_methods and 
            not factory_characteristics.get('product_types', []) and
            pattern_info.pattern_type not in ["incomplete_factory_method", "basic_factory_method"]):
            # Solo es falso positivo si no hay evidencia de creación
            evidences = pattern_info.evidences
            if not evidences.get('product_return', {}).get('found', False):
                return True
        
        # Clase que parece más un utility que un factory
        if self._looks_like_utility_class(class_info):
            return True
        
        return False
    
    def _has_concrete_subclasses(self, class_name: str, class_info: Dict[str, Any]) -> bool:
        """Verifica si una clase abstracta tiene subclases concretas"""
        # Esta verificación se haría con información adicional del análisis
        # Por ahora, asumimos que si tiene métodos abstractos, debería tener subclases
        factory_characteristics = class_info.get('factory_characteristics', {})
        has_abstract = factory_characteristics.get('has_abstract_methods', False)
        
        # Si no tiene métodos abstractos, no necesita subclases
        if not has_abstract:
            return True
        
        # En un análisis completo, verificaríamos las relaciones de herencia
        # Por ahora, damos el beneficio de la duda
        return True
    
    def _looks_like_utility_class(self, class_info: Dict[str, Any]) -> bool:
        """Verifica si la clase parece más una utility class que un factory"""
        methods = class_info.get('methods', [])
        
        # Todos los métodos son estáticos
        static_methods = [m for m in methods if m.get('is_static', False)]
        if len(static_methods) == len(methods) and len(methods) > 3:
            return True
        
        # Muchos métodos públicos sin relación clara con factory
        factory_methods = [m for m in methods if m.get('is_factory_method', False)]
        public_methods = [m for m in methods if m.get('visibility') == 'public']
        
        if len(public_methods) > 5 and len(factory_methods) <= 1:
            return True
        
        return False
    
    def _print_optimization_results(self):
        print(f"Optimización completada:")
        print(f"   Falsos positivos removidos: {len(self.optimized_data.false_positives_removed)}")
        if self.optimized_data.false_positives_removed:
            print(f"      Clases: {', '.join(self.optimized_data.false_positives_removed)}")
        
        if self.optimized_data.confidence_adjustments:
            print(f"   Ajustes de confianza aplicados: {len(self.optimized_data.confidence_adjustments)}")
            
            # Mostrar los ajustes más significativos
            significant_adjustments = {k: v for k, v in self.optimized_data.confidence_adjustments.items() 
                                     if abs(v) > 0.10}
            if significant_adjustments:
                print("   Ajustes significativos:")
                for class_name, adjustment in significant_adjustments.items():
                    direction = "↑" if adjustment > 0 else "↓"
                    print(f"      {class_name}: {direction}{abs(adjustment):.2f}")
        
        # Estadísticas de relaciones optimizadas
        total_relations = len(self.optimized_data.relations)
        inheritance_rels = len([r for r in self.optimized_data.relations 
                               if r.relation_type == RelationType.INHERITANCE])
        dependency_rels = len([r for r in self.optimized_data.relations 
                              if r.relation_type == RelationType.DEPENDENCY])
        
        print(f"   Relaciones optimizadas: {total_relations}")
        print(f"      Herencia: {inheritance_rels}, Dependencias: {dependency_rels}")

# =============================================================================
# FASE 6: GENERACIÓN DE CÓDIGO UML - FACTORY METHOD
# =============================================================================

class UMLFactoryCodeGenerator:
    """Generador de código UML a partir del análisis de patrones Factory Method"""
    
    def __init__(self):
        self.templates = {
            'abstract_factory_method': self._template_abstract_factory,
            'concrete_factory_method': self._template_concrete_factory,
            'factory_class': self._template_factory_class,
            'creator_method': self._template_creator_method,
            'product_return': self._template_product_return,
            'conditional_creation': self._template_conditional_creation,
            'incomplete_factory': self._template_incomplete,
            'no_factory': self._template_no_pattern,
            'false_positive': self._template_false_positive
        }
        
        self.relation_styles = {
            RelationType.INHERITANCE: "--|>",
            RelationType.COMPOSITION: "*--",
            RelationType.ASSOCIATION: "--",
            RelationType.DEPENDENCY: "..>",
            RelationType.DECORATOR: "..>",
            RelationType.OBSERVER: "..>"
        }
    
    def generate_uml(self, optimized_data: OptimizedData, ir_data: IntermediateRepresentation = None) -> str:
        """Genera el código UML final para Factory Method"""
        print("\nFASE 6: GENERACION DE CODIGO UML - FACTORY METHOD")
        print("-" * 50)
        
        self._ir_data = ir_data
        
        uml_content = self._generate_header()
        
        # CAMBIO: Filtrar mejor los patrones válidos
        factory_patterns = [p for p in optimized_data.patterns.values() 
                           if p.pattern_type not in ["no_factory", "false_positive"]]
        
        # Debug: Imprimir información de patrones encontrados
        print(f"DEBUG: Patrones encontrados: {len(factory_patterns)}")
        for pattern in factory_patterns:
            print(f"  - {pattern.class_name}: {pattern.pattern_type} (conf: {pattern.confidence:.2f})")
        
        all_involved_classes = set()
        for pattern in factory_patterns:
            all_involved_classes.add(pattern.class_name)
        
        for relation in optimized_data.relations:
            all_involved_classes.add(relation.source)
            all_involved_classes.add(relation.target)
        
        if not factory_patterns:
            uml_content += self._generate_no_patterns_found()
        else:
            # CAMBIO: Generar TODOS los patrones encontrados
            for pattern in factory_patterns:
                print(f"DEBUG: Generando template para {pattern.class_name} ({pattern.pattern_type})")
                template_func = self.templates.get(pattern.pattern_type, self._template_unknown)
                uml_content += template_func(pattern)
            
            # Clases relacionadas que no son parte del patrón
            for class_name in all_involved_classes:
                if class_name not in [p.class_name for p in factory_patterns]:
                    print(f"DEBUG: Generando clase relacionada: {class_name}")
                    uml_content += self._template_related_class(class_name)
            
            uml_content += self._generate_relations(optimized_data.relations)
        
        if optimized_data.false_positives_removed:
            uml_content += self._generate_false_positives_section(optimized_data.false_positives_removed)
        
        uml_content += "\n@enduml"
        
        self._print_generation_results(len(factory_patterns), len(optimized_data.relations))
        return uml_content
    
    def _generate_header(self) -> str:
        """Genera el encabezado del archivo UML"""
        return f"""@startuml
!theme cerulean-outline

title Análisis de Patrones Factory Method
' Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    def _template_abstract_factory(self, pattern: SemanticInfo) -> str:
        """Template para Factory Method abstracto"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        factory_method = self._highlight_factory_method(class_info)
        
        return f"""class {pattern.class_name} <<Creator>> {{
{attributes}
  --
{methods}
  --
{factory_method}
}}

note top of {pattern.class_name}
  <b>Creator Abstracto</b>
  Factory Method Pattern
  Confianza: {int(pattern.confidence * 100)}%
  Evidencias: {', '.join(pattern.evidences[:3]) if hasattr(pattern, 'evidences') else 'N/A'}
end note

"""
    
    def _template_concrete_factory(self, pattern: SemanticInfo) -> str:
        """Template para Factory Method concreto"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        factory_method = self._highlight_factory_method(class_info)
        
        return f"""class {pattern.class_name} <<ConcreteCreator>> {{
{attributes}
  --
{methods}
  --
{factory_method}
}}

note top of {pattern.class_name}
  <b>Creator Concreto</b>
  Implementa Factory Method
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_factory_class(self, pattern: SemanticInfo) -> str:
        """Template para clase Factory"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} <<Factory>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  <b>Factory Class</b>
  Patrón Factory
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_creator_method(self, pattern: SemanticInfo) -> str:
        """Template para método creator"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} <<Creator>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  <b>Método Creator</b>
  Contiene lógica de creación
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_product_return(self, pattern: SemanticInfo) -> str:
        """Template para clase Product"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} <<Product>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  <b>Product Class</b>
  Producto del Factory
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_conditional_creation(self, pattern: SemanticInfo) -> str:
        """Template para creación condicional"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} <<ConditionalFactory>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  <b>Creación Condicional</b>
  Factory con lógica condicional
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_incomplete(self, pattern: SemanticInfo) -> str:
        """Template para Factory incompleto"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} <<Incompleto>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name} #FFE4B5
  <b>Factory Incompleto</b>
  Implementación parcial
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _template_no_pattern(self, pattern: SemanticInfo) -> str:
        """Template para clases sin patrón"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} {{
{attributes}
  --
{methods}
}}

"""
    
    def _template_related_class(self, class_name: str) -> str:
        """Template para clases relacionadas - MEJORADO"""
        class_info = self._get_class_info(class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        # CAMBIO: Asegurar que siempre se muestre contenido
        if not attributes or attributes.strip() == "' Sin atributos":
            attributes = "  ' Sin atributos públicos"
        
        if not methods or methods.strip() == "":
            methods = "  + __init__()\n  ' Otros métodos..."
        
        return f"""class {class_name} {{
{attributes}
  --
{methods}
}}

"""
    
    def _template_false_positive(self, pattern: SemanticInfo) -> str:
        """Template para falsos positivos"""
        return f"""class {pattern.class_name} <<FalsoPositivo>> {{
  ' Detectado como falso positivo
}}

"""
    
    def _template_unknown(self, pattern: SemanticInfo) -> str:
        """Template para patrones desconocidos - MEJORADO"""
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""class {pattern.class_name} <<Factory>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  <b>Patrón Factory Method</b>
  Tipo: {pattern.pattern_type}
  Confianza: {int(pattern.confidence * 100)}%
end note

"""
    
    def _get_class_info(self, class_name: str) -> Dict[str, Any]:
        """Obtiene información de una clase desde la representación intermedia - MEJORADO"""
        if hasattr(self, '_ir_data') and self._ir_data and class_name in self._ir_data.classes:
            return self._ir_data.classes[class_name]
        
        # CAMBIO: Proporcionar información básica si no está disponible
        print(f"DEBUG: No se encontró información para la clase {class_name}")
        return {
            'variables': [{'name': 'data', 'type': 'Any', 'is_private': False}],
            'methods': [{'name': '__init__', 'is_static': False}, {'name': 'process', 'is_static': False}]
        }
    
    def _format_attributes(self, class_info: Dict[str, Any]) -> str:
        """Formatea los atributos para mostrar en UML - MEJORADO"""
        variables = class_info.get('variables', [])
        if not variables:
            return "  ' Sin atributos definidos"
        
        formatted_attrs = []
        
        # Separar variables de clase e instancia
        class_vars = [v for v in variables if v.get('is_class_var', False)]
        instance_vars = [v for v in variables if not v.get('is_class_var', False)]
        
        # Variables de clase (máximo 3)
        for var in class_vars[:3]:
            visibility = "-" if var.get('is_private', False) else "+"
            var_name = var.get('name', 'unknown')
            var_type = var.get('type', 'Object')
            formatted_attrs.append(f"  {visibility} {var_name}: {var_type} {{static}}")
        
        # Variables de instancia (máximo 5)
        for var in instance_vars[:5]:
            visibility = "-" if var.get('is_private', False) else "+"
            var_name = var.get('name', 'unknown')
            var_type = var.get('type', 'Object')
            formatted_attrs.append(f"  {visibility} {var_name}: {var_type}")
        
        if len(variables) > 8:
            formatted_attrs.append(f"  ' ... y {len(variables) - 8} más")
        
        return "\n".join(formatted_attrs) if formatted_attrs else "  ' Sin atributos"
    
    def _format_methods(self, class_info: Dict[str, Any]) -> str:
        """Formatea los métodos para mostrar en UML - MEJORADO"""
        methods = class_info.get('methods', [])
        if not methods:
            return "  + __init__()\n  ' Sin métodos definidos"
        
        formatted_methods = []
        excluded_methods = ['__str__', '__repr__', '__del__', '__hash__', '__eq__']
        factory_methods = ['create', 'make', 'build', 'get_instance', 'factory', 'produce',
                           'crear', 'hacer', 'construir', 'producir', 'new', 'generate']
        
        # Priorizar métodos importantes
        important_methods = []
        regular_methods = []
        
        for method in methods:
            method_name = method.get('name', 'unknown')
            
            if method_name in excluded_methods:
                continue
            
            # Determinar visibilidad
            if method_name.startswith('__') and method_name.endswith('__'):
                visibility = "+"
            elif method_name.startswith('_'):
                visibility = "-"
            else:
                visibility = "+"
            
            # Crear representación del método
            if method.get('is_static', False):
                method_display = f"{method_name}() {{static}}"
            elif method.get('is_classmethod', False):
                method_display = f"{method_name}() {{class}}"
            else:
                method_display = f"{method_name}()"
            
            # Destacar métodos factory
            is_factory_method = any(fm in method_name.lower() for fm in factory_methods)
            if is_factory_method:
                method_display += " <<factory>>"
                important_methods.append(f"  {visibility} {method_display}")
            else:
                regular_methods.append(f"  {visibility} {method_display}")
        
        # Combinar métodos importantes primero
        all_methods = important_methods + regular_methods
        
        # Limitar a 8 métodos máximo
        if len(all_methods) > 8:
            formatted_methods = all_methods[:8]
            formatted_methods.append(f"  ' ... y {len(all_methods) - 8} más")
        else:
            formatted_methods = all_methods
        
        if not formatted_methods:
            return "  + __init__()"
        
        return "\n".join(formatted_methods)
    
    def _highlight_factory_method(self, class_info: Dict[str, Any]) -> str:
        """Resalta el método factory principal"""
        methods = class_info.get('methods', [])
        factory_methods = ['create', 'make', 'build', 'get_instance', 'factory', 'produce',
                           'crear', 'hacer', 'construir', 'producir', 'new', 'generate']
        
        for method in methods:
            method_name = method.get('name', 'unknown')
            if any(fm in method_name.lower() for fm in factory_methods):
                return f"  <<abstract>> + {method_name}(): Product"
        
        return "  <<abstract>> + create_product(): Product"
    
    def _generate_relations(self, relations: List[ClassRelation]) -> str:
        """Genera las relaciones UML - MEJORADO"""
        if not relations:
            return "\n' Sin relaciones definidas\n"
        
        relations_uml = "\n' === Relaciones Factory Method ===\n"
        
        for i, relation in enumerate(relations):
            style = self._get_relation_style(relation.relation_type)
            
            # Añadir etiquetas específicas para Factory Method
            label = ""
            if relation.relation_type == RelationType.INHERITANCE:
                label = " : extends"
            elif relation.relation_type == RelationType.DEPENDENCY:
                label = " : creates"
            elif relation.relation_type == RelationType.ASSOCIATION:
                label = " : uses"
            elif relation.relation_type == RelationType.COMPOSITION:
                label = " : contains"
            
            relations_uml += f"{relation.source} {style} {relation.target}{label}\n"
        
        return relations_uml + "\n"
    
    def _get_relation_style(self, relation_type: RelationType) -> str:
        """Obtiene el estilo UML para el tipo de relación"""
        return self.relation_styles.get(relation_type, "-->")
    
    def _generate_no_patterns_found(self) -> str:
        """Genera nota cuando no se encuentran patrones"""
        return """note as NoPatterns #FFE4E1
  <b>No se encontraron patrones Factory Method</b>
  en el código analizado.
end note

"""
    
    def _generate_false_positives_section(self, false_positives: List[str]) -> str:
        """Genera sección de falsos positivos - MEJORADO"""
        fp_list = ', '.join(false_positives[:5])  # Máximo 5 nombres
        if len(false_positives) > 5:
            fp_list += f" (+{len(false_positives) - 5} más)"
            
        return f"""
note as FalsePositives #FFFACD
  <b>Falsos Positivos Detectados: {len(false_positives)}</b>
  
  Clases descartadas:
  {fp_list}
  
  Estos elementos fueron identificados inicialmente
  como patrones pero descartados tras el análisis.
end note

"""
    
    def _print_generation_results(self, patterns_count: int, relations_count: int):
        print(f"Código UML Factory Method generado:")
        print(f"   ✓ Patrones detectados: {patterns_count}")
        print(f"   ✓ Relaciones incluidas: {relations_count}")
        print(f"   ✓ Tipos de patrón soportados:")
        print(f"     • Abstract Factory Method")
        print(f"     • Concrete Factory Method") 
        print(f"     • Factory Class")
        print(f"     • Creator Method")
        print(f"     • Product Return")
        print(f"     • Conditional Creation")
        print(f"   ✓ Incluye notas explicativas y stereotypes")
        print(f"   ✓ Información detallada de clases y métodos")


#class UMLFactoryCodeGenerator:
#    """Generador de código UML a partir del análisis de patrones Factory Method"""
#    
#    def __init__(self):
#        self.templates = {
#            'abstract_factory_method': self._template_abstract_factory,
#            'concrete_factory_method': self._template_concrete_factory,
#            'factory_class': self._template_factory_class,
#            'creator_method': self._template_creator_method,
#            'product_return': self._template_product_return,
#            'conditional_creation': self._template_conditional_creation,
#            'incomplete_factory': self._template_incomplete,
#            'no_factory': self._template_no_pattern,
#            'false_positive': self._template_false_positive
#        }
#        
#        self.relation_styles = {
#            RelationType.INHERITANCE: "--|>",
#            RelationType.COMPOSITION: "*--",
#            RelationType.ASSOCIATION: "--",
#            RelationType.DEPENDENCY: "..>",
#            RelationType.DECORATOR: "..>",
#            RelationType.OBSERVER: "..>"
#        }
#    
#    def generate_uml(self, optimized_data: OptimizedData, ir_data: IntermediateRepresentation = None) -> str:
#        """Genera el código UML final para Factory Method"""
#        print("\nFASE 6: GENERACION DE CODIGO UML - FACTORY METHOD")
#        print("-" * 50)
#        
#        self._ir_data = ir_data
#        
#        uml_content = self._generate_header()
#        
#        factory_patterns = [p for p in optimized_data.patterns.values() 
#                           if p.pattern_type != "no_factory" and p.pattern_type != "false_positive"]
#        
#        all_involved_classes = set()
#        for pattern in factory_patterns:
#            all_involved_classes.add(pattern.class_name)
#        
#        for relation in optimized_data.relations:
#            all_involved_classes.add(relation.source)
#            all_involved_classes.add(relation.target)
#        
#        if not factory_patterns:
#            uml_content += self._generate_no_patterns_found()
#        else:
#            # Separar patrones por categorías
#            creators = [p for p in factory_patterns if 'creator' in p.pattern_type.lower()]
#            factories = [p for p in factory_patterns if 'factory' in p.pattern_type.lower()]
#            products = [p for p in factory_patterns if 'product' in p.pattern_type.lower()]
#            others = [p for p in factory_patterns if p not in creators + factories + products]
#            
#            # Generar creators primero
#            for pattern in creators:
#                template_func = self.templates.get(pattern.pattern_type, self._template_unknown)
#                uml_content += template_func(pattern)
#            
#            # Luego factories
#            for pattern in factories:
#                template_func = self.templates.get(pattern.pattern_type, self._template_unknown)
#                uml_content += template_func(pattern)
#            
#            # Después products
#            for pattern in products:
#                template_func = self.templates.get(pattern.pattern_type, self._template_unknown)
#                uml_content += template_func(pattern)
#            
#            # Finalmente otros
#            for pattern in others:
#                template_func = self.templates.get(pattern.pattern_type, self._template_unknown)
#                uml_content += template_func(pattern)
#            
#            # Clases relacionadas que no son parte del patrón
#            for class_name in all_involved_classes:
#                if class_name not in [p.class_name for p in factory_patterns]:
#                    uml_content += self._template_related_class(class_name)
#            
#            uml_content += self._generate_relations(optimized_data.relations)
#        
#        if optimized_data.false_positives_removed:
#            uml_content += self._generate_false_positives_section(optimized_data.false_positives_removed)
#        
#        uml_content += "\n@enduml"
#        
#        self._print_generation_results(len(factory_patterns), len(optimized_data.relations))
#        return uml_content
#    
#    def _generate_header(self) -> str:
#        """Genera el encabezado del archivo UML"""
#        return f"""@startuml
#!theme cerulean-outline
#
#title Análisis de Patrones Factory Method
#' Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
#"""
#    
#    def _template_abstract_factory(self, pattern: SemanticInfo) -> str:
#        """Template para Factory Method abstracto"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<Creator>> {{
#{attributes}
#  --
#{methods}
#  {self._highlight_factory_method(class_info)}
#}}
#
#note top of {pattern.class_name}
#  Creator Abstracto
#  Factory Method Pattern
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_concrete_factory(self, pattern: SemanticInfo) -> str:
#        """Template para Factory Method concreto"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<ConcreteCreator>> {{
#{attributes}
#  --
#{methods}
#  {self._highlight_factory_method(class_info)}
#}}
#
#note top of {pattern.class_name}
#  Creator Concreto
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_factory_class(self, pattern: SemanticInfo) -> str:
#        """Template para clase Factory"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<Factory>> {{
#{attributes}
#  --
#{methods}
#}}
#
#note top of {pattern.class_name}
#  Factory Class
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_creator_method(self, pattern: SemanticInfo) -> str:
#        """Template para método creator"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<Creator>> {{
#{attributes}
#  --
#{methods}
#}}
#
#note top of {pattern.class_name}
#  Método Creator
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_product_return(self, pattern: SemanticInfo) -> str:
#        """Template para clase Product"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<Product>> {{
#{attributes}
#  --
#{methods}
#}}
#
#note top of {pattern.class_name}
#  Product Class
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_conditional_creation(self, pattern: SemanticInfo) -> str:
#        """Template para creación condicional"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<ConditionalFactory>> {{
#{attributes}
#  --
#{methods}
#}}
#
#note top of {pattern.class_name}
#  Creación Condicional
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_incomplete(self, pattern: SemanticInfo) -> str:
#        """Template para Factory incompleto"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} <<Incompleto>> {{
#{attributes}
#  --
#{methods}
#}}
#
#note top of {pattern.class_name}
#  Factory incompleto
#  Confianza: {int(pattern.confidence * 100)}%
#end note
#
#"""
#    
#    def _template_no_pattern(self, pattern: SemanticInfo) -> str:
#        """Template para clases sin patrón"""
#        class_info = self._get_class_info(pattern.class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {pattern.class_name} {{
#{attributes}
#  --
#{methods}
#}}
#
#"""
#    
#    def _template_related_class(self, class_name: str) -> str:
#        """Template para clases relacionadas"""
#        class_info = self._get_class_info(class_name)
#        attributes = self._format_attributes(class_info)
#        methods = self._format_methods(class_info)
#        
#        return f"""
#class {class_name} {{
#{attributes}
#  --
#{methods}
#}}
#
#"""
#    
#    def _template_false_positive(self, pattern: SemanticInfo) -> str:
#        """Template para falsos positivos"""
#        return f"""
#class {pattern.class_name} <<FalsoPositivo>> {{
#}}
#
#"""
#    
#    def _template_unknown(self, pattern: SemanticInfo) -> str:
#        """Template para patrones desconocidos"""
#        return f"""
#class {pattern.class_name} {{
#}}
#
#"""
#    
#    def _get_class_info(self, class_name: str) -> Dict[str, Any]:
#        """Obtiene información de una clase desde la representación intermedia"""
#        if hasattr(self, '_ir_data') and self._ir_data and class_name in self._ir_data.classes:
#            return self._ir_data.classes[class_name]
#        return {'variables': [], 'methods': []}
#    
#    def _format_attributes(self, class_info: Dict[str, Any]) -> str:
#        """Formatea los atributos para mostrar en UML"""
#        variables = class_info.get('variables', [])
#        if not variables:
#            return "  ' Sin atributos"
#        
#        formatted_attrs = []
#        
#        class_vars = [v for v in variables if v.get('is_class_var', False)]
#        instance_vars = [v for v in variables if not v.get('is_class_var', False)]
#        
#        for var in class_vars[:3]:
#            visibility = "-" if var.get('is_private', False) else "+"
#            var_name = var.get('name', 'unknown')
#            var_type = var.get('type', 'Object')
#            formatted_attrs.append(f"  {visibility} {var_name}: {var_type} {{static}}")
#        
#        for var in instance_vars[:5]:
#            visibility = "-" if var.get('is_private', False) else "+"
#            var_name = var.get('name', 'unknown')
#            var_type = var.get('type', 'Object')
#            formatted_attrs.append(f"  {visibility} {var_name}: {var_type}")
#        
#        return "\n".join(formatted_attrs) if formatted_attrs else "  ' Sin atributos"
#    
#    def _format_methods(self, class_info: Dict[str, Any]) -> str:
#        """Formatea los métodos para mostrar en UML"""
#        methods = class_info.get('methods', [])
#        if not methods:
#            return "  + __init__()"
#        
#        formatted_methods = []
#        excluded_methods = ['__str__', '__repr__', '__del__', '__hash__', '__eq__']
#        factory_methods = ['create', 'make', 'build', 'get_instance', 'factory', 'produce',
#                           'crear', 'hacer', 'construir', 'producir',]
#        
#        for method in methods:
#            method_name = method.get('name', 'unknown')
#            
#            if method_name in excluded_methods:
#                continue
#            
#            if method_name.startswith('__') and method_name.endswith('__'):
#                visibility = "+"
#            elif method_name.startswith('_'):
#                visibility = "-"
#            else:
#                visibility = "+"
#            
#            # Destacar métodos factory
#            is_factory_method = any(fm in method_name.lower() for fm in factory_methods)
#            
#            if method.get('is_static', False):
#                method_display = f"{method_name}() {{static}}"
#            elif method.get('is_classmethod', False):
#                method_display = f"{method_name}() {{class}}"
#            else:
#                method_display = f"{method_name}()"
#            
#            if is_factory_method:
#                method_display += " <<factory>>"
#            
#            formatted_methods.append(f"  {visibility} {method_display}")
#            
#            if len(formatted_methods) >= 6:
#                break
#        
#        if not formatted_methods:
#            return "  + __init__()"
#        
#        return "\n".join(formatted_methods)
#    
#    def _highlight_factory_method(self, class_info: Dict[str, Any]) -> str:
#        """Resalta el método factory principal"""
#        methods = class_info.get('methods', [])
#        factory_methods = ['create', 'make', 'build', 'get_instance', 'factory', 'produce',
#                           'crear', 'hacer', 'construir', 'producir']
#        
#        for method in methods:
#            method_name = method.get('name', 'unknown', 'nombre', 'desconocido')
#            if any(fm in method_name.lower() for fm in factory_methods):
#                return f"  <<abstract>> + {method_name}(): Product"
#        
#        return "  <<abstract>> + create_product(): Product"
#    
#    def _generate_relations(self, relations: List[ClassRelation]) -> str:
#        """Genera las relaciones UML"""
#        if not relations:
#            return ""
#        
#        relations_uml = "\n' Relaciones Factory Method\n"
#        
#        for relation in relations:
#            style = self._get_relation_style(relation.relation_type)
#            
#            # Añadir etiquetas específicas para Factory Method
#            label = ""
#            if relation.relation_type == RelationType.INHERITANCE:
#                label = " : extends"
#            elif relation.relation_type == RelationType.DEPENDENCY:
#                label = " : creates"
#            elif relation.relation_type == RelationType.ASSOCIATION:
#                label = " : uses"
#            
#            relations_uml += f"{relation.source} {style} {relation.target}{label}\n"
#        
#        return relations_uml + "\n"
#    
#    def _get_relation_style(self, relation_type: RelationType) -> str:
#        """Obtiene el estilo UML para el tipo de relación"""
#        return self.relation_styles.get(relation_type, "-->")
#    
#    def _generate_no_patterns_found(self) -> str:
#        """Genera nota cuando no se encuentran patrones"""
#        return """
#note as NoPatterns
#  No se encontraron patrones Factory Method
#  en el código analizado.
#end note
#
#"""
#    
#    def _generate_false_positives_section(self, false_positives: List[str]) -> str:
#        """Genera sección de falsos positivos"""
#        return f"""
#note as FalsePositives
#  Falsos Positivos: {len(false_positives)}
#  Clases: {', '.join(false_positives)}
#end note
#
#"""
#    
#    def _print_generation_results(self, patterns_count: int, relations_count: int):
#        print(f"Código UML Factory Method generado:")
#        print(f"   Patrones detectados: {patterns_count}")
#        print(f"   Relaciones incluidas: {relations_count}")
#        print(f"   Tipos de patrón soportados:")
#        print(f"     - Abstract Factory Method")
#        print(f"     - Concrete Factory Method")
#        print(f"     - Factory Class")
#        print(f"     - Creator Method")
#        print(f"     - Product Return")
#        print(f"     - Conditional Creation")

# =============================================================================
# COMPILADOR PRINCIPAL - FACTORY METHOD
# =============================================================================

class FactoryMethodCompiler:
    """Compilador principal que coordina todas las fases del análisis de Factory Method"""
    
    def __init__(self):
        self.lexer = LexicalAnalyzer()
        self.parser = SyntaxAnalyzer()
        self.semantic = SemanticAnalyzer()
        self.ir_generator = IntermediateCodeGenerator()
        self.optimizer = Optimizer()
        self.code_generator = UMLFactoryCodeGenerator()
    
    def compile(self, input_file: str, output_file: str) -> bool:
        """Ejecuta el proceso completo de compilación para Factory Method"""
        print("\nANALISIS DE PATRONES FACTORY METHOD")
        print("=" * 65)
        #print(f"Archivo de entrada: {input_file}")
        #print(f"Archivo de salida: {output_file}")
        print(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)
        
        try:
            # Leer archivo fuente
            with open(input_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            print(f"Líneas de código: {len(source_code.splitlines())}")
            print(f"Caracteres: {len(source_code)}")
            print("-" * 65)
            
            # FASE 1: Análisis Léxico
            tokens = self.lexer.tokenize(source_code)
            if not tokens:
                print("WARNING: No se encontraron tokens relacionados con Factory Method")
            
            # FASE 2: Análisis Sintáctico
            ast_nodes, relations = self.parser.parse(source_code, tokens)
            
            # FASE 3: Análisis Semántico
            semantic_info = self.semantic.analyze(ast_nodes, tokens)
            
            # FASE 4: Generación de Representación Intermedia
            ir = self.ir_generator.generate(ast_nodes, semantic_info, relations)
            
            # FASE 5: Optimización
            optimized_data = self.optimizer.optimize(ir)
            
            # FASE 6: Generación de Código UML
            uml_code = self.code_generator.generate_uml(optimized_data, ir)
            
            # Escribir archivo de salida
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(uml_code)
            
            # Resumen de resultados
            self._print_compilation_summary(optimized_data, output_file)
            
            print("=" * 65)
            return True
            
        except FileNotFoundError:
            print(f"ERROR: No se encontró el archivo {input_file}")
            return False
        except PermissionError:
            print(f"ERROR: Sin permisos para leer/escribir archivos")
            return False
        except Exception as e:
            print(f"ERROR INESPERADO: {e}")
            print("Revise el formato del archivo de entrada y los permisos")
            return False
    
    def _print_compilation_summary(self, optimized_data: OptimizedData, output_file: str):
        """Imprime resumen de la compilación"""
        print("\n" + "=" * 65)
        print("COMPILACION COMPLETADA - RESUMEN")
        print("=" * 65)
        
        # Estadísticas de patrones
        factory_patterns = [p for p in optimized_data.patterns.values() 
                           if p.pattern_type not in ["no_factory", "false_positive"]]
        
        print(f"Diagrama UML generado: {output_file}")
        print("-" * 65)
        
        print("ESTADÍSTICAS DE ANÁLISIS:")
        print(f"  • Patrones Factory Method detectados: {len(factory_patterns)}")
        print(f"  • Relaciones entre clases: {len(optimized_data.relations)}")
        print(f"  • Falsos positivos eliminados: {len(optimized_data.false_positives_removed)}")
        
        if factory_patterns:
            print("\nTIPOS DE FACTORY METHOD ENCONTRADOS:")
            pattern_counts = {}
            for pattern in factory_patterns:
                pattern_type = pattern.pattern_type
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            for pattern_type, count in sorted(pattern_counts.items()):
                pattern_name = self._get_pattern_display_name(pattern_type)
                print(f"  • {pattern_name}: {count}")
        
        # Estadísticas de confianza
        if factory_patterns:
            confidences = [p.confidence for p in factory_patterns]
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            print(f"\nCONFIANZA DEL ANÁLISIS:")
            print(f"  • Confianza promedio: {avg_confidence:.1%}")
            print(f"  • Confianza máxima: {max_confidence:.1%}")
            print(f"  • Confianza mínima: {min_confidence:.1%}")
        
        print("-" * 65)
    
    def _get_pattern_display_name(self, pattern_type: str) -> str:
        """Convierte el tipo de patrón a nombre legible"""
        display_names = {
            'abstract_factory_method': 'Factory Method Abstracto',
            'concrete_factory_method': 'Factory Method Concreto',
            'factory_class': 'Clase Factory',
            'creator_method': 'Método Creator',
            'product_return': 'Clase Product',
            'conditional_creation': 'Creación Condicional',
            'incomplete_factory': 'Factory Incompleto'
        }
        return display_names.get(pattern_type, pattern_type.replace('_', ' ').title())