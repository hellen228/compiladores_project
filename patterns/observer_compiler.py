import ast
import re
import subprocess
from typing import List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from data_structure import ASTNode, ClassRelation, RelationType, IntermediateRepresentation
from data_structure import OptimizedData, SemanticInfo

# =============================================================================
# DEFINICIÓN DE ESTRUCTURAS DE DATOS
# =============================================================================

class TokenType(Enum):
    # Tokens para Classic Observer
    OBSERVER_LIST = "observer_list"
    ATTACH_METHOD = "attach_method"
    DETACH_METHOD = "detach_method"
    NOTIFY_METHOD = "notify_method"
    UPDATE_METHOD = "update_method"
    
    # Tokens para Event-Driven Observer
    EVENT_HANDLER = "event_handler"
    EVENT_LISTENER = "event_listener"
    CALLBACK_REF = "callback_reference"
    ON_METHOD = "on_method"
    EMIT_METHOD = "emit_method"
    
    # Tokens para Publisher-Subscriber
    SUBSCRIBE_METHOD = "subscribe_method"
    UNSUBSCRIBE_METHOD = "unsubscribe_method"
    PUBLISH_METHOD = "publish_method"
    TOPIC_REF = "topic_reference"
    CHANNEL_REF = "channel_reference"
    
    # Tokens generales
    CLASS_DEF = "class_definition"
    LOOP_NOTIFY = "loop_notification"

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
        
        # Patrones específicos para Observer pattern
        self.patterns = {
            # Classic Observer patterns
            TokenType.OBSERVER_LIST: [
                r'self\._observers\b', r'self\.observers\b',
                r'self\._listeners\b', r'self\.listeners\b',
                r'observers\s*=\s*\[\]', r'listeners\s*=\s*\[\]'
            ],
            TokenType.ATTACH_METHOD: [
                r'def\s+attach\s*\(', r'def\s+add_observer\s*\(',
                r'def\s+register\s*\(', r'def\s+add_listener\s*\('
            ],
            TokenType.DETACH_METHOD: [
                r'def\s+detach\s*\(', r'def\s+remove_observer\s*\(',
                r'def\s+unregister\s*\(', r'def\s+remove_listener\s*\('
            ],
            TokenType.NOTIFY_METHOD: [
                r'def\s+notify\s*\(', r'def\s+notify_all\s*\(',
                r'def\s+notify_observers\s*\('
            ],
            TokenType.UPDATE_METHOD: [
                r'def\s+update\s*\(', r'def\s+on_update\s*\(',
                r'def\s+handle_update\s*\('
            ],
            
            # Event-Driven Observer patterns
            TokenType.EVENT_HANDLER: [
                r'def\s+on_\w+\s*\(', r'def\s+handle_\w+\s*\(',
                r'event_handler', r'@event_handler'
            ],
            TokenType.EVENT_LISTENER: [
                r'add_event_listener', r'addEventListener',
                r'bind_event', r'on_event'
            ],
            TokenType.CALLBACK_REF: [
                r'callback\s*=', r'self\.callback\b',
                r'handler\s*=', r'self\.handler\b'
            ],
            TokenType.ON_METHOD: [
                r'def\s+on\s*\(', r'\.on\s*\(',
                r'def\s+bind\s*\('
            ],
            TokenType.EMIT_METHOD: [
                r'def\s+emit\s*\(', r'\.emit\s*\(',
                r'def\s+trigger\s*\(', r'\.trigger\s*\('
            ],
            
            # Publisher-Subscriber patterns
            TokenType.SUBSCRIBE_METHOD: [
                r'def\s+subscribe\s*\(', r'\.subscribe\s*\(',
                r'def\s+sub\s*\('
            ],
            TokenType.UNSUBSCRIBE_METHOD: [
                r'def\s+unsubscribe\s*\(', r'\.unsubscribe\s*\(',
                r'def\s+unsub\s*\('
            ],
            TokenType.PUBLISH_METHOD: [
                r'def\s+publish\s*\(', r'\.publish\s*\(',
                r'def\s+pub\s*\('
            ],
            TokenType.TOPIC_REF: [
                r'topic\s*=', r'self\.topic\b',
                r'topics\s*=', r'self\.topics\b'
            ],
            TokenType.CHANNEL_REF: [
                r'channel\s*=', r'self\.channel\b',
                r'channels\s*=', r'self\.channels\b'
            ],
            
            # General patterns
            TokenType.LOOP_NOTIFY: [
                r'for\s+\w+\s+in\s+.*observers',
                r'for\s+\w+\s+in\s+.*listeners',
                r'for\s+\w+\s+in\s+.*subscribers'
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
            # Classic Observer
            TokenType.OBSERVER_LIST: {
                '_observers': 0.95, 'observers': 0.90, '_listeners': 0.85, 'listeners': 0.80
            },
            TokenType.ATTACH_METHOD: {
                'attach': 0.95, 'add_observer': 0.90, 'register': 0.75, 'add_listener': 0.85
            },
            TokenType.DETACH_METHOD: {
                'detach': 0.95, 'remove_observer': 0.90, 'unregister': 0.75, 'remove_listener': 0.85
            },
            TokenType.NOTIFY_METHOD: {
                'notify': 0.95, 'notify_all': 0.90, 'notify_observers': 0.95
            },
            TokenType.UPDATE_METHOD: {
                'update': 0.85, 'on_update': 0.90, 'handle_update': 0.85
            },
            
            # Event-Driven Observer
            TokenType.EVENT_HANDLER: {
                'on_': 0.80, 'handle_': 0.85, 'event_handler': 0.95
            },
            TokenType.EMIT_METHOD: {
                'emit': 0.95, 'trigger': 0.90
            },
            
            # Publisher-Subscriber
            TokenType.SUBSCRIBE_METHOD: {
                'subscribe': 0.95, 'sub': 0.85
            },
            TokenType.PUBLISH_METHOD: {
                'publish': 0.95, 'pub': 0.85
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
            
            self.ast_nodes = self._build_ast(tree, tokens, classes)
            self.relations = self._detect_relations(tree, classes)
            
            print(f"Clases encontradas: {len(classes)}")
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
                observer_attrs = self._find_observer_attributes(node)
                event_attrs = self._find_event_attributes(node)
                pubsub_attrs = self._find_pubsub_attributes(node)
                constructor_info = self._analyze_constructor(node)
                
                classes[node.name] = {
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [self._get_base_name(base) for base in node.bases],
                    'methods': [method.name for method in node.body 
                               if isinstance(method, ast.FunctionDef)],
                    'node': node,
                    'observer_attributes': observer_attrs,
                    'event_attributes': event_attrs,
                    'pubsub_attributes': pubsub_attrs,
                    'constructor_info': constructor_info
                }
        
        return classes
    
    def _analyze_constructor(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        analysis = {
            'compositions': [],
            'associations': [],
            'observer_registrations': []
        }
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                params = [arg.arg for arg in item.args.args if arg.arg != 'self']
                
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                
                                attr_name = target.attr
                                
                                # Composición: self.attr = Class()
                                if isinstance(stmt.value, ast.Call):
                                    if isinstance(stmt.value.func, ast.Name):
                                        class_created = stmt.value.func.id
                                        analysis['compositions'].append({
                                            'attribute': attr_name,
                                            'target_class': class_created,
                                            'line': stmt.lineno
                                        })
                                
                                # Asociación: self.attr = param
                                elif isinstance(stmt.value, ast.Name) and stmt.value.id in params:
                                    analysis['associations'].append({
                                        'attribute': attr_name,
                                        'parameter': stmt.value.id,
                                        'line': stmt.lineno
                                    })
                    
                    # Registro Observer: subject.attach(self)
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if (isinstance(stmt.value.func, ast.Attribute) and
                            stmt.value.func.attr in ['attach', 'add_observer', 'register']):
                            
                            for arg in stmt.value.args:
                                if isinstance(arg, ast.Name) and arg.id == 'self':
                                    if isinstance(stmt.value.func.value, ast.Attribute):
                                        subject_attr = stmt.value.func.value.attr
                                        analysis['observer_registrations'].append({
                                            'subject_attribute': subject_attr,
                                            'method': stmt.value.func.attr,
                                            'line': stmt.lineno
                                        })
                break
        
        return analysis
    
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
        
        # Composición
        for class_name, class_info in classes.items():
            constructor_info = class_info.get('constructor_info', {})
            
            for composition in constructor_info.get('compositions', []):
                target_class = composition['target_class']
                if target_class in classes:
                    relations.append(ClassRelation(
                        source=class_name,
                        target=target_class,
                        relation_type=RelationType.COMPOSITION,
                        description=f"{class_name} compone {target_class}",
                        line=composition['line'],
                        confidence=0.90
                    ))
        
        # Asociación
        for class_name, class_info in classes.items():
            constructor_info = class_info.get('constructor_info', {})
            
            for association in constructor_info.get('associations', []):
                param_name = association['parameter']
                
                for potential_class in classes:
                    if (potential_class.lower() in param_name.lower() or
                        param_name.lower() in potential_class.lower()):
                        relations.append(ClassRelation(
                            source=class_name,
                            target=potential_class,
                            relation_type=RelationType.ASSOCIATION,
                            description=f"{class_name} usa {potential_class}",
                            line=association['line'],
                            confidence=0.75
                        ))
                        break
        
        # Registro Observer
        for class_name, class_info in classes.items():
            constructor_info = class_info.get('constructor_info', {})
            
            for registration in constructor_info.get('observer_registrations', []):
                subject_attr = registration['subject_attribute']
                
                for potential_subject in classes:
                    if (potential_subject.lower() in subject_attr.lower() or
                        subject_attr.lower() in potential_subject.lower()):
                        relations.append(ClassRelation(
                            source=class_name,
                            target=potential_subject,
                            relation_type=RelationType.OBSERVER,
                            description=f"{class_name} observa {potential_subject}",
                            line=registration['line'],
                            confidence=0.85
                        ))
                        break
        
        # Dependencias por uso de métodos
        relations.extend(self._detect_method_dependencies(tree, classes))
        
        return relations
    
    def _detect_method_dependencies(self, tree: ast.AST, classes: Dict[str, Dict[str, Any]]) -> List[ClassRelation]:
        dependencies = []
        
        for class_name, class_info in classes.items():
            class_node = class_info['node']
            
            for method in class_node.body:
                if isinstance(method, ast.FunctionDef) and method.name != '__init__':
                    
                    for stmt in ast.walk(method):
                        if isinstance(stmt, ast.Call):
                            if isinstance(stmt.func, ast.Attribute):
                                if isinstance(stmt.func.value, ast.Attribute):
                                    attr_name = stmt.func.value.attr
                                    
                                    for potential_class in classes:
                                        if (potential_class.lower() in attr_name.lower() or
                                            attr_name.lower() in potential_class.lower()):
                                            
                                            existing = any(
                                                r.source == class_name and r.target == potential_class
                                                for r in dependencies
                                            )
                                            
                                            if not existing:
                                                dependencies.append(ClassRelation(
                                                    source=class_name,
                                                    target=potential_class,
                                                    relation_type=RelationType.DEPENDENCY,
                                                    description=f"{class_name} depende de {potential_class}",
                                                    line=method.lineno,
                                                    confidence=0.70
                                                ))
                                            break
        
        return dependencies
    
    def _find_observer_attributes(self, class_node: ast.ClassDef) -> List[str]:
        observer_attrs = []
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for node in ast.walk(item):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                attr_name = target.attr
                                if any(keyword in attr_name.lower() for keyword in 
                                      ['observer', 'listener', 'watcher']):
                                    observer_attrs.append(attr_name)
        
        return observer_attrs
    
    def _find_event_attributes(self, class_node: ast.ClassDef) -> List[str]:
        event_attrs = []
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for node in ast.walk(item):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                attr_name = target.attr
                                if any(keyword in attr_name.lower() for keyword in 
                                      ['event', 'callback', 'handler']):
                                    event_attrs.append(attr_name)
        
        return event_attrs
    
    def _find_pubsub_attributes(self, class_node: ast.ClassDef) -> List[str]:
        pubsub_attrs = []
        
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for node in ast.walk(item):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (isinstance(target, ast.Attribute) and 
                                isinstance(target.value, ast.Name) and 
                                target.value.id == 'self'):
                                attr_name = target.attr
                                if any(keyword in attr_name.lower() for keyword in 
                                      ['topic', 'channel', 'subscriber', 'subscription']):
                                    pubsub_attrs.append(attr_name)
        
        return pubsub_attrs
    
    def _get_base_name(self, base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else base.attr
        return str(base)
    
    def _build_ast(self, tree: ast.AST, tokens: List[Token], classes: Dict[str, Dict[str, Any]]) -> List[ASTNode]:
        nodes = []
        
        for class_name, class_info in classes.items():
            class_node = class_info['node']
            
            ast_node = ASTNode(
                node_type="class",
                name=class_name,
                line=class_info['line'],
                metadata={
                    'bases': class_info['bases'],
                    'methods': class_info['methods'],
                    'observer_attributes': class_info['observer_attributes'],
                    'event_attributes': class_info['event_attributes'],
                    'pubsub_attributes': class_info['pubsub_attributes'],
                    'constructor_info': class_info['constructor_info']
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
                            'has_loop_notify': self._check_loop_notification(item),
                            'calls_update': self._check_update_calls(item)
                        }
                    )
                    ast_node.children.append(method_node)
            
            nodes.append(ast_node)
        
        return nodes
    
    def _check_loop_notification(self, method_node: ast.FunctionDef) -> bool:
        for node in ast.walk(method_node):
            if isinstance(node, ast.For):
                if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Attribute):
                    if any(keyword in str(node.iter.attr).lower() for keyword in 
                          ['observer', 'listener', 'subscriber']):
                        return True
        return False
    
    def _check_update_calls(self, method_node: ast.FunctionDef) -> bool:
        for node in ast.walk(method_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['update', 'on_update', 'handle_update']:
                        return True
        return False

# =============================================================================
# FASE 3: ANALIZADOR SEMÁNTICO
# =============================================================================

class SemanticAnalyzer:
    def __init__(self):
        self.semantic_errors = []
        self.pattern_info = {}
    
    def analyze(self, ast_nodes: List[ASTNode], tokens: List[Token]) -> Dict[str, SemanticInfo]:
        print("Iniciando análisis semántico...")
        
        results = {}
        
        for class_node in [node for node in ast_nodes if node.node_type == "class"]:
            analysis = self._analyze_class(class_node, tokens)
            results[class_node.name] = analysis
        
        patterns_found = len([p for p in results.values() 
                             if p.pattern_type not in ["no_observer", "observer_client"]])
        print(f"Patrones encontrados: {patterns_found}")
        
        return results
    
    def _analyze_class(self, class_node: ASTNode, tokens: List[Token]) -> SemanticInfo:
        evidences = {}
        
        # Verificar patrones Classic Observer
        classic_evidences = self._check_classic_observer(class_node)
        evidences.update(classic_evidences)
        
        # Verificar patrones Event-Driven Observer  
        event_evidences = self._check_event_driven_observer(class_node)
        evidences.update(event_evidences)
        
        # Verificar patrones Publisher-Subscriber
        pubsub_evidences = self._check_publisher_subscriber(class_node)
        evidences.update(pubsub_evidences)
        
        # Verificar si es cliente de observers
        client_evidences = self._check_observer_client(class_node)
        evidences.update(client_evidences)
        
        pattern_type = self._determine_pattern_type(evidences)
        confidence = self._calculate_confidence(evidences, pattern_type)
        violations = self._check_violations(class_node, evidences, pattern_type)
        
        return SemanticInfo(
            class_name=class_node.name,
            pattern_type=pattern_type,
            confidence=confidence,
            evidences=evidences,
            violations=violations
        )
    
    def _check_observer_client(self, class_node: ASTNode) -> Dict[str, Any]:
        evidences = {}
        constructor_info = class_node.metadata.get('constructor_info', {})
        
        compositions = constructor_info.get('compositions', [])
        observer_compositions = []
        
        for comp in compositions:
            target_class = comp['target_class']
            if any(keyword in target_class.lower() for keyword in 
                  ['agency', 'station', 'subject', 'observable', 'event', 'broker']):
                observer_compositions.append(comp)
        
        if observer_compositions:
            evidences['is_observer_client'] = {
                'found': True,
                'compositions': observer_compositions,
                'confidence': 0.80
            }
        
        return evidences
    
    def _check_classic_observer(self, class_node: ASTNode) -> Dict[str, Any]:
        evidences = {}
        
        # Lista de observadores
        observer_attrs = class_node.metadata.get('observer_attributes', [])
        if observer_attrs:
            evidences['has_observer_list'] = {
                'found': True,
                'attributes': observer_attrs,
                'confidence': 0.90
            }
        
        # Métodos attach/detach/notify
        methods = class_node.metadata.get('methods', [])
        
        attach_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                         ['attach', 'add_observer', 'register', 'add_listener'])]
        if attach_methods:
            evidences['has_attach_method'] = {
                'found': True,
                'methods': attach_methods,
                'confidence': 0.85
            }
        
        detach_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                         ['detach', 'remove_observer', 'unregister', 'remove_listener'])]
        if detach_methods:
            evidences['has_detach_method'] = {
                'found': True,
                'methods': detach_methods,
                'confidence': 0.85
            }
        
        notify_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                         ['notify', 'notify_all', 'notify_observers'])]
        if notify_methods:
            evidences['has_notify_method'] = {
                'found': True,
                'methods': notify_methods,
                'confidence': 0.90
            }
        
        # Update method (para Observer)
        update_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                         ['update', 'on_update', 'handle_update'])]
        if update_methods:
            evidences['has_update_method'] = {
                'found': True,
                'methods': update_methods,
                'confidence': 0.85
            }
        
        # Loop de notificación
        loop_methods = [m for m in class_node.children 
                       if m.node_type == "method" and m.metadata.get('has_loop_notify', False)]
        if loop_methods:
            evidences['has_loop_notification'] = {
                'found': True,
                'methods': [m.name for m in loop_methods],
                'confidence': 0.95
            }
        
        return evidences
    
    def _check_event_driven_observer(self, class_node: ASTNode) -> Dict[str, Any]:
        evidences = {}
        
        # Atributos de eventos
        event_attrs = class_node.metadata.get('event_attributes', [])
        if event_attrs:
            evidences['has_event_attributes'] = {
                'found': True,
                'attributes': event_attrs,
                'confidence': 0.80
            }
        
        # Métodos de eventos
        methods = class_node.metadata.get('methods', [])
        
        # Métodos on_* (event handlers)
        on_methods = [m for m in methods if m.startswith('on_') or m.startswith('handle_')]
        if on_methods:
            evidences['has_event_handlers'] = {
                'found': True,
                'methods': on_methods,
                'confidence': 0.85
            }
        
        # Métodos emit/trigger
        emit_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                       ['emit', 'trigger', 'fire_event'])]
        if emit_methods:
            evidences['has_emit_methods'] = {
                'found': True,
                'methods': emit_methods,
                'confidence': 0.90
            }
        
        # Métodos bind/on para registrar eventos
        bind_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                       ['bind', 'on', 'add_event_listener'])]
        if bind_methods:
            evidences['has_bind_methods'] = {
                'found': True,
                'methods': bind_methods,
                'confidence': 0.85
            }
        
        return evidences
    
    def _check_publisher_subscriber(self, class_node: ASTNode) -> Dict[str, Any]:
        evidences = {}
        
        # Atributos pub/sub
        pubsub_attrs = class_node.metadata.get('pubsub_attributes', [])
        if pubsub_attrs:
            evidences['has_pubsub_attributes'] = {
                'found': True,
                'attributes': pubsub_attrs,
                'confidence': 0.85
            }
        
        # Métodos pub/sub
        methods = class_node.metadata.get('methods', [])
        
        subscribe_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                           ['subscribe', 'sub'])]
        if subscribe_methods:
            evidences['has_subscribe_methods'] = {
                'found': True,
                'methods': subscribe_methods,
                'confidence': 0.95
            }
        
        unsubscribe_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                             ['unsubscribe', 'unsub'])]
        if unsubscribe_methods:
            evidences['has_unsubscribe_methods'] = {
                'found': True,
                'methods': unsubscribe_methods,
                'confidence': 0.95
            }
        
        publish_methods = [m for m in methods if any(keyword in m.lower() for keyword in 
                         ['publish', 'pub'])]
        if publish_methods:
            evidences['has_publish_methods'] = {
                'found': True,
                'methods': publish_methods,
                'confidence': 0.95
            }
        
        return evidences
    
    def _determine_pattern_type(self, evidences: Dict[str, Any]) -> str:
        # Filtro: Si es cliente, no es patrón Observer
        if evidences.get('is_observer_client', {}).get('found', False):
            return "observer_client"
        
        # Publisher-Subscriber (más específico)
        pubsub_score = 0
        if evidences.get('has_pubsub_attributes', {}).get('found', False):
            pubsub_score += 1
        if evidences.get('has_subscribe_methods', {}).get('found', False):
            pubsub_score += 1
        if evidences.get('has_unsubscribe_methods', {}).get('found', False):
            pubsub_score += 1
        if evidences.get('has_publish_methods', {}).get('found', False):
            pubsub_score += 1
        
        if pubsub_score >= 3:
            return "publisher_subscriber"
        
        # Event-Driven Observer
        event_score = 0
        if evidences.get('has_event_attributes', {}).get('found', False):
            event_score += 1
        if evidences.get('has_event_handlers', {}).get('found', False):
            event_score += 1
        if evidences.get('has_emit_methods', {}).get('found', False):
            event_score += 1
        if evidences.get('has_bind_methods', {}).get('found', False):
            event_score += 1
        
        if event_score >= 2:
            return "event_driven_observer"
        
        # Classic Observer
        classic_score = 0
        if evidences.get('has_observer_list', {}).get('found', False):
            classic_score += 1
        if evidences.get('has_attach_method', {}).get('found', False):
            classic_score += 1
        if evidences.get('has_detach_method', {}).get('found', False):
            classic_score += 1
        if evidences.get('has_notify_method', {}).get('found', False):
            classic_score += 1
        if evidences.get('has_loop_notification', {}).get('found', False):
            classic_score += 1
        
        if classic_score >= 3:
            return "classic_observer"
        
        # Solo update method
        if evidences.get('has_update_method', {}).get('found', False):
            return "observer_implementation"
        
        return "no_observer"
    
    def _calculate_confidence(self, evidences: Dict[str, Any], pattern_type: str) -> float:
        if pattern_type in ["no_observer", "observer_client"]:
            return 0.1 if pattern_type == "observer_client" else 0.0
        
        weights = {
            # Classic Observer weights
            'has_observer_list': 0.20,
            'has_attach_method': 0.15,
            'has_detach_method': 0.15,
            'has_notify_method': 0.20,
            'has_loop_notification': 0.25,
            'has_update_method': 0.15,
            
            # Event-Driven Observer weights
            'has_event_attributes': 0.20,
            'has_event_handlers': 0.25,
            'has_emit_methods': 0.30,
            'has_bind_methods': 0.25,
            
            # Publisher-Subscriber weights
            'has_pubsub_attributes': 0.15,
            'has_subscribe_methods': 0.30,
            'has_unsubscribe_methods': 0.25,
            'has_publish_methods': 0.30
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for evidence_type, weight in weights.items():
            evidence = evidences.get(evidence_type, {})
            if isinstance(evidence, dict) and evidence.get('found', False):
                confidence = evidence.get('confidence', 0.0)
                total_confidence += confidence * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _check_violations(self, class_node: ASTNode, evidences: Dict[str, Any], pattern_type: str) -> List[str]:
        violations = []
        
        if pattern_type == "classic_observer":
            if not evidences.get('has_observer_list', {}).get('found', False):
                violations.append("Falta lista de observadores")
            if not evidences.get('has_notify_method', {}).get('found', False):
                violations.append("Falta método de notificación")
            
        elif pattern_type == "event_driven_observer":
            if not evidences.get('has_emit_methods', {}).get('found', False):
                violations.append("Falta método de emisión de eventos")
                
        elif pattern_type == "publisher_subscriber":
            if not evidences.get('has_publish_methods', {}).get('found', False):
                violations.append("Falta método publish")
            if not evidences.get('has_subscribe_methods', {}).get('found', False):
                violations.append("Falta método subscribe")
        
        return violations

# =============================================================================
# FASE 4: GENERADOR DE CÓDIGO INTERMEDIO
# =============================================================================

class IntermediateCodeGenerator:
    def __init__(self):
        self.ir = None
    
    def generate(self, ast_nodes: List[ASTNode], patterns: Dict[str, SemanticInfo], 
                relations: List[ClassRelation]) -> IntermediateRepresentation:
        print("Generando código intermedio...")
        
        classes = {}
        
        for class_node in [node for node in ast_nodes if node.node_type == "class"]:
            class_name = class_node.name
            
            classes[class_name] = {
                'name': class_name,
                'line': class_node.line,
                'bases': class_node.metadata.get('bases', []),
                'methods': self._extract_methods(class_node),
                'observer_attributes': class_node.metadata.get('observer_attributes', []),
                'event_attributes': class_node.metadata.get('event_attributes', []),
                'pubsub_attributes': class_node.metadata.get('pubsub_attributes', []),
                'constructor_info': class_node.metadata.get('constructor_info', {}),
                'complexity': self._calculate_complexity(class_node)
            }
        
        filtered_relations = self._filter_relations(relations, patterns)
        
        metadata = {
            'total_classes': len(classes),
            'observer_patterns': len([p for p in patterns.values() 
                                    if p.pattern_type not in ["no_observer", "observer_client"]]),
            'observer_clients': len([p for p in patterns.values() 
                                   if p.pattern_type == "observer_client"]),
            'classic_observers': len([p for p in patterns.values() if p.pattern_type == "classic_observer"]),
            'event_driven_observers': len([p for p in patterns.values() if p.pattern_type == "event_driven_observer"]),
            'publisher_subscribers': len([p for p in patterns.values() if p.pattern_type == "publisher_subscriber"]),
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
    
    def _extract_methods(self, class_node: ASTNode) -> List[Dict[str, Any]]:
        methods = []
        for child in class_node.children:
            if child.node_type == "method":
                methods.append({
                    'name': child.name,
                    'line': child.line,
                    'has_loop_notify': child.metadata.get('has_loop_notify', False),
                    'calls_update': child.metadata.get('calls_update', False),
                    'visibility': 'private' if child.name.startswith('_') else 'public'
                })
        return methods
    
    def _calculate_complexity(self, class_node: ASTNode) -> int:
        complexity = len([c for c in class_node.children if c.node_type == "method"])
        complexity += len(class_node.metadata.get('observer_attributes', []))
        complexity += len(class_node.metadata.get('event_attributes', []))
        complexity += len(class_node.metadata.get('pubsub_attributes', []))
        return complexity
    
    def _filter_relations(self, relations: List[ClassRelation], 
                         patterns: Dict[str, SemanticInfo]) -> List[ClassRelation]:
        observer_classes = set()
        
        for class_name, pattern_info in patterns.items():
            if (pattern_info.pattern_type not in ["no_observer", "observer_client"] and 
                pattern_info.confidence >= 0.5):
                observer_classes.add(class_name)
        
        filtered = []
        for relation in relations:
            if (relation.target in observer_classes or 
                relation.source in observer_classes):
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
        
        for class_name, pattern_info in ir.patterns.items():
            optimized_pattern = self._optimize_pattern(pattern_info, ir.classes[class_name], ir.relations)
            
            if optimized_pattern.pattern_type == "observer_client":
                false_positives.append(class_name)
                optimized_pattern.confidence = 0.1
            elif self._is_false_positive(optimized_pattern, ir.classes[class_name]):
                false_positives.append(class_name)
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
            false_positives_removed=false_positives,
            confidence_adjustments=confidence_adjustments
        )
        
        print(f"Falsos positivos removidos: {len(false_positives)}")
        return self.optimized_data
    
    def _optimize_pattern(self, pattern_info: SemanticInfo, class_info: Dict[str, Any], 
                         relations: List[ClassRelation]) -> SemanticInfo:
        optimized = SemanticInfo(
            class_name=pattern_info.class_name,
            pattern_type=pattern_info.pattern_type,
            confidence=pattern_info.confidence,
            evidences=pattern_info.evidences.copy(),
            violations=pattern_info.violations.copy()
        )
        
        # Bonificar patrones más completos
        if pattern_info.pattern_type == "publisher_subscriber":
            optimized.confidence = min(optimized.confidence * 1.15, 1.0)
        elif pattern_info.pattern_type == "event_driven_observer":
            optimized.confidence = min(optimized.confidence * 1.10, 1.0)
        elif pattern_info.pattern_type == "classic_observer":
            optimized.confidence = min(optimized.confidence * 1.05, 1.0)
        elif pattern_info.pattern_type == "observer_client":
            optimized.confidence = 0.1
        
        # Bonificar si otras clases usan el observer
        usage_relations = [r for r in relations if r.target == pattern_info.class_name 
                          and r.relation_type in [RelationType.ASSOCIATION, RelationType.DEPENDENCY]]
        if len(usage_relations) >= 2:
            optimized.confidence = min(optimized.confidence * 1.10, 1.0)
        elif len(usage_relations) == 1:
            optimized.confidence = min(optimized.confidence * 1.05, 1.0)
        
        return optimized
    
    def _optimize_relations(self, relations: List[ClassRelation], 
                          patterns: Dict[str, SemanticInfo]) -> List[ClassRelation]:
        optimized_relations = []
        
        for relation in relations:
            source_pattern = patterns.get(relation.source)
            target_pattern = patterns.get(relation.target)
            
            valid_source = (source_pattern and 
                           source_pattern.pattern_type not in ["no_observer", "false_positive", "observer_client"] and
                           source_pattern.confidence >= 0.5)
            
            valid_target = (target_pattern and 
                           target_pattern.pattern_type not in ["no_observer", "false_positive", "observer_client"] and
                           target_pattern.confidence >= 0.5)
            
            if valid_source or valid_target:
                optimized_relations.append(relation)
        
        return optimized_relations
    
    def _is_false_positive(self, pattern_info: SemanticInfo, class_info: Dict[str, Any]) -> bool:
        if pattern_info.confidence < 0.4:
            return True
        
        if len(pattern_info.violations) >= 3:
            return True
        
        total_elements = (len(class_info.get('methods', [])) + 
                         len(class_info.get('observer_attributes', [])) + 
                         len(class_info.get('event_attributes', [])) + 
                         len(class_info.get('pubsub_attributes', [])))
        if total_elements < 2 and pattern_info.confidence < 0.7:
            return True
        
        return False

# =============================================================================
# FASE 6: GENERADOR UML
# =============================================================================

class UMLGenerator:
    def __init__(self):
        self.templates = {
            'classic_observer': self._classic_observer_template,
            'event_driven_observer': self._event_driven_observer_template,
            'publisher_subscriber': self._publisher_subscriber_template,
            'observer_implementation': self._observer_implementation_template,
            'observer_client': self._observer_client_template,
            'no_observer': self._no_pattern_template,
            'false_positive': self._false_positive_template
        }
        
        self.relation_styles = {
            RelationType.INHERITANCE: "--|>",
            RelationType.COMPOSITION: "*--",
            RelationType.ASSOCIATION: "--",
            RelationType.DEPENDENCY: "..>",
            RelationType.OBSERVER: "..>"
        }
    
    def generate(self, optimized_data: OptimizedData, ir_data: IntermediateRepresentation = None) -> str:
        print("Generando código UML...")
        
        self._ir_data = ir_data
        
        uml_content = self._create_header()
        
        observer_patterns = [p for p in optimized_data.patterns.values() 
                           if p.pattern_type not in ["no_observer", "false_positive", "observer_client"]]
        
        observer_clients = [p for p in optimized_data.patterns.values() 
                          if p.pattern_type == "observer_client"]
        
        if not observer_patterns:
            uml_content += self._no_patterns_section()
        else:
            for pattern in observer_patterns:
                template_func = self.templates.get(pattern.pattern_type, self._unknown_template)
                uml_content += template_func(pattern)
            
            uml_content += self._generate_relations(optimized_data.relations)
        
        if observer_clients:
            uml_content += self._clients_section(observer_clients)
        
        real_false_positives = [fp for fp in optimized_data.false_positives_removed
                              if not any(p.class_name == fp and p.pattern_type == "observer_client" 
                                       for p in optimized_data.patterns.values())]
        if real_false_positives:
            uml_content += self._false_positives_section(real_false_positives)
        
        uml_content += "\n@enduml"
        
        print(f"Patrones generados: {len(observer_patterns)}")
        print(f"Clientes identificados: {len(observer_clients)}")
        print(f"Relaciones incluidas: {len(optimized_data.relations)}")
        
        return uml_content
    
    def _create_header(self) -> str:
        return f"""@startuml
!theme cerulean-outline

title Análisis de Patrones Observer
' Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    def _classic_observer_template(self, pattern: SemanticInfo) -> str:
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_observer_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<ClassicObserver>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Classic Observer Pattern
  Confianza: {int(pattern.confidence * 100)}%
  Subject con attach/detach/notify
end note

"""
    
    def _event_driven_observer_template(self, pattern: SemanticInfo) -> str:
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_event_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<EventDrivenObserver>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Event-Driven Observer
  Confianza: {int(pattern.confidence * 100)}%
  Sistema de eventos con callbacks
end note

"""
    
    def _publisher_subscriber_template(self, pattern: SemanticInfo) -> str:
        class_info = self._get_class_info(pattern.class_name)
        attributes = self._format_pubsub_attributes(class_info)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<PublisherSubscriber>> {{
{attributes}
  --
{methods}
}}

note top of {pattern.class_name}
  Publisher-Subscriber Pattern
  Confianza: {int(pattern.confidence * 100)}%
  Pub/Sub con topics/channels
end note

"""
    
    def _observer_implementation_template(self, pattern: SemanticInfo) -> str:
        class_info = self._get_class_info(pattern.class_name)
        methods = self._format_methods(class_info)
        
        return f"""
class {pattern.class_name} <<ObserverImpl>> {{
  --
{methods}
}}

note top of {pattern.class_name}
  Observer Implementation
  Confianza: {int(pattern.confidence * 100)}%
  Implementa interfaz Observer
end note

"""
    
    def _observer_client_template(self, pattern: SemanticInfo) -> str:
        return f"""
class {pattern.class_name} <<Client>> {{
  - observer_components: Observer[]
  --
  + coordinate_observers()
}}

note top of {pattern.class_name}
  Observer Client
  Coordina observers pero no ES observer
end note

"""
    
    def _no_pattern_template(self, pattern: SemanticInfo) -> str:
        return ""
    
    def _false_positive_template(self, pattern: SemanticInfo) -> str:
        return f"""
class {pattern.class_name} <<FalsoPositivo>> {{
}}

"""
    
    def _unknown_template(self, pattern: SemanticInfo) -> str:
        return f"""
class {pattern.class_name} {{
}}

"""
    
    def _get_class_info(self, class_name: str) -> Dict[str, Any]:
        if hasattr(self, '_ir_data') and self._ir_data and class_name in self._ir_data.classes:
            return self._ir_data.classes[class_name]
        return {'methods': [], 'observer_attributes': [], 'event_attributes': [], 'pubsub_attributes': []}
    
    def _format_observer_attributes(self, class_info: Dict[str, Any]) -> str:
        attrs = class_info.get('observer_attributes', [])
        if not attrs:
            return "  - _observers: List[Observer]"
        
        formatted_attrs = []
        for attr in attrs[:3]:
            formatted_attrs.append(f"  - {attr}: List[Observer]")
        
        return "\n".join(formatted_attrs)
    
    def _format_event_attributes(self, class_info: Dict[str, Any]) -> str:
        attrs = class_info.get('event_attributes', [])
        if not attrs:
            return "  - _event_handlers: Dict[str, Callable]"
        
        formatted_attrs = []
        for attr in attrs[:3]:
            formatted_attrs.append(f"  - {attr}: Callable")
        
        return "\n".join(formatted_attrs)
    
    def _format_pubsub_attributes(self, class_info: Dict[str, Any]) -> str:
        attrs = class_info.get('pubsub_attributes', [])
        if not attrs:
            return "  - _subscribers: Dict[str, List[Callable]]"
        
        formatted_attrs = []
        for attr in attrs[:3]:
            formatted_attrs.append(f"  - {attr}: Dict[str, List]")
        
        return "\n".join(formatted_attrs)
    
    def _format_methods(self, class_info: Dict[str, Any]) -> str:
        methods = class_info.get('methods', [])
        if not methods:
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
  No se encontraron patrones Observer
  en el código analizado.
end note

"""
    
    def _clients_section(self, observer_clients: List[SemanticInfo]) -> str:
        if not observer_clients:
            return ""
        
        clients_list = ', '.join([client.class_name for client in observer_clients])
        return f"""
note as ObserverClients
  Observer Clients: {len(observer_clients)}
  Clases: {clients_list}
  (Coordinan observers sin implementar el patrón)
end note

"""
    
    def _false_positives_section(self, false_positives: List[str]) -> str:
        return f"""
note as FalsePositives
  Falsos Positivos: {len(false_positives)}
  Clases: {', '.join(false_positives)}
end note

"""

# =============================================================================
# COMPILADOR PRINCIPAL
# =============================================================================

class ObserverCompiler:
    def __init__(self):
        self.lexer = LexicalAnalyzer()
        self.parser = SyntaxAnalyzer()
        self.semantic = SemanticAnalyzer()
        self.ir_generator = IntermediateCodeGenerator()
        self.optimizer = Optimizer()
        self.uml_generator = UMLGenerator()
    
    def compile(self, input_file: str, output_file: str) -> bool:
        print("COMPILADOR DE PATRONES OBSERVER")
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
#
#def main():
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
#    compiler = ObserverCompiler()
#    success = compiler.compile(input_file, output_file)
#    
#    sys.exit(0 if success else 1)
#
#if __name__ == "__main__":
#    main()