from typing import Any, Dict, List
from datetime import datetime
from models import RelationType, SemanticInfo, ClassRelation, OptimizedData, IntermediateRepresentation

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
