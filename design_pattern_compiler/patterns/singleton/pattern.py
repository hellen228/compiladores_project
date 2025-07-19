from analyzer import SingletonLexicalAnalyzer, SingletonSemanticAnalyzer, SingletonSyntaxAnalyzer
from core.models import SemanticInfo
from intermediate.ir_generator import IntermediateCodeGenerator
from optimizer.optimizer import Optimizer
from core.uml_generator import UMLCodeGenerator

class SingletonAnalyzer:
    def get_name(self) -> str:
        return "Singleton"
    
    def __init__(self):
        self.lexer = SingletonLexicalAnalyzer()
        self.parser = SingletonSyntaxAnalyzer()
        self.semantic = SingletonSemanticAnalyzer()
    
    def compile(self, input_file: str, output_file: str) -> bool:
        """Ejecuta el proceso completo de compilación"""
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
