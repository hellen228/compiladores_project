from patterns.singleton_compiler import SingletonCompiler
from patterns.decorator_compiler import DecoratorCompiler
from patterns.observer_compiler import ObserverCompiler
from patterns.factory_method_compiler import FactoryMethodCompiler

class MultiPatternCompiler:
    def __init__(self):
        self.compilers = [
            SingletonCompiler(),
            DecoratorCompiler(),
            ObserverCompiler(),
            FactoryMethodCompiler()
        ]
        self.results = []  # para guardar UMLs o IR combinados

    def compile(self, input_file: str, output_file: str) -> bool:
        uml_parts = []
        detected_any = False

        for compiler in self.compilers:
            success = compiler.compile(input_file, output_file + ".temp")
            if success:
                detected_any = True
                # Leer el UML parcial generado
                with open(output_file + ".temp", "r", encoding="utf-8") as f:
                    uml_parts.append(f.read())

        # Combinar todos los UMLs encontrados
        if detected_any:
            combined_uml = self.merge_uml_parts(uml_parts)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(combined_uml)
            print("\nSe generó el diagrama UML con múltiples patrones de diseño.")
            return True
        else:
            print("\nNo se detectaron patrones de diseño.")
            return False

    def merge_uml_parts(self, parts, filtrar_falsos_positivos = True):
        start = "@startuml\n"
        end = "@enduml\n"
        body = ""
        seen_theme = False
        inside_false_notes = False

        for part in parts:
            lines = part.strip().splitlines()
            for line in lines:
                stripped = line.strip()
                
                # Ignorar bloques de inicio y fin repetidos
                if stripped.startswith("@startuml") or stripped.startswith("@enduml"):
                    continue

                # Evitar múltiples !theme
                if stripped.startswith("!theme"):
                    if seen_theme:
                        continue
                    else:
                        seen_theme = True

                # Filtrar bloques de falsos positivos si se desea
                if filtrar_falsos_positivos:
                    if "note as FalsePositives" in stripped:
                        inside_false_notes = True
                        continue
                    if inside_false_notes:
                        if stripped == "end note":
                            inside_false_notes = False
                        continue
                # Opcional: podrías filtrar titles también
                if stripped.startswith("title Análisis de Patrones"):
                    continue

                body += line + "\n"

        return start + body + end
