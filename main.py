import sys
import os
from plantuml import PlantUML
from patterns.singleton_compiler import SingletonCompiler
from patterns.decorator_compiler import DecoratorCompiler
from patterns.observer_compiler import ObserverCompiler
from patterns.factory_method_compiler import FactoryMethodCompiler

def main():
    """Función principal del programa"""

    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        print("Modo interactivo: no se detectaron argumentos.")
        input_file = input("Ingrese el nombre del archivo de entrada (.py): ")
        output_file = input("Ingrese el nombre del archivo de salida (.puml): ")

    #print(f"Archivo a compilar: {input_file}")
    #print(f"Archivo de salida: {output_file}")

    if not os.path.exists(input_file):
        print(f"ERROR: El archivo '{input_file}' no existe.")
        sys.exit(1)

    compiler = SingletonCompiler()
    compiler = DecoratorCompiler()
    compiler = ObserverCompiler()
    compiler = FactoryMethodCompiler()
    success = compiler.compile(input_file, output_file)

    #compiler2 = DecoratorCompiler()
    #success2 = compiler2.compile(input_file, output_file)

    if not output_file.endswith('.puml'):
        print("RECOMENDACIÓN: Use extensión '.puml' para el archivo de salida.")

    plantuml = PlantUML(url='http://www.plantuml.com/plantuml/img/') 
    plantuml.processes_file(output_file)

    sys.exit(0 if success else 1)
    #sys.exit(0 if success2 else 1)

if __name__ == "__main__":
    main()
