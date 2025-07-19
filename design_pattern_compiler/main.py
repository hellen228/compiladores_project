import sys
import os
from patterns.singleton.pattern import SingletonAnalyzer

def main():
    if len(sys.argv) != 3:
        print("Uso: python singleton_compiler.py <entrada.py> <salida.puml>")
        print("Ejemplo: python singleton_compiler.py codigo.py diagrama.puml")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"ERROR: El archivo {input_file} no existe")
        sys.exit(1)
    
    if not output_file.endswith('.puml'):
        print("RECOMENDACION: Use extensión .puml para el archivo de salida")

    print("INICIANDO COMPILACION - ANALISIS DE PATRONES SINGLETON")
    print("=" * 60)
    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo de salida: {output_file}")
    print("=" * 60)

    compiler = SingletonAnalyzer()
    success = compiler.compile(input_file, output_file)

#sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()