import functools

# Singleton
class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def log(self, message):
        print(f"[LOG] {message}")

# Decorator
def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = Logger()
        logger.log(f"Ejecutando '{func.__name__}'")
        result = func(*args, **kwargs)
        logger.log(f"Finalizó '{func.__name__}'")
        return result
    return wrapper

# Uso
@log_execution
def saludar(nombre):
    print(f"Hola, {nombre}!")

saludar("Hellen")
