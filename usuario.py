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
        Logger().log(f"Ejecutando '{func.__name__}'")
        result = func(*args, **kwargs)
        Logger().log(f"Finalizó '{func.__name__}'")
        return result
    return wrapper

# Observer base
class Observer:
    def update(self, data):
        pass

# Subject base
class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, obs):
        self._observers.append(obs)

    def notify(self, data):
        for obs in self._observers:
            obs.update(data)

# Observer concreto
class EmailObserver(Observer):
    def update(self, data):
        Logger().log(f"[Email] Notificación: {data}")

class SMSObserver(Observer):
    def update(self, data):
        Logger().log(f"[SMS] Alerta recibida: {data}")

# Factory Method
class ObserverFactory:
    @staticmethod
    def create_observer(tipo):
        if tipo == "email":
            return EmailObserver()
        elif tipo == "sms":
            return SMSObserver()
        else:
            raise ValueError("Tipo de observador no soportado")

# Uso con Subject y Decorator
class AlertaSistema(Subject):
    @log_execution
    def nueva_alerta(self, mensaje):
        Logger().log(f"Generando alerta: {mensaje}")
        self.notify(mensaje)

# Main
if __name__ == "__main__":
    sistema = AlertaSistema()
    obs1 = ObserverFactory.create_observer("email")
    obs2 = ObserverFactory.create_observer("sms")
    sistema.attach(obs1)
    sistema.attach(obs2)
    sistema.nueva_alerta("Temperatura crítica detectada.")
