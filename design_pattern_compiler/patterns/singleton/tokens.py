from enum import Enum
#from core.models import Token

#tokens específicos del patrón

class SingletonTokenType(Enum):
    INSTANCE_VAR = "instance_variable"
    ACCESS_METHOD = "access_method"
    NEW_OVERRIDE = "new_override"
    CONTROL_FLOW = "control_flow"
    STATIC_DECORATOR = "static_decorator"
    CLASS_DEF = "class_definition"