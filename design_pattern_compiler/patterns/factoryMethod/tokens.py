from enum import Enum
#from core.models import Token

#tokens específicos del patrón

class FactoryTokenType(Enum):
    FACTORY_METHOD = "factory_method"
    FACTORY_CLASS = "factory_class"
    CREATOR_METHOD = "creator_method"
    PRODUCT_RETURN = "product_return"
    CONDITIONAL_CREATION = "conditional_creation"
    FACTORY_PARAMS = "factory_params"
