from typing import List
import re
from tokens import SingletonTokenType as TokenType
from core.models import Token

class SingletonLexicalAnalyzer:
    """Detecta tokens relacionados con patrones Singleton"""
    def __init__(self):
        self.tokens = []
        self.symbol_table = {}
        
        self.token_patterns = {
            TokenType.INSTANCE_VAR: [
                r'_instance\b', r'__instance\b', r'_obj\b', 
                r'_singleton\b', r'__singleton\b'
            ],
            TokenType.ACCESS_METHOD: [
                r'get_instance\b', r'getInstance\b', r'instance\b',
                r'get_singleton\b', r'create\b'
            ],
            TokenType.NEW_OVERRIDE: [r'__new__\b'],
            TokenType.CONTROL_FLOW: [
                r'hasattr\b', r'is None\b', r'== None\b', r'if\b'
            ],
            TokenType.STATIC_DECORATOR: [r'@staticmethod\b', r'@classmethod\b'],
            TokenType.CLASS_DEF: [r'class\s+\w+']
        }

    def tokenize(self, source_code: str) -> List[Token]:
        """Ejecuta el análisis léxico del código fuente"""
        print("FASE 1: ANALISIS LEXICO")
        print("-" * 40)
        
        lines = source_code.split('\n')
        tokens = []
        
        for line_num, line in enumerate(lines, 1):
            for token_type, patterns in self.token_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        token = Token(
                            type=token_type,
                            value=match.group(),
                            line=line_num,
                            column=match.start(),
                            confidence=self._calculate_token_confidence(token_type, match.group())
                        )
                        tokens.append(token)
                        
                        if token.value not in self.symbol_table:
                            self.symbol_table[token.value] = {
                                'type': token_type,
                                'occurrences': 0,
                                'lines': []
                            }
                        self.symbol_table[token.value]['occurrences'] += 1
                        self.symbol_table[token.value]['lines'].append(line_num)
        
        self.tokens = tokens
        self._print_lexical_results()
        return tokens

    def _calculate_token_confidence(self, token_type: TokenType, value: str) -> float:
        """Calcula el nivel de confianza de un token basado en su tipo y valor"""
        confidence_map = {
            TokenType.INSTANCE_VAR: {
                '_instance': 0.95, '__instance': 0.90, '_obj': 0.70,
                '_singleton': 0.85, '__singleton': 0.80
            },
            TokenType.ACCESS_METHOD: {
                'get_instance': 0.95, 'getInstance': 0.90, 'instance': 0.75,
                'get_singleton': 0.85, 'create': 0.60
            },
            TokenType.NEW_OVERRIDE: {'__new__': 0.95},
            TokenType.CONTROL_FLOW: {
                'hasattr': 0.85, 'is None': 0.80, '== None': 0.75, 'if': 0.50
            },
            TokenType.STATIC_DECORATOR: {
                '@staticmethod': 0.90, '@classmethod': 0.85
            }
        }
        return confidence_map.get(token_type, {}).get(value, 0.50)
    
    def get_token_stats(self) -> dict:
        """Retorna estadísticas para usar donde quieras"""
        return {tt.value: len([t for t in self.tokens if t.type == tt]) 
                for tt in TokenType if len([t for t in self.tokens if t.type == tt]) > 0}

    def _print_lexical_results(self):
        """Ahora usa get_token_stats()"""
        print(f"Tokens encontrados: {len(self.tokens)}")
        stats = self.get_token_stats()
        for token_type, count in stats.items():
            print(f" {token_type}: {count}")

