import random
import secrets

class ClassicalRandomGenerator:
    def generate_random_bits(self, num_bits: int) -> str:
        """Gera bits pseudoaleatórios usando geradores clássicos"""
        return ''.join([str(secrets.randbelow(2)) for _ in range(num_bits)])
    
    def generate_weak_random_bits(self, num_bits: int) -> str:
        """Gera bits com gerador mais fraco para comparação"""
        return ''.join([str(random.randint(0, 1)) for _ in range(num_bits)])