import math
from typing import List

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

class QuantumRandomGenerator:
    def __init__(self, seed: int = None):
        """
        Inicializa o gerador de números aleatórios quânticos.

        Args:
            seed: Semente opcional para o simulador para resultados reprodutíveis.
        """
        self.simulator = AerSimulator(seed_simulator=seed) if seed is not None else AerSimulator()

    def generate_random_bits(self, num_bits: int) -> str:
        """
        Gera `num_bits` aleatórios de forma eficiente usando circuitos quânticos em batch.

        Em vez de criar um circuito para cada bit, esta função cria circuitos maiores
        para gerar múltiplos bits de uma vez, melhorando significativamente a performance.

        Args:
            num_bits: O número de bits aleatórios a serem gerados.

        Returns:
            Uma string contendo os bits aleatórios.
        """
        if num_bits <= 0:
            return ""

        # Estratégia: empacotar múltiplos bits por circuito para eficiência.
        # O simulador Aer pode lidar com um grande número de qubits.
        # 1024 é um valor seguro e eficiente.
        max_qubits_per_circuit = 1024
        collected_bits: List[str] = []

        bits_to_generate = num_bits
        while bits_to_generate > 0:
            n_qubits = min(bits_to_generate, max_qubits_per_circuit)

            # Cria um circuito para o batch de bits
            qc = QuantumCircuit(n_qubits, n_qubits)
            qc.h(range(n_qubits))
            qc.measure(range(n_qubits), range(n_qubits))

            # Executa com 1 shot para obter uma única string de bits aleatórios.
            # A transpilação não é estritamente necessária para o AerSimulator
            # quando usamos portas de base.
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()

            # `get_counts()` retorna um dicionário como {'10110...': 1}.
            # A chave é o bitstring.
            bitstring = next(iter(counts))

            # A ordem dos bits em Qiskit é c_{n-1}...c_1c_0 (big-endian).
            # O código de PoC `poc_qrng_otp.py` inverte a string.
            # Para manter a consistência, fazemos o mesmo.
            bitstring = bitstring[::-1]

            collected_bits.extend(list(bitstring))
            bits_to_generate -= n_qubits

        # Garante que o número de bits retornado é exatamente o solicitado.
        return "".join(collected_bits[:num_bits])