#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PoC: QRNG (via Qiskit + Aer) -> OTP -> Comparação com PRNG
Autor: você :)
Requisitos: pip install qiskit qiskit-aer numpy scipy

Notas:
- Em simulador, a "aleatoriedade quântica" é emulada por um PRNG.
- Use esta PoC para demonstrar o FLUXO. Para evidência de QRNG real,
  execute em um backend físico (IBM Quantum), se possível.
"""

import gzip
import io
import math
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from scipy.stats import norm, chisquare

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.result import Result

# ---------------------------
# Configurações principais
# ---------------------------

N_BITS = 100_000          # tamanho da amostra para testes estatísticos
MSG = b"Seguranca digital do Serpro com OTP e QRNG (demo)!"  # mensagem de exemplo para OTP
RANDOM_SEED = None        # defina um int para resultados reprodutíveis do simulador

# ---------------------------
# Utilitários gerais
# ---------------------------

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Converte array de bits (0/1) para bytes."""
    # Preenche até múltiplo de 8
    pad_len = (-len(bits)) % 8
    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    # Agrupa em bytes
    bits = bits.reshape(-1, 8)
    # bit mais significativo primeiro
    byte_vals = np.packbits(bits, axis=1, bitorder='big').flatten()
    return byte_vals.tobytes()

def bytes_to_bits(data: bytes) -> np.ndarray:
    """Converte bytes para array de bits (0/1)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='big')
    return bits.astype(np.uint8)

def otp_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """OTP: cifra = XOR byte a byte (key deve ter mesmo tamanho de plaintext)."""
    if len(key) < len(plaintext):
        raise ValueError("Chave menor que a mensagem para OTP.")
    return bytes([p ^ k for p, k in zip(plaintext, key[:len(plaintext)])])

def otp_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """OTP: decifra = XOR (mesma operação)."""
    return otp_encrypt(ciphertext, key)

def gzip_size(data: bytes) -> int:
    """Retorna tamanho comprimido com gzip (proxy de compressibilidade)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(data)
    return len(buf.getvalue())

# ---------------------------
# QRNG com Qiskit Aer
# ---------------------------

def qrng_bits_qiskit(n_bits: int, seed: int | None = None) -> Tuple[np.ndarray, float]:
    """
    Gera n_bits via circuito quântico simples (H + medida) no simulador.
    Retorna (bits, tempo_segundos).
    """
    start = time.time()

    # Estratégia: empacotar múltiplos bits por circuito para eficiência.
    # Ex.: 1024 qubits por circuito, 1 shot, medir todos.
    # Ajuste esse "fanout" conforme memória/tempo disponível.
    fanout = 1024
    rounds = math.ceil(n_bits / fanout)

    sim = AerSimulator(seed_simulator=seed) if seed is not None else AerSimulator()

    collected: List[int] = []
    for _ in range(rounds):
        n = min(fanout, n_bits - len(collected))
        qc = QuantumCircuit(n, n)
        qc.h(range(n))         # Hadamards em todos os qubits
        qc.measure(range(n), range(n))
        job = sim.run(qc, shots=1)
        result: Result = job.result()
        counts = result.get_counts()
        # counts é um dict: {'0101...': 1}
        bitstring = next(iter(counts.keys()))
        # Qiskit retorna bitstring little-endian por padrão; alinhar ao sentido clássico
        # Aqui, como mapeamos qubits->clássicos na mesma ordem, vamos inverter:
        bitstring = bitstring[::-1]
        bits = [int(b) for b in bitstring[:n]]
        collected.extend(bits)

    end = time.time()
    arr = np.array(collected[:n_bits], dtype=np.uint8)
    return arr, (end - start)

# ---------------------------
# PRNG seguro (baseline)
# ---------------------------

def prng_bits_secure(n_bits: int) -> Tuple[np.ndarray, float]:
    """
    Usa os.urandom/secrets via NumPy para obter bits “verdadeiramente” imprevisíveis do SO.
    Retorna (bits, tempo_segundos).
    """
    import secrets
    start = time.time()
    n_bytes = (n_bits + 7) // 8
    data = secrets.token_bytes(n_bytes)
    bits = bytes_to_bits(data)[:n_bits]
    end = time.time()
    return bits, (end - start)

# ---------------------------
# Testes estatísticos básicos
# ---------------------------

@dataclass
class RandTestResult:
    name: str
    statistic: float
    p_value: float
    passed: bool
    details: str = ""

def monobit_frequency_test(bits: np.ndarray) -> RandTestResult:
    """
    NIST SP 800-22 Monobit: verifica se #1s ≈ #0s.
    Usa aproximação normal para p-value.
    """
    n = len(bits)
    s_obs = abs(np.sum(2*bits - 1)) / math.sqrt(n)  # estatística S (normalizada)
    p = 2 * (1 - norm.cdf(s_obs))
    return RandTestResult("Monobit Frequency", s_obs, p, p >= 0.01, f"n={n}")

def runs_test(bits: np.ndarray) -> RandTestResult:
    """
    NIST Runs Test: número de runs vs esperado (assume p(1) ~ 0.5).
    """
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) > (2 / math.sqrt(n)):  # condição NIST
        return RandTestResult("Runs", float('nan'), 0.0, False, "Frequência muito distante de 0.5")
    # conta runs
    runs = 1 + np.sum(bits[1:] != bits[:-1])
    expected = 2 * n * pi * (1 - pi)
    var = 2 * n * (2 * pi * (1 - pi))**2 / (n - 1) if n > 1 else 0.0
    z = (runs - expected) / math.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return RandTestResult("Runs", z, p, p >= 0.01, f"runs={runs}, pi={pi:.4f}")

def block_frequency_test(bits: np.ndarray, M: int = 1000) -> RandTestResult:
    """
    Block Frequency: divide em blocos de tamanho M e testa desvios.
    Usa qui-quadrado.
    """
    n = len(bits)
    N = n // M
    if N == 0:
        return RandTestResult("Block Frequency", float('nan'), 0.0, False, "Amostra menor que 1 bloco")
    blocks = bits[:N*M].reshape(N, M)
    pis = np.mean(blocks, axis=1)
    chi_sq = 4 * M * np.sum((pis - 0.5)**2)
    p = 1 - math.gamma(N/2) * (1/2)**(N/2) * (chi_sq)**(N/2 - 1) * math.exp(-chi_sq/2) / math.gamma(N/2)  # não use; reserva
    # Melhor: usar CDF qui-quadrado da scipy:
    from scipy.stats import chi2
    p = 1 - chi2.cdf(chi_sq, df=N)
    return RandTestResult("Block Frequency", chi_sq, p, p >= 0.01, f"N={N}, M={M}")

def serial_test_pairs(bits: np.ndarray) -> RandTestResult:
    """
    Serial test (ordem 2): frequência de pares 00,01,10,11 via qui-quadrado.
    """
    # forma os pares
    n = len(bits) // 2
    if n == 0:
        return RandTestResult("Serial(2)", float('nan'), 0.0, False, "Amostra muito pequena")
    pairs = bits[:2*n].reshape(n, 2)
    vals = pairs[:, 0] * 2 + pairs[:, 1]
    counts = np.bincount(vals, minlength=4)
    # esperado uniforme = n/4 por categoria
    expected = np.ones(4) * (n / 4)
    chi, p = chisquare(counts, f_exp=expected)
    return RandTestResult("Serial(2)", chi, p, p >= 0.01, f"counts={counts.tolist()}")

def summary_suite(bits: np.ndarray, label: str) -> List[RandTestResult]:
    res = [
        monobit_frequency_test(bits),
        runs_test(bits),
        block_frequency_test(bits, M=1000),
        serial_test_pairs(bits),
    ]
    print(f"\n=== Suite de testes: {label} ===")
    for r in res:
        print(f"{r.name:20s} | stat={r.statistic:.4f} | p={r.p_value:.4f} | pass={r.passed} | {r.details}")
    bias = abs(np.mean(bits) - 0.5)
    comp = gzip_size(bits_to_bytes(bits))
    print(f"Bias: {bias:.6f} | gzip_size: {comp} bytes")
    return res

# ---------------------------
# Execução principal
# ---------------------------

def main():
    print("=== PoC QRNG -> OTP -> Comparação ===")
    print(f"N_BITS para teste: {N_BITS}")

    # 1) Geração QRNG (simulador)
    q_bits, t_q = qrng_bits_qiskit(N_BITS, seed=RANDOM_SEED)
    print(f"QRNG(sim) gerou {len(q_bits)} bits em {t_q:.3f}s")

    # 2) Geração PRNG (OS)
    p_bits, t_p = prng_bits_secure(N_BITS)
    print(f"PRNG(OS) gerou {len(p_bits)} bits em {t_p:.3f}s")

    # 3) OTP demo com uma mensagem simples
    print("\n--- OTP DEMO ---")
    key_qrng_bytes = bits_to_bytes(q_bits)         # chave do QRNG(sim)
    key_prng_bytes = bits_to_bytes(p_bits)         # chave do PRNG
    c_q = otp_encrypt(MSG, key_qrng_bytes)
    m_q = otp_decrypt(c_q, key_qrng_bytes)
    c_p = otp_encrypt(MSG, key_prng_bytes)
    m_p = otp_decrypt(c_p, key_prng_bytes)
    assert m_q == MSG and m_p == MSG
    print(f"Mensagem original: {MSG}")
    print(f"Cifra com QRNG(sim): {c_q.hex()[:64]}... (len={len(c_q)})")
    print(f"Cifra com PRNG(OS): {c_p.hex()[:64]}... (len={len(c_p)})")
    print("OTP ok (decifragem íntegra).")

    # 4) Testes estatísticos
    qr = summary_suite(q_bits, "QRNG(sim)")
    pr = summary_suite(p_bits, "PRNG(OS)")

    # 5) Conclusão rápida
    pass_q = all(t.passed for t in qr)
    pass_p = all(t.passed for t in pr)
    print("\n--- Conclusão preliminar ---")
    print(f"QRNG(sim) passou todos? {pass_q}")
    print(f"PRNG(OS) passou todos? {pass_p}")
    print("\nObservação: No simulador, a aleatoriedade é pseudo; para validar QRNG real, "
          "considere executar em backend físico da IBM Quantum (opcional).")

if __name__ == "__main__":
    main()