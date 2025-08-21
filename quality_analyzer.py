import numpy as np
from scipy import stats
import math
from collections import Counter
from scipy.special import erfc
from typing import Dict, Any

class RandomnessAnalyzer:
    def __init__(self):
        pass
    
    def frequency_test(self, bits: str) -> dict:
        """Teste de frequência (deve ser ~50% 0s e 50% 1s)"""
        zeros = bits.count('0')
        ones = bits.count('1')
        total = len(bits)
        
        return {
            'zeros_count': zeros,
            'ones_count': ones,
            'zeros_percentage': zeros / total * 100,
            'ones_percentage': ones / total * 100,
            'balance_score': abs(50 - (ones / total * 100)),
            'total_bits': total
        }
    
    def runs_test(self, bits: str) -> dict:
        """Teste de sequências (runs test) expandido"""
        runs = []
        current_run = 1
        current_bit = bits[0] if bits else '0'
        
        for i in range(1, len(bits)):
            if bits[i] == bits[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Análise estatística dos runs
        runs_array = np.array(runs)
        
        return {
            'total_runs': len(runs),
            'average_run_length': np.mean(runs),
            'median_run_length': np.median(runs),
            'std_run_length': np.std(runs),
            'min_run_length': min(runs),
            'max_run_length': max(runs),
            'runs_distribution': runs,
            'run_lengths_histogram': dict(Counter(runs)),
            'long_runs_count': len([r for r in runs if r > 5]),
            'very_long_runs_count': len([r for r in runs if r > 10])
        }
    
    def entropy_calculation(self, bits: str) -> float:
        """Calcula entropia de Shannon"""
        zeros = bits.count('0')
        ones = bits.count('1')
        total = len(bits)
        
        if zeros == 0 or ones == 0:
            return 0
        
        p0 = zeros / total
        p1 = ones / total
        
        entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        return entropy
    
    def block_entropy(self, bits: str, block_size: int = 8) -> dict:
        """Calcula entropia em blocos"""
        if len(bits) < block_size:
            return {'error': 'Insufficient bits for block analysis'}
        
        entropies = []
        for i in range(0, len(bits) - block_size + 1, block_size):
            block = bits[i:i+block_size]
            if len(block) == block_size:
                entropies.append(self.entropy_calculation(block))
        
        return {
            'block_size': block_size,
            'num_blocks': len(entropies),
            'average_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_entropy': min(entropies) if entropies else 0,
            'max_entropy': max(entropies) if entropies else 0,
            'entropy_variance': np.var(entropies)
        }
    
    def autocorrelation_test(self, bits: str, max_lag: int = 20) -> dict:
        """Teste de autocorrelação para detectar padrões"""
        bit_array = np.array([int(b) for b in bits])
        n = len(bit_array)
        
        autocorrs = []
        for lag in range(1, min(max_lag + 1, n//2)):
            if n - lag > 0:
                corr = np.corrcoef(bit_array[:-lag], bit_array[lag:])[0,1]
                if not np.isnan(corr):
                    autocorrs.append(corr)
                else:
                    autocorrs.append(0.0)
        
        return {
            'max_lag_tested': len(autocorrs),
            'autocorrelations': autocorrs,
            'max_autocorr': max(autocorrs) if autocorrs else 0,
            'avg_autocorr': np.mean(np.abs(autocorrs)) if autocorrs else 0,
            'significant_correlations': len([c for c in autocorrs if abs(c) > 0.1])
        }
    
    def pattern_analysis(self, bits: str) -> dict:
        """Análise de padrões específicos"""
        patterns_2bit = {'00': 0, '01': 0, '10': 0, '11': 0}
        patterns_3bit = {}
        patterns_4bit = {}
        
        # Padrões de 2 bits
        for i in range(len(bits) - 1):
            pattern = bits[i:i+2]
            if pattern in patterns_2bit:
                patterns_2bit[pattern] += 1
        
        # Padrões de 3 bits
        for i in range(len(bits) - 2):
            pattern = bits[i:i+3]
            patterns_3bit[pattern] = patterns_3bit.get(pattern, 0) + 1
        
        # Padrões de 4 bits
        for i in range(len(bits) - 3):
            pattern = bits[i:i+4]
            patterns_4bit[pattern] = patterns_4bit.get(pattern, 0) + 1
        
        return {
            'patterns_2bit': patterns_2bit,
            'patterns_3bit': dict(patterns_3bit),
            'patterns_4bit': dict(patterns_4bit),
            'unique_3bit_patterns': len(patterns_3bit),
            'unique_4bit_patterns': len(patterns_4bit),
            'most_common_3bit': max(patterns_3bit.items(), key=lambda x: x[1]) if patterns_3bit else ('', 0),
            'most_common_4bit': max(patterns_4bit.items(), key=lambda x: x[1]) if patterns_4bit else ('', 0)
        }
    
    def chi_square_test(self, bits: str, block_size: int = 8) -> dict:
        """Teste qui-quadrado para uniformidade"""
        if len(bits) < block_size:
            return {'error': 'Insufficient bits'}
        
        # Dividir em blocos e contar valores
        values = []
        for i in range(0, len(bits) - block_size + 1, block_size):
            block = bits[i:i+block_size]
            if len(block) == block_size:
                values.append(int(block, 2))
        
        if not values:
            return {'error': 'No complete blocks'}
        
        # Calcular qui-quadrado
        expected_freq = len(values) / (2**block_size)
        observed_counts = Counter(values)
        
        chi_square = 0
        for i in range(2**block_size):
            observed = observed_counts.get(i, 0)
            chi_square += (observed - expected_freq)**2 / expected_freq
        
        degrees_freedom = 2**block_size - 1
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_freedom)
        
        return {
            'chi_square_statistic': chi_square,
            'degrees_of_freedom': degrees_freedom,
            'p_value': p_value,
            'block_size': block_size,
            'num_blocks': len(values),
            'passes_test': p_value > 0.01
        }
    
    def compression_test(self, bits: str) -> dict:
        """Teste de compressibilidade como medida de aleatoriedade"""
        import zlib
        
        # Converter bits para bytes
        byte_data = bytearray()
        for i in range(0, len(bits), 8):
            byte_chunk = bits[i:i+8]
            if len(byte_chunk) == 8:
                byte_data.append(int(byte_chunk, 2))
        
        original_size = len(byte_data)
        if original_size == 0:
            return {'error': 'No data to compress'}
        
        compressed = zlib.compress(bytes(byte_data))
        compression_ratio = len(compressed) / original_size
        
        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': len(compressed),
            'compression_ratio': compression_ratio,
            'compression_percentage': (1 - compression_ratio) * 100,
            'randomness_indicator': compression_ratio  # Closer to 1.0 = more random
        }
    
    def generate_report(self, quantum_bits: str, classical_bits: str) -> dict:
        """Gera relatório comparativo expandido"""
        return {
            'quantum': {
                'basic_stats': {
                    'total_bits': len(quantum_bits),
                    'frequency': self.frequency_test(quantum_bits),
                    'runs': self.runs_test(quantum_bits),
                    'entropy': self.entropy_calculation(quantum_bits)
                },
                'advanced_analysis': {
                    'block_entropy_8bit': self.block_entropy(quantum_bits, 8),
                    'block_entropy_16bit': self.block_entropy(quantum_bits, 16),
                    'autocorrelation': self.autocorrelation_test(quantum_bits),
                    'pattern_analysis': self.pattern_analysis(quantum_bits),
                    'chi_square_8bit': self.chi_square_test(quantum_bits, 8),
                    'chi_square_4bit': self.chi_square_test(quantum_bits, 4),
                    'compression': self.compression_test(quantum_bits)
                }
            },
            'classical': {
                'basic_stats': {
                    'total_bits': len(classical_bits),
                    'frequency': self.frequency_test(classical_bits),
                    'runs': self.runs_test(classical_bits),
                    'entropy': self.entropy_calculation(classical_bits)
                },
                'advanced_analysis': {
                    'block_entropy_8bit': self.block_entropy(classical_bits, 8),
                    'block_entropy_16bit': self.block_entropy(classical_bits, 16),
                    'autocorrelation': self.autocorrelation_test(classical_bits),
                    'pattern_analysis': self.pattern_analysis(classical_bits),
                    'chi_square_8bit': self.chi_square_test(classical_bits, 8),
                    'chi_square_4bit': self.chi_square_test(classical_bits, 4),
                    'compression': self.compression_test(classical_bits)
                }
            }
        }
    
    def nist_monobit_test(self, bits: str) -> dict:
        """NIST SP 800-22 Monobit Frequency Test"""
        import math
        from scipy.special import erfc
        
        n = len(bits)
        s_n = sum(int(bit) for bit in bits)
        s_obs = abs(s_n - n/2) / math.sqrt(n/4)
        p_value = erfc(s_obs / math.sqrt(2))
        
        return {
            'test_name': 'NIST Monobit',
            'statistic': s_obs,
            'p_value': p_value,
            'passes': p_value >= 0.01,
            'critical_value': 1.96  # 95% confidence
        }
    
# In file: /home/s299259068/Área de Trabalho/daily/premio_inovacao_2025/quantum/pipeline_projeto/src/quality_analyzer.py

# Replace the existing nist_runs_test function with this one.
# You will need to import 'math' and 'erfc' from 'scipy.special' at the top of the file if they aren't already.
# import math
# from scipy.special import erfc

    def nist_runs_test(self, bits_str: str):
        """
        NIST Runs Test (SP 800-22, Section 2.3).
        Checks for oscillations in the sequence that are too fast or too slow.
        This implementation is corrected to prevent 'math domain error'.
        """
        n = len(bits_str)
        if n == 0:
            return {'p_value': 0.0, 'passes': False, 'notes': 'Input sequence is empty.'}

        # Convert bit string to a list of integers
        bits = [int(b) for b in bits_str]
        pi = sum(bits) / n

        # Pre-test 1: The test requires the proportion of ones to be close to 0.5.
        # This is the monobit test prerequisite from the NIST documentation.
        tau = 2 / math.sqrt(n)
        if abs(pi - 0.5) >= tau:
            return {'p_value': 0.0, 'passes': False, 'notes': f'Monobit pre-test failed (pi={pi:.4f})'}

        # Pre-test 2: Handle edge case where sequence is all 0s or all 1s.
        if pi == 0 or pi == 1:
            return {'p_value': 0.0, 'passes': False, 'notes': 'Sequence is all 0s or all 1s.'}

        # Count the total number of runs (V_n)
        v_n = 1
        for i in range(n - 1):
            if bits[i] != bits[i+1]:
                v_n += 1
        
        # Calculate p-value directly using the formula from NIST SP 800-22.
        # This avoids the intermediate, error-prone variance calculation.
        # p_value = erfc(|V_n - 2*n*pi*(1-pi)| / (2*sqrt(2*n)*pi*(1-pi)))
        
        numerator = abs(v_n - 2 * n * pi * (1 - pi))
        denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)

        # This check prevents a ZeroDivisionError, although the pre-test should already handle it.
        if denominator < 1e-10:
            return {'p_value': 0.0, 'passes': False, 'notes': 'Denominator is near zero, cannot compute p-value.'}

        p_value = erfc(numerator / denominator)

        return {
            'p_value': p_value,
            'passes': p_value >= 0.01
        }


    def spectral_test(self, bits: str) -> dict:
        """Discrete Fourier Transform (Spectral) Test"""
        import numpy as np
        from scipy.special import erfc
        
        n = len(bits)
        x = np.array([2*int(bit) - 1 for bit in bits])  # Convert to +1,-1
        
        # Apply DFT
        S = np.fft.fft(x)
        M = np.abs(S[:n//2])
        
        # Theoretical threshold
        T = math.sqrt(math.log(1/0.05) * n)
        
        # Count peaks below threshold
        N_0 = 0.95 * n/2
        N_1 = len([m for m in M if m < T])
        
        d = (N_1 - N_0) / math.sqrt(n * 0.95 * 0.05 / 4)
        p_value = erfc(abs(d) / math.sqrt(2))
        
        return {
            'test_name': 'Spectral (DFT)',
            'statistic': d,
            'p_value': p_value,
            'passes': p_value >= 0.01,
            'peaks_below_threshold': N_1,
            'expected_peaks': N_0
        }
    
    @staticmethod
    def calculate_comparison_score(quality_report: Dict, nist_tests: Dict) -> Dict[str, Any]:
        """Calcula score comparativo de forma robusta, centralizando a lógica."""

        # Testes que retornam um booleano 'passes'
        pass_fail_tests = [
            "monobit", "runs", "spectral", "template", "linear_complexity"
        ]

        def count_passes(results: dict) -> int:
            # Safely count passes, ignoring tests that might have failed to run (e.g., due to small sample size)
            return sum(results.get(test, {}).get("passes", False) for test in pass_fail_tests if results.get(test))

        quantum_passes = count_passes(nist_tests.get("quantum", {}))
        classical_passes = count_passes(nist_tests.get("classical", {}))
        total_tests = len(pass_fail_tests)

        # Determinar vencedor
        q_report = quality_report.get("quantum", {}).get("basic_stats", {})
        c_report = quality_report.get("classical", {}).get("basic_stats", {})
        q_entropy = q_report.get("entropy", 0)
        c_entropy = c_report.get("entropy", 0)

        if quantum_passes > classical_passes:
            winner = "quantum"
            verdict = "Geração Quântica demonstrou superioridade estatística nos testes NIST."
        elif classical_passes > quantum_passes:
            winner = "classical"
            verdict = "Geração Clássica demonstrou superioridade estatística nos testes NIST."
        else:
            # Critério de desempate: entropia
            if abs(q_entropy - c_entropy) < 1e-6:
                 winner = "tie"
                 verdict = "Ambas as abordagens apresentaram performance estatística equivalente."
            elif q_entropy > c_entropy:
                 winner = "quantum"
                 verdict = "Performance NIST equivalente, com ligeira vantagem de entropia para a Geração Quântica."
            else:
                 winner = "classical"
                 verdict = "Performance NIST equivalente, com ligeira vantagem de entropia para a Geração Clássica."

        return {
            "quantum_score": quantum_passes,
            "classical_score": classical_passes,
            "total_tests": total_tests,
            "winner": winner,
            "verdict": verdict,
            "quantum_entropy": q_entropy,
            "classical_entropy": c_entropy,
            "quantum_balance": q_report.get("frequency", {}).get("balance_score", 0),
            "classical_balance": c_report.get("frequency", {}).get("balance_score", 0)
        }
    
    def overlapping_template_test(self, bits: str, template: str = "101010101") -> dict:
        """Overlapping Template Matching Test"""
        import math
        from scipy.special import gammainc
        
        n = len(bits)
        m = len(template)
        M = 1032  # Block length
        N = n // M
        
        if N == 0:
            return {'error': 'Insufficient data'}
        
        # Count overlapping matches in each block
        W = []
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            count = 0
            for j in range(len(block) - m + 1):
                if block[j:j+m] == template:
                    count += 1
            W.append(count)
        
        # Calculate chi-square statistic
        lambda_val = (M - m + 1) / (2**m)
        eta = lambda_val / 2
        
        pi = [math.exp(-lambda_val)]  # P(W=0)
        for i in range(1, 5):
            pi.append(math.exp(-lambda_val) * (lambda_val**i) / math.factorial(i))
        pi.append(1 - sum(pi))  # P(W>=5)
        
        observed = [0] * 6
        for w in W:
            observed[min(w, 5)] += 1
        
        chi_square = sum((observed[i] - N*pi[i])**2 / (N*pi[i]) for i in range(6))
        p_value = 1 - gammainc(5/2, chi_square/2)
        
        return {
            'test_name': 'Overlapping Template',
            'statistic': chi_square,
            'p_value': p_value,
            'passes': p_value >= 0.01,
            'template': template,
            'observed_frequencies': observed
        }
    
    def kolmogorov_complexity_estimate(self, bits: str) -> dict:
        """Estimativa de complexidade de Kolmogorov via compressão"""
        import zlib, bz2, lzma
        
        data = bytes(int(bits[i:i+8], 2) for i in range(0, len(bits)-7, 8))
        original_size = len(data)
        
        if original_size == 0:
            return {'error': 'No data'}
        
        # Diferentes algoritmos de compressão
        zlib_size = len(zlib.compress(data, level=9))
        bz2_size = len(bz2.compress(data, compresslevel=9))
        lzma_size = len(lzma.compress(data, preset=9))
        
        return {
            'original_size': original_size,
            'zlib_ratio': zlib_size / original_size,
            'bz2_ratio': bz2_size / original_size,
            'lzma_ratio': lzma_size / original_size,
            'best_compression': min(zlib_size, bz2_size, lzma_size) / original_size,
            'complexity_score': 1 - min(zlib_size, bz2_size, lzma_size) / original_size
        }
    
    def linear_complexity_test(self, bits: str) -> dict:
        """Teste de Complexidade Linear (NIST)"""
        n = len(bits)
        if n < 500:
            return {'error': 'Insufficient bits for linear complexity test'}
        
        M = 500  # Block size
        N = n // M
        
        def berlekamp_massey(sequence):
            """Algoritmo Berlekamp-Massey para encontrar LFSR mínimo"""
            n = len(sequence)
            b = [0] * n
            c = [0] * n
            b[0] = c[0] = 1
            l = 0
            m = -1
            
            for i in range(n):
                d = sequence[i]
                for j in range(1, l + 1):
                    d ^= c[j] & sequence[i - j]
                
                if d:
                    t = c[:]
                    for j in range(i - m, n):
                        if j < n:
                            c[j] ^= b[j - (i - m)]
                    
                    if 2 * l <= i:
                        l = i + 1 - l
                        m = i
                        b = t
            
            return l
        
        complexities = []
        for i in range(N):
            block = [int(b) for b in bits[i*M:(i+1)*M]]
            complexity = berlekamp_massey(block)
            complexities.append(complexity)
        
        # Statistical analysis
        expected = M/2 + (9 + (-1)**(M+1))/36 - (M/3 + 2/9)/2**M
        variance = M/12 + 7/18 - (M/6 + 1/2)/2**(M-1) + (M/3 + 2/9)/2**M
        
        chi_square = sum((L - expected)**2 / variance for L in complexities)
        degrees_freedom = N - 1
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_freedom)
        
        return {
            'test_name': 'Linear Complexity',
            'complexities': complexities,
            'average_complexity': np.mean(complexities),
            'expected_complexity': expected,
            'chi_square': chi_square,
            'p_value': p_value,
            'passes': p_value >= 0.01
        }