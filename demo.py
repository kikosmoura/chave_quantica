from src.qrng_generator import QuantumRandomGenerator
from src.classical_rng import ClassicalRandomGenerator
from src.otp_cipher import OneTimePad
from src.quality_analyzer import RandomnessAnalyzer
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
    print("=== POC: Geração Quântica de Números Aleatórios ===\n")
    
    # Inicializar geradores
    qrng = QuantumRandomGenerator()
    crng = ClassicalRandomGenerator()
    analyzer = RandomnessAnalyzer()
    
    # Mensagem de teste
    #message = "SERPRO QUANTUM SECURITY TEST"
    message = "a"
    key_length = len(message) * 8  # 8 bits por caractere
    
    print(f"Mensagem original: {message}")
    print(f"Tamanho da chave necessária: {key_length} bits\n")
    
    # Gerar chaves
    print("Gerando chaves...")
    quantum_key = qrng.generate_random_bits(key_length)
    classical_key = crng.generate_random_bits(key_length)
    
    print(f"Chave quântica (primeiros 50 bits): {quantum_key[:50]}...")
    print(f"Chave clássica (primeiros 50 bits): {classical_key[:50]}...\n")
    
    # Cifrar e decifrar com OTP
    print("=== Cifragem com One-Time Pad ===")
    
    # Com chave quântica
    encrypted_quantum = OneTimePad.encrypt(message, quantum_key)
    decrypted_quantum = OneTimePad.decrypt(encrypted_quantum, quantum_key)
    
    # Com chave clássica
    encrypted_classical = OneTimePad.encrypt(message, classical_key)
    decrypted_classical = OneTimePad.decrypt(encrypted_classical, classical_key)
    
    print(f"Cifrado quântico: {encrypted_quantum[:50]}...")
    print(f"Decifrado quântico: {decrypted_quantum}")
    print(f"Cifrado clássico: {encrypted_classical[:50]}...")
    print(f"Decifrado clássico: {decrypted_classical}\n")
    
    # Análise de qualidade
    print("=== Análise de Qualidade das Chaves ===")
    
    # Gerar amostras maiores para análise detalhada
    large_quantum = qrng.generate_random_bits(10000)  # Aumentado para análise robusta
    large_classical = crng.generate_random_bits(10000)
    
    report = analyzer.generate_report(large_quantum, large_classical)
    
    # Exibir análise básica
    print("=== Análise Básica ===")
    print("Análise Quântica:")
    print(f"  Entropia: {report['quantum']['basic_stats']['entropy']:.6f}")
    print(f"  Balance (0s/1s): {report['quantum']['basic_stats']['frequency']['balance_score']:.4f}%")
    print(f"  Runs médias: {report['quantum']['basic_stats']['runs']['average_run_length']:.4f}")
    print(f"  Runs longas (>5): {report['quantum']['basic_stats']['runs']['long_runs_count']}")
    print(f"  Runs muito longas (>10): {report['quantum']['basic_stats']['runs']['very_long_runs_count']}")
    
    print("\nAnálise Clássica:")
    print(f"  Entropia: {report['classical']['basic_stats']['entropy']:.6f}")
    print(f"  Balance (0s/1s): {report['classical']['basic_stats']['frequency']['balance_score']:.4f}%")
    print(f"  Runs médias: {report['classical']['basic_stats']['runs']['average_run_length']:.4f}")
    print(f"  Runs longas (>5): {report['classical']['basic_stats']['runs']['long_runs_count']}")
    print(f"  Runs muito longas (>10): {report['classical']['basic_stats']['runs']['very_long_runs_count']}")
    
    # Exibir análise avançada
    print("\n=== Análise Avançada ===")
    print("Autocorrelação Quântica:")
    print(f"  Max correlação: {report['quantum']['advanced_analysis']['autocorrelation']['max_autocorr']:.6f}")
    print(f"  Correlações significativas: {report['quantum']['advanced_analysis']['autocorrelation']['significant_correlations']}")
    
    print("\nAutocorrelação Clássica:")
    print(f"  Max correlação: {report['classical']['advanced_analysis']['autocorrelation']['max_autocorr']:.6f}")
    print(f"  Correlações significativas: {report['classical']['advanced_analysis']['autocorrelation']['significant_correlations']}")
    
    print("\nTeste Qui-quadrado (8 bits):")
    print(f"  Quântico - p-value: {report['quantum']['advanced_analysis']['chi_square_8bit']['p_value']:.6f}")
    print(f"  Clássico - p-value: {report['classical']['advanced_analysis']['chi_square_8bit']['p_value']:.6f}")
    
    print("\nTeste de Compressão:")
    print(f"  Quântico - ratio: {report['quantum']['advanced_analysis']['compression']['compression_ratio']:.6f}")
    print(f"  Clássico - ratio: {report['classical']['advanced_analysis']['compression']['compression_ratio']:.6f}")
    
    print("\nAnálise de Padrões (3-bit):")
    print(f"  Quântico - padrões únicos: {report['quantum']['advanced_analysis']['pattern_analysis']['unique_3bit_patterns']}")
    print(f"  Clássico - padrões únicos: {report['classical']['advanced_analysis']['pattern_analysis']['unique_3bit_patterns']}")
    
    print("\nEntropia por Blocos (8-bit):")
    print(f"  Quântico - média: {report['quantum']['advanced_analysis']['block_entropy_8bit']['average_entropy']:.6f}")
    print(f"  Clássico - média: {report['classical']['advanced_analysis']['block_entropy_8bit']['average_entropy']:.6f}")
    
    # Salvar relatório
    with open('quality_report.json', 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nRelatório completo salvo em 'quality_report.json'")
    
    # Análise expandida com testes NIST
    print("\n=== Testes NIST SP 800-22 ===")
    
    # Testes NIST para amostras quânticas
    nist_mono_q = analyzer.nist_monobit_test(large_quantum)
    nist_runs_q = analyzer.nist_runs_test(large_quantum)
    spectral_q = analyzer.spectral_test(large_quantum)
    template_q = analyzer.overlapping_template_test(large_quantum)
    complexity_q = analyzer.kolmogorov_complexity_estimate(large_quantum)
    linear_comp_q = analyzer.linear_complexity_test(large_quantum)
    
    # Testes NIST para amostras clássicas
    nist_mono_c = analyzer.nist_monobit_test(large_classical)
    nist_runs_c = analyzer.nist_runs_test(large_classical)
    spectral_c = analyzer.spectral_test(large_classical)
    template_c = analyzer.overlapping_template_test(large_classical)
    complexity_c = analyzer.kolmogorov_complexity_estimate(large_classical)
    linear_comp_c = analyzer.linear_complexity_test(large_classical)
    
    print("NIST Monobit Test:")
    print(f"  Quântico - p-value: {nist_mono_q['p_value']:.6f} | Pass: {nist_mono_q['passes']}")
    print(f"  Clássico - p-value: {nist_mono_c['p_value']:.6f} | Pass: {nist_mono_c['passes']}")
    
    print("\nNIST Runs Test:")
    print(f"  Quântico - p-value: {nist_runs_q['p_value']:.6f} | Pass: {nist_runs_q['passes']}")
    print(f"  Clássico - p-value: {nist_runs_c['p_value']:.6f} | Pass: {nist_runs_c['passes']}")
    
    print("\nSpectral Test (DFT):")
    print(f"  Quântico - p-value: {spectral_q['p_value']:.6f} | Pass: {spectral_q['passes']}")
    print(f"  Clássico - p-value: {spectral_c['p_value']:.6f} | Pass: {spectral_c['passes']}")
    
    print("\nComplexidade de Kolmogorov:")
    print(f"  Quântico - Score: {complexity_q['complexity_score']:.6f} | Melhor ratio: {complexity_q['best_compression']:.6f}")
    print(f"  Clássico - Score: {complexity_c['complexity_score']:.6f} | Melhor ratio: {complexity_c['best_compression']:.6f}")
    
    print("\nComplexidade Linear:")
    print(f"  Quântico - Média: {linear_comp_q['average_complexity']:.2f} | p-value: {linear_comp_q['p_value']:.6f}")
    print(f"  Clássico - Média: {linear_comp_c['average_complexity']:.2f} | p-value: {linear_comp_c['p_value']:.6f}")
    
    # Score comparativo
    # Reúne os resultados dos testes NIST em um dicionário para o score
    nist_results = {
        "quantum": {
            "monobit": nist_mono_q,
            "runs": nist_runs_q,
            "spectral": spectral_q,
            "template": template_q,
            "complexity": complexity_q,
            "linear_complexity": linear_comp_q,
        },
        "classical": {
            "monobit": nist_mono_c,
            "runs": nist_runs_c,
            "spectral": spectral_c,
            "template": template_c,
            "complexity": complexity_c,
            "linear_complexity": linear_comp_c,
        }
    }

    print("\n=== Score Comparativo ===")
    comparison_score = analyzer.calculate_comparison_score(report, nist_results)
    print(f"Score Quântico: {comparison_score['quantum_score']}/{comparison_score['total_tests']} testes aprovados")
    print(f"Score Clássico: {comparison_score['classical_score']}/{comparison_score['total_tests']} testes aprovados")
    print(f"Veredito: {comparison_score['verdict']}")

if __name__ == "__main__":
    main()