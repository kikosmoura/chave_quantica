from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
import json
import logging
import numpy as np
from pathlib import Path
import os
from google import genai
from google.genai.types import Content, Part

from src.qrng_generator import QuantumRandomGenerator
from src.classical_rng import ClassicalRandomGenerator
from src.otp_cipher import OneTimePad
from src.quality_analyzer import RandomnessAnalyzer

app = FastAPI(title="Quantum vs Classical RNG Comparator", version="1.0.0")

# Configurar logging para capturar informações detalhadas de erros
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=50, description="Texto para gerar chave (máximo 50 caracteres)")

# Instâncias globais
qrng = QuantumRandomGenerator()
crng = ClassicalRandomGenerator()
analyzer = RandomnessAnalyzer()

# Configurar cliente Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "SUA-KEY")
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.warning(f"Não foi possível inicializar o cliente Gemini: {e}")
    gemini_client = None

class AnalysisResult(BaseModel):
    quantum_key: str
    classical_key: str
    encrypted_quantum: str
    encrypted_classical: str
    decrypted_quantum: str
    decrypted_classical: str
    quality_report: Dict[str, Any]
    nist_tests: Dict[str, Any]
    comparison_score: Dict[str, Any]
    ai_analysis: str = ""  # Nova campo

def convert_numpy_types(data: Any) -> Any:
    """Converte recursivamente os tipos de dados do NumPy em uma estrutura de dados para tipos nativos do Python."""
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    return data

async def run_nist_tests(quantum_bits: str, classical_bits: str) -> Dict[str, Any]:
    """Executa testes NIST de forma assíncrona e em paralelo"""
    
    tests_to_run = {
        "monobit": analyzer.nist_monobit_test,
        "runs": analyzer.nist_runs_test,
        "spectral": analyzer.spectral_test,
        "template": analyzer.overlapping_template_test,
        "complexity": analyzer.kolmogorov_complexity_estimate,
        "linear_complexity": analyzer.linear_complexity_test,
    }

    quantum_tasks = [asyncio.to_thread(test_func, quantum_bits) for test_func in tests_to_run.values()]
    classical_tasks = [asyncio.to_thread(test_func, classical_bits) for test_func in tests_to_run.values()]

    quantum_results = await asyncio.gather(*quantum_tasks)
    classical_results = await asyncio.gather(*classical_tasks)
    
    return {
        "quantum": dict(zip(tests_to_run.keys(), quantum_results)),
        "classical": dict(zip(tests_to_run.keys(), classical_results)),
    }

async def generate_ai_analysis(quality_report: Dict, nist_tests: Dict, comparison_score: Dict) -> str:
    """Gera análise com IA dos resultados"""
    if not gemini_client:
        return "Análise de IA não disponível - verifique a configuração da API."
    
    try:
        # Preparar dados para o prompt
        q_entropy = quality_report['quantum']['basic_stats']['entropy']
        c_entropy = quality_report['classical']['basic_stats']['entropy']
        q_balance = quality_report['quantum']['basic_stats']['frequency']['balance_score']
        c_balance = quality_report['classical']['basic_stats']['frequency']['balance_score']
        
        q_score = comparison_score['quantum_score']
        c_score = comparison_score['classical_score']
        total_tests = comparison_score['total_tests']
        verdict = comparison_score['verdict']
        
        prompt = f"""
        Analise os seguintes resultados de comparação entre geradores de números aleatórios quânticos (QRNG) e clássicos (CRNG):

        **Métricas de Qualidade:**
        - Entropia Quântica: {q_entropy:.6f} | Clássica: {c_entropy:.6f}
        - Balance Quântico: {q_balance:.2f}% | Clássico: {c_balance:.2f}%
        
        **Resultados dos Testes:**
        - Score Quântico: {q_score}/{total_tests} testes aprovados
        - Score Clássico: {c_score}/{total_tests} testes aprovados
        - Veredito: {verdict}
        
        **Testes NIST:**
        - Monobit Quântico: {'PASSOU' if nist_tests['quantum']['monobit']['passes'] else 'FALHOU'}
        - Monobit Clássico: {'PASSOU' if nist_tests['classical']['monobit']['passes'] else 'FALHOU'}
        - Runs Quântico: {'PASSOU' if nist_tests['quantum']['runs']['passes'] else 'FALHOU'}
        - Runs Clássico: {'PASSOU' if nist_tests['classical']['runs']['passes'] else 'FALHOU'}

        Forneça uma análise técnica CONCISA (máximo 3 parágrafos) explicando:
        1. Qual gerador teve melhor performance e por quê
        2. Analise as métricas principais (entropia, balance) e as métricas dos testes NIST SP 800-22
        3. Conclusão prática sobre a qualidade dos números aleatórios gerados
        
        Use linguagem técnica mas acessível. Seja direto e objetivo.
        """
        
        msg = Content(role="user", parts=[Part(text=prompt)])
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.0-flash-exp",
            contents=[msg]
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Erro na análise de IA: {e}")
        return f"Erro ao gerar análise de IA: {str(e)}"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Página principal"""
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>QRNG vs CRNG</title></head>
    <body><h1>Arquivo index.html não encontrado</h1></body>
    </html>
    """, status_code=404)

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_text(input_data: TextInput):
    """Analisa o texto e gera comparação entre QRNG e CRNG"""
    try:
        text = input_data.text
        key_length = len(text) * 8  # 8 bits por caractere
        
        # Gerar chaves e amostras em paralelo para melhor performance
        key_gen_tasks = [
            asyncio.to_thread(qrng.generate_random_bits, key_length),
            asyncio.to_thread(crng.generate_random_bits, key_length),
            asyncio.to_thread(qrng.generate_random_bits, 10000), # Amostra maior para testes
            asyncio.to_thread(crng.generate_random_bits, 10000)
        ]
        quantum_key, classical_key, large_quantum, large_classical = await asyncio.gather(*key_gen_tasks)
        
        # Criptografia OTP
        encrypted_quantum = OneTimePad.encrypt(text, quantum_key)
        decrypted_quantum = OneTimePad.decrypt(encrypted_quantum, quantum_key)
        encrypted_classical = OneTimePad.encrypt(text, classical_key)
        decrypted_classical = OneTimePad.decrypt(encrypted_classical, classical_key)
        
        # Análise de qualidade
        quality_report = await asyncio.to_thread(
            analyzer.generate_report, large_quantum, large_classical
        )
        
        # Testes NIST
        nist_tests = await run_nist_tests(large_quantum, large_classical)
        
        # Score comparativo
        comparison_score = analyzer.calculate_comparison_score(quality_report, nist_tests)
        
        # Gerar análise de IA
        ai_analysis = await generate_ai_analysis(quality_report, nist_tests, comparison_score)
        
        # Converter tipos numpy para tipos nativos do Python antes de retornar
        quality_report = convert_numpy_types(quality_report)
        nist_tests = convert_numpy_types(nist_tests)
        comparison_score = convert_numpy_types(comparison_score)

        return AnalysisResult(
            quantum_key=quantum_key,
            classical_key=classical_key,
            encrypted_quantum=encrypted_quantum,
            encrypted_classical=encrypted_classical,
            decrypted_quantum=decrypted_quantum,
            decrypted_classical=decrypted_classical,
            quality_report=quality_report,
            nist_tests=nist_tests,
            comparison_score=comparison_score,
            ai_analysis=ai_analysis
        )
        
    except Exception as e:
        logger.exception("Ocorreu um erro durante a análise do texto.")
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "QRNG vs CRNG API está funcionando"}

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
