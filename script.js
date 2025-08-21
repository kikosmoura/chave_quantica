class QRNGComparator {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        this.form = document.getElementById('analysisForm');
        this.textInput = document.getElementById('textInput');
        this.charCount = document.getElementById('charCount');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.loading = document.getElementById('loading');
        this.results = document.getElementById('results');
    }

    attachEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.textInput.addEventListener('input', () => this.updateCharCount());
    }

    updateCharCount() {
        const count = this.textInput.value.length;
        this.charCount.textContent = count;
        
        if (count > 25) {
            this.charCount.style.color = '#e74c3c';
        } else if (count > 20) {
            this.charCount.style.color = '#f39c12';
        } else {
            this.charCount.style.color = '#666';
        }
    }

    async handleSubmit(e) {
        e.preventDefault();
        
        const text = this.textInput.value.trim();
        if (!text) {
            alert('Por favor, digite um texto para análise.');
            return;
        }

        this.showLoading();
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }

            const data = await response.json();
            this.displayResults(data);
        } catch (error) {
            console.error('Erro na análise:', error);
            alert('Erro ao processar a análise. Tente novamente.');
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analisando...';
        this.loading.classList.remove('hidden');
        this.results.classList.add('hidden');
    }

    hideLoading() {
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.innerHTML = '<i class="fas fa-analysis"></i> Analisar';
        this.loading.classList.add('hidden');
    }

    displayResults(data) {
        this.displaySummary(data.comparison_score);
        this.displayEncryption(data);
        this.displayMetrics(data.quality_report, data.comparison_score);
        this.displayNISTTests(data.nist_tests);
        this.displayAIAnalysis(data.ai_analysis);  // Nova linha
        
        this.results.classList.remove('hidden');
        this.results.scrollIntoView({ behavior: 'smooth' });
    }

    displayAIAnalysis(aiAnalysis) {
        const aiContent = document.getElementById('aiAnalysis');
        
        if (aiAnalysis && aiAnalysis.trim()) {
            // Converter quebras de linha em parágrafos
            const paragraphs = aiAnalysis.split('\n\n').filter(p => p.trim());
            const formattedText = paragraphs.map(p => `<p>${p.trim()}</p>`).join('');
            aiContent.innerHTML = formattedText;
        } else {
            aiContent.innerHTML = '<p><em>Análise de IA não disponível.</em></p>';
        }
    }

    displaySummary(score) {
        const winnerDisplay = document.getElementById('winnerDisplay');
        const quantumScore = document.getElementById('quantumScore');
        const classicalScore = document.getElementById('classicalScore');

        // Atualizar scores
        quantumScore.textContent = `${score.quantum_score}/${score.total_tests}`;
        classicalScore.textContent = `${score.classical_score}/${score.total_tests}`;

        // Determinar vencedor e estilo
        winnerDisplay.textContent = score.verdict;
        winnerDisplay.className = `winner-display ${score.winner}`;

        // Adicionar ícone baseado no resultado
        let icon = '';
        if (score.winner === 'quantum') {
            icon = '<i class="fas fa-atom"></i> ';
        } else if (score.winner === 'classical') {
            icon = '<i class="fas fa-desktop"></i> ';
        } else {
            icon = '<i class="fas fa-equals"></i> ';
        }
        
        winnerDisplay.innerHTML = icon + score.verdict;
    }

    displayEncryption(data) {
        document.getElementById('quantumKey').textContent = this.truncateString(data.quantum_key, 100);
        document.getElementById('classicalKey').textContent = this.truncateString(data.classical_key, 100);
        document.getElementById('quantumCipher').textContent = this.truncateString(data.encrypted_quantum, 100);
        document.getElementById('classicalCipher').textContent = this.truncateString(data.encrypted_classical, 100);
    }

    displayMetrics(qualityReport, comparisonScore) {
        // Entropia
        document.getElementById('quantumEntropy').textContent = 
            comparisonScore.quantum_entropy.toFixed(6);
        document.getElementById('classicalEntropy').textContent = 
            comparisonScore.classical_entropy.toFixed(6);

        // Balance
        document.getElementById('quantumBalance').textContent = 
            comparisonScore.quantum_balance.toFixed(2) + '%';
        document.getElementById('classicalBalance').textContent = 
            comparisonScore.classical_balance.toFixed(2) + '%';

        // Autocorrelação
        const quantumCorr = qualityReport.quantum.advanced_analysis.autocorrelation.max_autocorr;
        const classicalCorr = qualityReport.classical.advanced_analysis.autocorrelation.max_autocorr;
        
        document.getElementById('quantumCorr').textContent = quantumCorr.toFixed(6);
        document.getElementById('classicalCorr').textContent = classicalCorr.toFixed(6);

        // Compressão
        const quantumComp = qualityReport.quantum.advanced_analysis.compression.compression_ratio;
        const classicalComp = qualityReport.classical.advanced_analysis.compression.compression_ratio;
        
        document.getElementById('quantumCompression').textContent = quantumComp.toFixed(6);
        document.getElementById('classicalCompression').textContent = classicalComp.toFixed(6);
    }

    displayNISTTests(nistTests) {
        const testGrid = document.getElementById('nistTestGrid');
        testGrid.innerHTML = '';

        const tests = [
            { key: 'monobit', name: 'Monobit Frequency' },
            { key: 'runs', name: 'Runs Test' },
            { key: 'spectral', name: 'Spectral (DFT)' },
            { key: 'template', name: 'Template Matching' },
            { key: 'linear_complexity', name: 'Linear Complexity' }
        ];

        tests.forEach(test => {
            const testItem = document.createElement('div');
            testItem.className = 'test-item';

            const quantumResult = nistTests.quantum[test.key];
            const classicalResult = nistTests.classical[test.key];

            testItem.innerHTML = `
                <h4>${test.name}</h4>
                <div class="test-results">
                    <div class="test-result ${quantumResult.passes ? 'pass' : 'fail'}">
                        Quântico: ${quantumResult.passes ? 'PASSOU' : 'FALHOU'}
                        <br><small>p-value: ${quantumResult.p_value?.toFixed(6) || 'N/A'}</small>
                    </div>
                    <div class="test-result ${classicalResult.passes ? 'pass' : 'fail'}">
                        Clássico: ${classicalResult.passes ? 'PASSOU' : 'FALHOU'}
                        <br><small>p-value: ${classicalResult.p_value?.toFixed(6) || 'N/A'}</small>
                    </div>
                </div>
            `;

            testGrid.appendChild(testItem);
        });
    }

    truncateString(str, maxLength) {
        if (str.length <= maxLength) return str;
        return str.substring(0, maxLength) + '...';
    }
}

// Inicializar quando a página carregar
document.addEventListener('DOMContentLoaded', () => {
    new QRNGComparator();
});