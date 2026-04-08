# SPEC — insights-extract

> Contrato técnico do extractor. Se algo neste documento mudar, **bumpa a versão e atualiza CLAUDE.md**. Esta é a fonte da verdade do que o script faz.

---

## 1. Objetivo

Transformar **um vídeo** (URL do YouTube ou arquivo local) em **um JSON estruturado** que ajude um profissional técnico a decidir, em menos de 30 segundos, se vale a pena assistir o vídeo inteiro — e, se não valer, extrair os insights úteis sem ter assistido.

**Não** é resumo. **É** decisão suportada por evidência.

## 2. Input

### 2.1 Tipos aceitos

| Tipo | Como | Validação |
|---|---|---|
| URL do YouTube | string começando com `https://www.youtube.com/`, `https://youtu.be/`, `https://m.youtube.com/` | regex simples no `extract.py` |
| Arquivo de vídeo local | path para `.mp4`, `.mkv`, `.webm`, `.mov`, `.avi` | `Path.exists()` + extensão na whitelist |
| Arquivo de áudio local | path para `.wav`, `.mp3`, `.m4a`, `.flac` | mesmo critério |

### 2.2 Limites

- Vídeo > 60 minutos: aviso, mas processa
- Vídeo > 180 minutos: erro, recusa
- Sem áudio detectado: erro com mensagem clara
- Idioma: assume PT-BR ou EN; outros idiomas funcionam mas qualidade de output não é garantida

## 3. Output — Schema do JSON (contrato público)

Validado por Pydantic. Esta é a estrutura **exata** que o LLM tem que devolver. Versão `1.0.0`.

```python
class Insight(BaseModel):
    schema_version: Literal["1.0.0"]
    source: SourceInfo
    decision: Decision
    key_concepts: list[KeyConcept]      # min 3, max 5
    architectural_risks: list[str]      # min 0, max 5 (vazio é válido)
    open_questions: list[str]           # min 1, max 5
    actionable_items: list[str]         # min 0, max 7 (vazio é válido)
    metadata: Metadata

class SourceInfo(BaseModel):
    type: Literal["youtube", "local_video", "local_audio"]
    url_or_path: str
    title: str | None
    duration_seconds: int

class Decision(BaseModel):
    watch_full: bool
    confidence: Literal["low", "medium", "high"]
    rationale: str                       # max 280 chars (caber num tweet)

class KeyConcept(BaseModel):
    name: str                            # max 60 chars
    explanation: str                     # max 240 chars
    timestamp_seconds: int | None        # opcional, se whisper devolveu segments

class Metadata(BaseModel):
    extracted_at: datetime
    transcription_model: str             # ex: "whisper-base"
    llm_model: str                       # ex: "qwen2.5:7b"
    transcription_duration_seconds: float
    llm_duration_seconds: float
    language_detected: str               # iso 639-1
```

### 3.1 Por que esses campos e não outros

| Campo | Razão |
|---|---|
| `decision.watch_full` | É o ponto do projeto. Se o output não responde isso, perdeu o propósito. |
| `decision.rationale` (max 280) | Forçar concisão. Se cabe num tweet, é uma decisão real, não uma desculpa. |
| `key_concepts` (3-5) | Menos que 3 = vídeo trivial. Mais que 5 = não é insight, é resumo disfarçado. |
| `architectural_risks` (0-5, opcional) | Lente de Arquitetura. Vazio se vídeo não trata disso (ex: vídeo de cooking). |
| `open_questions` (1-5) | Lente de QA. Sempre tem pelo menos 1 — bom QA sempre encontra algo a perguntar. |
| `actionable_items` (0-7) | Lente de prática. Distingue conteúdo "consumível" de "executável". |
| `metadata` | Rastreabilidade — qual modelo, quanto tempo. Necessário pra debug e pra benchmark. |

## 4. Estratégia de prompt

### 4.1 Estrutura sandwich

```
[SYSTEM PROMPT — papel + idioma + restrições]

[CONTEXTO — transcrição completa do vídeo]

[INSTRUÇÃO FINAL — formato exato do output]
[SCHEMA JSON — em forma de exemplo]
```

A instrução crítica e o schema vão **no final**, perto do ponto de geração. Modelos pequenos respeitam mais quando a instrução está adjacente à resposta esperada.

### 4.2 System prompt base (PT-BR)

```
Você é um assistente técnico especializado em extrair insights estruturados
de conteúdo educacional e técnico. Você responde SEMPRE em português
brasileiro, é técnico e preciso, e usa terminologia correta da área quando
identificada. Você SEMPRE devolve um único bloco JSON válido, sem texto
antes ou depois. Você nunca inventa informação que não está na transcrição.
Quando não tiver evidência suficiente para um campo, deixe a lista vazia
ou marque a confiança como "low".
```

### 4.3 Instrução final

A instrução final inclui o schema JSON literal como exemplo preenchido com placeholders explícitos (`<string max 60 chars>`, `<int 0..7>`). Modelos 7B respondem melhor a exemplos que a descrições.

## 5. Pipeline de execução

```
input
  ↓
detect input type (URL ou path)
  ↓
[se URL] yt-dlp baixa áudio → wav 16kHz mono
[se path] valida e converte se necessário
  ↓
whisper transcribe → texto + segments (com timestamps)
  ↓
build prompt (sandwich) com transcript + schema
  ↓
call_ollama com retry (max 2) e validação Pydantic
  ↓
attach metadata (durations, models, language)
  ↓
write JSON to stdout ou --output
```

## 6. Tratamento de erros

| Erro | Comportamento |
|---|---|
| URL inválida | Sai com código 2 + mensagem `Invalid URL or file path` |
| Vídeo > 180min | Sai com código 3 + mensagem com duração detectada |
| Whisper falha | Sai com código 4 + erro do whisper na stderr |
| LLM timeout (>120s) | Sai com código 5 + sugestão de modelo menor |
| LLM devolve JSON inválido após 2 retries | Sai com código 6 + dump do output bruto pra debug |
| Ollama não responde | Sai com código 7 + comando para iniciar Ollama |

Códigos de saída são parte do contrato. Scripts upstream podem capturar.

## 7. Critérios de aceitação para considerar episódio 1 "pronto"

- [ ] `python -m src.extract <url-publica-de-5min>` roda em <90s no M4
- [ ] Output valida no schema versão 1.0.0
- [ ] `decision.watch_full` corresponde à intuição humana em pelo menos 4 de 5 vídeos de teste
- [ ] `examples/output.json` existe e é gerado pelo próprio script
- [ ] README quick-start funciona em máquina limpa (testar em VM ou container)
- [ ] Nenhum dado privado ou referência a MBA no repo
- [ ] LICENSE MIT no root, copyright correto
- [ ] CLAUDE.md, SPEC.md, README.md alinhados (sem contradição)

## 8. Fora de escopo (deixar pro próximo episódio)

Tudo isso é tentação que tem que ficar de fora do episódio 01:

- Persistir múltiplos vídeos numa base
- Buscar entre múltiplos vídeos
- Embeddings ou retrieval (vai pro episódio 02)
- UI / web / API server
- Suporte a múltiplos LLMs simultâneos
- Fine-tuning ou few-shot avançado
- Cache de transcrições (yt-dlp já cacheia download)
- Agendamento / automação

Se aparecer urgência de adicionar qualquer um destes, **escreva um issue em vez de codar**. O escopo apertado é a feature, não o defeito.

## 9. Exemplos de output esperado

### Vídeo 1: palestra técnica de 8 minutos sobre Hexagonal Architecture

```json
{
  "schema_version": "1.0.0",
  "source": {
    "type": "youtube",
    "url_or_path": "https://www.youtube.com/watch?v=EXEMPLO",
    "title": "Hexagonal Architecture in 8 minutes",
    "duration_seconds": 480
  },
  "decision": {
    "watch_full": true,
    "confidence": "high",
    "rationale": "Vídeo curto, denso, com exemplos de código reais. Vale assistir se você quer entender ports & adapters sem teoria abstrata."
  },
  "key_concepts": [
    {
      "name": "Ports and Adapters",
      "explanation": "Separação entre lógica de negócio (core) e detalhes de infraestrutura via interfaces (ports) implementadas por adaptadores externos.",
      "timestamp_seconds": 45
    }
  ],
  "architectural_risks": [
    "Risco de over-engineering quando aplicado a CRUDs simples",
    "Curva de aprendizado alta para times sem experiência prévia em DDD"
  ],
  "open_questions": [
    "Como aplicar isso em projetos legados sem big bang refactor?",
    "Qual a granularidade ideal de adapters em microserviços?"
  ],
  "actionable_items": [
    "Identificar 1 service do seu projeto atual e mapear quais são os ports",
    "Escrever um adapter de teste em memória para um repository existente"
  ],
  "metadata": {
    "extracted_at": "2026-04-08T14:23:00Z",
    "transcription_model": "whisper-base",
    "llm_model": "qwen2.5:7b",
    "transcription_duration_seconds": 23.4,
    "llm_duration_seconds": 18.7,
    "language_detected": "en"
  }
}
```

### Vídeo 2: tutorial de 90 minutos sobre framework JavaScript do mês

```json
{
  "decision": {
    "watch_full": false,
    "confidence": "high",
    "rationale": "90 minutos é caro. Os 4 conceitos centrais aparecem nos primeiros 12 minutos. Recomendado: ver até 12:00 e parar."
  }
}
```

(O `decision.watch_full: false` é tão útil quanto o `true`. O ponto é o tempo de quem assiste.)

## 10. Versionamento do schema

- `1.0.0` — versão inicial (este documento)
- Campos novos opcionais → bump minor (`1.1.0`)
- Campos removidos ou tipo alterado → bump major (`2.0.0`)
- Schema version no JSON é obrigatório — clientes downstream podem validar.
