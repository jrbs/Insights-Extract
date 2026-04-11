"""Prompt templates for insight extraction.

Keeping prompts in a separate module means:
- Prompt changes show up cleanly in git diffs (reviewable like code)
- The "contrato do pedido" (what we ask the LLM) is visível de relance
- Ajustes de idioma, tom, ou schema não precisam mexer no pipeline
- Futuras variantes (A/B, idioma, domínio) ficam lado a lado

Estrutura do sandwich prompt (dados → reforço → instrução):

    1. SYSTEM_PREAMBLE       — quem você é, para quem serve, honestidade
    2. [TRANSCRIPTION]       — os dados brutos (maior volume)
    3. LANGUAGE_ENFORCEMENT  — reforço final do idioma (PT-BR obrigatório)
    4. [INSTRUCTION] header  — marca o ponto onde o modelo deve agir
    5. SCHEMA_TEMPLATE       — o formato JSON esperado
    6. CRITICAL_RULES        — regras por lente de leitura
    7. OUTPUT_FORMAT         — "only JSON, nothing else"

Modelos pequenos (7B) respondem melhor quando a instrução crítica vem no
fim (depois dos dados). Por isso tanto o reforço de idioma quanto as regras
por lente ficam no final, não só no preamble.
"""


SYSTEM_PREAMBLE = """Você é um assistente que extrai insights estruturados de transcrições de
vídeo para profissionais de TI (devs, QA, arquitetos, SREs, PMs técnicos).
O vídeo de entrada pode ser sobre qualquer assunto — tech, negócio, cultura,
gestão, educação. Seu trabalho é transformar a transcrição em um JSON útil
para alguém que lê com olho crítico e analítico: identificar a tese central,
os conceitos que sustentam, o que está implícito ou não comprovado, e o que
fica em aberto.

Você SEMPRE responde em português brasileiro, é preciso, e usa a terminologia
do domínio identificado no vídeo (se é culinária, fale de culinária; se é
finanças, fale de finanças). Você SEMPRE devolve um único bloco JSON válido,
sem texto antes ou depois. Você nunca inventa informação que não está na
transcrição. Quando não tiver evidência suficiente para um campo opcional,
deixe a lista vazia; para confidence, use "low"."""


LANGUAGE_ENFORCEMENT = """[IDIOMA — REGRA OBRIGATÓRIA]
TODOS os campos textuais do JSON DEVEM estar em português brasileiro (pt-BR),
INDEPENDENTE do idioma original do vídeo. Se o vídeo estiver em inglês,
espanhol, ou qualquer outro idioma, TRADUZA os conceitos, o resumo, a tese,
as ressalvas, as perguntas em aberto e as ações práticas para português.

Exceções (NÃO traduzir):
- notable_quotes: copie verbatim no idioma original do falante. Se ele disse
  em inglês, mantenha em inglês entre aspas. Citação é prova, não tradução.
- Valores técnicos do schema: schema_version, source.type, decision.confidence
  (low/medium/high), metadata.language_detected (código ISO 639-1).
- Nomes próprios, produtos, frameworks, termos técnicos consagrados
  (ex: "React", "Kubernetes", "Pydantic") — mantenha no original."""


SCHEMA_TEMPLATE = """{
  "schema_version": "1.1.0",
  "source": {
    "type": "<youtube|local_video|local_audio>",
    "url_or_path": "<input source>",
    "title": "<title or null>",
    "duration_seconds": <int>
  },
  "decision": {
    "watch_full": <true|false>,
    "confidence": "<low|medium|high>",
    "rationale": "<max 280 chars — cabe num tweet>"
  },
  "summary": "<1 parágrafo introdutório + bullet points com os pontos principais, max 700 chars>",
  "core_thesis": "<1 frase, max 280 chars>",
  "key_concepts": [
    {"name": "<max 60 chars>", "explanation": "<max 240 chars>", "timestamp_seconds": <int|null>},
    ...
  ],
  "caveats": ["<string>", ...],
  "open_questions": ["<string>", ...],
  "actionable_takeaways": ["<string>", ...],
  "notable_quotes": ["<string verbatim>", ...],
  "metadata": {
    "extracted_at": "<ISO datetime>",
    "transcription_model": "whisper-base",
    "llm_model": "qwen2.5:7b",
    "transcription_duration_seconds": <float>,
    "llm_duration_seconds": <float>,
    "language_detected": "<iso 639-1 code>"
  }
}"""


CRITICAL_RULES = """CRITICAL RULES (por lente de leitura):

[SUMMARY — dar a forma antes do detalhe]
- summary: parágrafo introdutório (2-3 frases) seguido de bullet points com
  os pontos principais do vídeo. Formato:
  "Frase introdutória sobre o tema.\n• Ponto 1\n• Ponto 2\n• Ponto 3"
  Max 700 chars total. OBRIGATÓRIO.

[CORE THESIS — a ideia em 1 frase]
- core_thesis: a ÚNICA ideia que o leitor deveria guardar. Se apagar tudo
  menos 1 frase, o que sobra e ainda faz sentido sozinho? Max 280 chars.
  OBRIGATÓRIO.

[KEY CONCEPTS — os conceitos que sustentam a tese]
- key_concepts: OBRIGATÓRIO 3-5 itens. Menos de 3 = conteúdo trivial. Mais
  de 5 = você está resumindo, não extraindo. Cada conceito usa a terminologia
  do domínio do vídeo (não força jargão técnico de TI se o vídeo não é de TI).

[CAVEATS — o que não foi dito ou passou batido]
- caveats: blind spots, suposições não sustentadas por evidência, afirmações
  que precisariam de verificação independente. Lacunas que um leitor crítico
  (com mentalidade de QA / engenheiro) notaria. Lista vazia é OK se o conteúdo
  é bem fundamentado. Max 5 itens.

[OPEN QUESTIONS — o que o vídeo deixa em aberto]
- open_questions: SEMPRE pelo menos 1. Perguntas que um leitor analítico sai
  querendo investigar depois. Max 5.

[ACTIONABLE TAKEAWAYS — o que dá pra aplicar ou compartilhar]
- actionable_takeaways: passos concretos que um leitor pode executar ou enviar
  pra alguém. Itens curtos, cada um cabe numa mensagem. Pode ser vazio se o
  vídeo é puramente conceitual. Max 7 itens.

[NOTABLE QUOTES — frases verbatim]
- notable_quotes: até 3 frases literais da transcrição que merecem citação
  direta. COPIE, não parafraseie. Mantenha no idioma original do falante.
  Opcional.

[DECISION]
- decision.watch_full: vale o tempo assistir o vídeo inteiro? true/false.
- decision.confidence: low = você não tem certeza. medium = dúvida razoável.
  high = padrão claro na transcrição.
- decision.rationale: MUST caber em 280 chars. Se ficar maior, você está
  super-explicando."""


OUTPUT_FORMAT = (
    "Return ONLY the JSON object. No markdown, no code blocks, no explanation. "
    "Lembre: todo conteúdo textual em pt-BR (exceto notable_quotes verbatim)."
)


# Exemplo concreto curto — modelos 7B respondem melhor a exemplos que a placeholders.
# IMPORTANTE: o exemplo usa um domínio completamente diferente (culinária) para
# que o modelo não confunda o exemplo com os dados reais da transcrição.
FEW_SHOT_EXAMPLE = """EXAMPLE (apenas para mostrar o FORMATO do JSON — NÃO copie este conteúdo.
Extraia os insights da transcrição REAL acima, não deste exemplo):
{
  "schema_version": "1.1.0",
  "source": {"type": "youtube", "url_or_path": "https://...", "title": "...", "duration_seconds": 360},
  "decision": {"watch_full": true, "confidence": "medium", "rationale": "Receita prática com técnica transferível, mas assume equipamento específico."},
  "summary": "Chef demonstra técnica de fermentação natural para pães artesanais, comparando tempos de descanso e proporções de hidratação com resultados visuais.\n• Autólise de 30 min desenvolve glúten sem trabalho mecânico\n• Fermentação lenta (12-18h) gera sabor mais complexo que a rápida\n• Hidratação acima de 70% exige técnica de manuseio diferente",
  "core_thesis": "Fermentação longa com menos fermento produz pães com sabor mais complexo e melhor textura do que fermentação rápida com mais fermento.",
  "key_concepts": [
    {"name": "Hidratação da massa", "explanation": "Proporção de água em relação à farinha, expressa em porcentagem. Acima de 70% exige técnica diferente de manuseio.", "timestamp_seconds": 120},
    {"name": "Autólise", "explanation": "Descanso inicial só com farinha e água antes de adicionar sal e fermento, para desenvolver glúten sem trabalho mecânico.", "timestamp_seconds": 45},
    {"name": "Fermentação lenta", "explanation": "Processo de 12-18h em geladeira que desenvolve ácidos orgânicos responsáveis pelo sabor complexo do pão.", "timestamp_seconds": 200}
  ],
  "caveats": ["Não menciona variações de clima e temperatura ambiente que afetam o tempo de fermentação."],
  "open_questions": ["Como adaptar os tempos para farinhas integrais que absorvem mais água?"],
  "actionable_takeaways": ["Testar autólise de 30 min antes de adicionar o fermento na próxima receita."],
  "notable_quotes": ["A pressa é inimiga do glúten."],
  "metadata": {"extracted_at": "2026-04-08T14:23:00Z", "transcription_model": "whisper-small", "llm_model": "qwen2.5:7b", "transcription_duration_seconds": 23.4, "llm_duration_seconds": 18.7, "language_detected": "pt"}
}"""


# Limite de caracteres para caber no context window de modelos 7B (32k tokens)
MAX_TRANSCRIPT_CHARS = 15_000


def _format_timestamp(seconds: float) -> str:
    """Format seconds as [MM:SS]."""
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"


def _format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Format transcript segments as timestamped lines."""
    lines = []
    for seg in segments:
        ts = _format_timestamp(seg["start"])
        lines.append(f"{ts} {seg['text']}")
    return "\n".join(lines)


def build_prompt(
    transcript: str,
    *,
    source_type: str | None = None,
    source_url_or_path: str | None = None,
    source_title: str | None = None,
    duration_seconds: int | None = None,
    language_detected: str | None = None,
    segments: list[dict] | None = None,
) -> str:
    """Build sandwich-technique prompt with transcript, metadata, schema, and rules.

    Order (data first, instruction last — favors small LLMs):
      SYSTEM_PREAMBLE → [SOURCE METADATA] → [TRANSCRIPTION] →
      LANGUAGE_ENFORCEMENT → [INSTRUCTION] → SCHEMA_TEMPLATE →
      FEW_SHOT_EXAMPLE → CRITICAL_RULES → OUTPUT_FORMAT

    Args:
        transcript: Full transcription text
        source_type: 'youtube', 'local_video', or 'local_audio'
        source_url_or_path: Original URL or file path
        source_title: Video title if available
        duration_seconds: Total duration
        language_detected: ISO 639-1 code from Whisper
        segments: Whisper segments with timestamps

    Returns:
        Complete prompt ready for LLM
    """
    # Bloco de metadata — o LLM deve copiar estes valores, nao inventar
    metadata_lines = []
    if source_type:
        metadata_lines.append(f"type: {source_type}")
    if source_url_or_path:
        metadata_lines.append(f"url_or_path: {source_url_or_path}")
    if source_title:
        metadata_lines.append(f"title: {source_title}")
    if duration_seconds is not None:
        metadata_lines.append(f"duration_seconds: {duration_seconds}")
    if language_detected:
        metadata_lines.append(f"language_detected: {language_detected}")

    metadata_block = ""
    if metadata_lines:
        metadata_block = (
            "\n[SOURCE METADATA — copie estes valores EXATAMENTE nos campos source e metadata do JSON]\n"
            + "\n".join(metadata_lines)
            + "\n"
        )

    # Transcrição: com timestamps se disponível, flat text se não
    if segments:
        transcript_text = _format_transcript_with_timestamps(segments)
        transcript_header = "[TRANSCRIPTION — with timestamps]"
    else:
        transcript_text = transcript
        transcript_header = "[TRANSCRIPTION]"

    # Truncar se necessário para caber no context window
    if len(transcript_text) > MAX_TRANSCRIPT_CHARS:
        total_len = len(transcript_text)
        transcript_text = transcript_text[:MAX_TRANSCRIPT_CHARS]
        transcript_text += (
            f"\n\n[... transcrição truncada de {total_len} para "
            f"{MAX_TRANSCRIPT_CHARS} caracteres. Analise somente o que está acima.]"
        )
        print(f"[prompts] warning: transcript truncated from {total_len} to {MAX_TRANSCRIPT_CHARS} chars")

    return f"""{SYSTEM_PREAMBLE}
{metadata_block}
{transcript_header}
{transcript_text}

{LANGUAGE_ENFORCEMENT}

[INSTRUCTION]
Analise a transcrição acima e extraia insights estruturados BASEADOS APENAS no
conteúdo da transcrição. NÃO copie o exemplo abaixo — ele é sobre culinária e
serve apenas para mostrar o formato do JSON. Devolva um JSON que valida contra
este schema EXATO:

{SCHEMA_TEMPLATE}

{FEW_SHOT_EXAMPLE}

{CRITICAL_RULES}

{OUTPUT_FORMAT}"""
