from agent_v1.graph.states import DataSample


def analyzer_prompt(conversations_text: str) -> str:
    return f"""You are an AI training data quality analyst.

You will receive 2-3 conversations where a user clicked "Dislike" or "Not Satisfied".

Your job:
1. Determine if the model's responses reveal a GENUINE knowledge or capability gap
2. If yes, identify 3-5 specific training topics that would address these gaps
3. If the dissatisfaction is about tone/style only (not missing knowledge), mark as NOT a real issue

Be conservative — only flag as a real issue if the model clearly lacks factual knowledge.

CONVERSATIONS:
{conversations_text}

Respond with a TopicAnalysis object."""


def link_fetcher_prompt(topics: list[str], search_results: str) -> str:
    return f"""You are a research curator selecting sources for LLM training data.

Topics that need training data: {", ".join(topics)}

Web search results:
{search_results}

Select the 3-5 most relevant, authoritative, and information-dense links.
Prefer: official government portals, academic papers, reputable news outlets.
Avoid: forums, opinion pieces, social media, low-quality aggregators.

IMPORTANT: For every link you select, you MUST provide all three fields:
- url: the full URL
- title: the page title
- relevance_reason: one sentence explaining why this source is relevant to the topics

Respond with a LinkBatch object."""

def question_gen_prompt(topics: list[str], scraped_content: str) -> str:
    return f"""You are an expert at generating training questions for language models.

TOPICS TO COVER: {", ".join(topics)}

REFERENCE CONTENT (sourced from authoritative web pages):
{scraped_content}

Generate exactly 10 training questions. Requirements:
- Mix factual (What is...?), procedural (How do I...?), and edge cases
- All questions must be answerable from the reference content
- Vary difficulty: 3 easy, 4 medium, 3 hard
- Make each question specific — avoid vague or generic phrasing
- Each question must stand alone without needing context

Respond with a QuestionSet object."""


def sampler_prompt(
    questions: list[str],
    topics: list[str],
    context: str,
    source_url: str
) -> str:
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""You are an expert at generating high-quality supervised training data for LLMs.

REFERENCE CONTEXT (from a verified source):
{context}

TOPICS: {", ".join(topics)}
SOURCE: {source_url}

QUESTIONS:
{questions_text}

For each question, generate a training sample:
- id: leave blank (will be assigned automatically)
- question: copy the question exactly as given
- answer: complete, accurate, detailed (minimum 3-5 sentences). Do NOT be vague.
- confidence_score: 0.0 to 1.0 — how well the context supports this answer
- topic: the most relevant topic from the list above
- source_url: use the provided source URL

Respond with a SampleBatch object."""


def validator_prompt(samples: list[DataSample]) -> str:
    samples_text = "\n\n".join(
        f"ID: {s.id}\nQ: {s.question}\nA: {s.answer}\nScore: {s.confidence_score}"
        for s in samples
    )
    return f"""You are a strict quality validator for LLM supervised training data.

Review each sample carefully. For each one:
- is_valid: True ONLY if the answer is factually coherent, specific, complete, and not hallucinated
- issues: list specific problems if invalid ("answer is vague", "contradicts question", "hallucinated fact")
- revised_answer: provide a corrected answer ONLY when you can genuinely improve it

SAMPLES TO VALIDATE:
{samples_text}

Rejection criteria:
- Confidence score below 0.4
- Answer shorter than 2 sentences
- Answer does not address the question
- Answer contains fabricated specific facts (names, dates, numbers not in context)

Respond with a ValidationReport object."""