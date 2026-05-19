from agent_v1.graph.states import DataSample


def analyzer_prompt(conversations_text: str) -> str:
    return f"""You are an expert at diagnosing why an AI chatbot gave a bad answer.

You will receive conversations where a user clicked "Dislike". Your job is to figure out WHY the model failed so we can fix it with better training data.

CONVERSATIONS:
{conversations_text}

Analyze what went wrong:
- Did the model get facts wrong or make things up? → REAL ISSUE
- Did the model not know something it should know? → REAL ISSUE  
- Did the model give a vague, unhelpful, or off-topic answer? → REAL ISSUE
- Was the user just unhappy with the tone or length? → NOT a real issue

If it IS a real issue, identify 3-5 precise training topics that would directly fix the model's gaps. Be specific — not "blockchain" but "how blockchain consensus mechanisms work".

You MUST respond with ALL three fields:
- is_real_issue: true or false
- reasoning: 1-2 sentences — what specifically did the model get wrong or not know?
- topics: list of 3-5 specific topic strings if is_real_issue is true, or [] if false

IMPORTANT: The "topics" field is REQUIRED even if it is an empty list. Never omit it.

Respond with a TopicAnalysis object."""


def link_fetcher_prompt(topics: list[str], search_results: str) -> str:
    return f"""You are selecting the best web sources to generate training data for an AI model.

The model failed on these topics: {", ".join(topics)}

You need sources that contain CLEAR, ACCURATE, DETAILED explanations of these topics — the kind of content that would teach the model the right answer.

SEARCH RESULTS:
{search_results}

Pick 3-5 links that best match. Good sources:
- Technical documentation, tutorials, explainers
- Educational sites, encyclopedias, reputable tech blogs
- Any page with detailed, factual content on the topic

Avoid: pages that are mostly ads, login walls, or thin content.

For every link you select, provide all three fields:
- url: the full URL
- title: the page title  
- relevance_reason: one sentence — what specific information does this page have that will help fix the model's knowledge gap?

Respond with a LinkBatch object."""


def question_gen_prompt(topics: list[str], scraped_content: str) -> str:
    return f"""You are generating training questions to fix gaps in an AI model's knowledge.

The model struggled with: {", ".join(topics)}

REFERENCE CONTENT (from web sources):
{scraped_content}

Generate exactly 10 questions that directly address where the model failed. Requirements:
- Every question must be answerable using only the reference content above
- Questions should probe the exact facts and concepts the model got wrong
- Mix question types: definition (What is X?), explanation (How does X work?), comparison (What's the difference between X and Y?), application (When would you use X?)
- Be specific — avoid vague questions like "Tell me about X"
- Each question must be self-contained (no "as mentioned above" or references to other questions)

Respond with a QuestionSet object."""


def sampler_prompt(
    questions: list[str],
    topics: list[str],
    context: str,
    source_url: str
) -> str:
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""You are generating supervised training data to fix an AI model's knowledge gaps.

REFERENCE CONTENT (ground truth — use only this):
{context}

SOURCE: {source_url}
TOPICS: {", ".join(topics)}

QUESTIONS TO ANSWER:
{questions_text}

For each question, write a training sample. Rules:
- Answer must be factually accurate and grounded entirely in the reference content above
- Answer must be complete and detailed — minimum 3 sentences, 60+ words
- Answer must directly address what the question asks — no padding or filler
- Write in clear, direct language as if explaining to someone learning this topic
- Do not repeat the question in the answer
- Do not make up facts not present in the reference content

Generate a JSON object with this structure:
{{
  "samples": [
    {{
      "id": "",
      "question": "exact question from the list above",
      "answer": "complete, detailed answer grounded in the reference content",
      "confidence_score": 0.85,
      "topic": "the most relevant topic from: {', '.join(topics)}",
      "source_url": "{source_url}"
    }}
  ]
}}

CRITICAL: One sample per question. Respond ONLY with valid JSON."""


def validator_prompt(samples: list[DataSample], total_count: int) -> str:
    samples_text = "\n\n".join(
        f"ID: {s.id}\nQuestion: {s.question}\nAnswer: {s.answer}\nConfidence: {s.confidence_score}"
        for s in samples
    )
    return f"""You are validating training data that will be used to fix an AI model's knowledge gaps. Your job is critical — bad training data makes the model worse.

SAMPLES TO VALIDATE:
{samples_text}

For each sample, check these things in order:

1. RELEVANCE — Does the answer actually address what the question asks? 
   Fail if: the answer is off-topic, answers a different question, or dodges the question.

2. ACCURACY — Is the answer factually coherent with no contradictions?
   Fail if: the answer contains statements that contradict each other or make no logical sense.

3. COMPLETENESS — Is the answer substantive enough to teach the model?
   Fail if: the answer is under 2 sentences, vague, or just restates the question.

4. FABRICATION — Does the answer invent specific facts (names, numbers, dates, URLs) not grounded in real knowledge?
   Fail if: the answer contains made-up specifics that look plausible but are likely hallucinated.

5. CONFIDENCE — Is the confidence_score below 0.4?
   Fail if: yes.

For FAILED samples: write a revised_answer that fixes the specific problem. If you cannot write a genuinely correct answer, set revised_answer to null and let it fail.

For PASSED samples: set revised_answer to null.

Generate a JSON object with this exact structure:
{{
  "results": [
    {{
      "sample_id": "the sample ID",
      "is_valid": true,
      "issues": [],
      "revised_answer": null
    }}
  ],
  "total": {total_count},
  "passed": 0,
  "failed": 0
}}

CRITICAL:
- Include ALL required fields: results, total, passed, failed
- total should be {total_count}
- Count and fill in passed and failed accurately
- Respond ONLY with valid JSON"""