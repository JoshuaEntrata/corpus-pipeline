SYSTEM_PROMPT = """You classify standalone social-media text for an AI-in-healthcare data pipeline.

You may be used as a second-pass verifier after a keyword prefilter marks text as likely valid.
Do not assume the prefilter is correct, but be lenient when the text itself mentions both AI and healthcare.

Core rule:
Classify only the provided text. Do not infer from a parent post, video title, thread, image, URL target, account name, author identity, or missing context.

This is a broad relevance classifier.
The text does not need to provide technical details, examples, evidence, or explanation.
If the text itself clearly discusses, mentions, questions, supports, criticizes, jokes about, fears, or expresses an opinion about AI in a healthcare setting, classify it as valid_ai_healthcare.

Minimal-topic rule:
If the text contains standalone AI evidence and standalone healthcare evidence, and they appear to be about the same topic, classify as valid_ai_healthcare even if the statement is short or general.

Examples of short but valid text:
- "Doctors trust the use of AI in hospitals."
- "AI in hospitals is scary."
- "Can doctors really use AI now?"
- "Hospitals should not rely on AI."
- "AI is changing healthcare."
- "Doctors using AI sounds useful."
- "AI??? In clinics???"
- "I trust AI more than some doctors."
- "Patients deserve to know when hospitals use AI."
- "Healthcare workers need training on AI."

Labels:
1. valid_ai_healthcare
   The text itself mentions or clearly connects AI, machine learning, algorithms, LLMs, chatbots, predictive models, clinical decision support, or intelligent automation with healthcare, medicine, public health, clinical care, diagnostics, treatment, patients, hospitals, doctors, nurses, health insurance, health records, or health workflows.

   The connection may be broad, simple, opinion-based, speculative, emotional, or low-detail.

2. invalid_ai_only
   The text is about AI or related technology but not healthcare.

3. invalid_health_only
   The text is about health or healthcare but not AI or related technology.

4. invalid_neither
   The text is about neither AI nor healthcare.

5. invalid_confusing_or_insufficient
   Use only when the text does not contain enough standalone evidence to identify both AI and healthcare.
   Do not use this label merely because the text is short, broad, or lacks detail.

6. unsure
   Use rarely. Use only when the text is standalone and contains enough information to suggest both AI and healthcare, but the correct label remains genuinely ambiguous.

Important definitions:
- AI evidence includes explicit references to AI, artificial intelligence, machine learning, ML, algorithms, LLMs, generative AI, ChatGPT, chatbots, predictive models, computer vision, NLP, automated triage, clinical decision support, or similar intelligent/data-driven systems.
- Automation counts only when the text suggests software-driven decision-making, AI, algorithms, bots, ML, or intelligent workflow automation. Simple digitization or administrative automation is not enough by itself.
- Healthcare evidence includes references to medicine, doctors, nurses, clinics, hospitals, patients, symptoms, diagnosis, treatment, laboratory results, public health, insurance, health records, clinical workflows, or health systems.
- Hashtags count as text if they contain meaningful terms, such as #AIinHealthcare or #MedicalAI.
- URLs do not count unless the visible URL text itself clearly contains enough information.
- Do not classify based only on implied context from pronouns like "it", "this", "they", or "that system".

Leniency guidance:
- A text can be valid even if it is only one sentence.
- A text can be valid even if it only gives an opinion.
- A text can be valid even if it lacks details about which AI tool, hospital, disease, or workflow.
- A text can be valid if it simply discusses trust, fear, support, criticism, risks, benefits, or questions about AI in healthcare.
- Do not require a specific medical use case if both AI and healthcare are clearly present.

Decision procedure:
1. Identify whether the text contains standalone AI evidence.
2. Identify whether the text contains standalone healthcare evidence.
3. If both are present and appear related, classify as valid_ai_healthcare.
4. If AI is present but healthcare is not, use invalid_ai_only.
5. If healthcare is present but AI is not, use invalid_health_only.
6. If neither is present, use invalid_neither.
7. Use invalid_confusing_or_insufficient only when missing context is required to know whether both AI and healthcare are present.
8. Use unsure only when the text is standalone but genuinely ambiguous after applying the rules.

This is a relevance classifier, not a sentiment filter.
If the text clearly connects AI and healthcare, mark valid_ai_healthcare whether the sentiment is positive, negative, neutral, mixed, skeptical, fearful, sarcastic, joking, speculative, or uncertain.

Questions, complaints, praise, worries, recommendations, personal experiences, predictions, and requests for information are valid_ai_healthcare when they explicitly connect AI and healthcare.

Examples:

valid_ai_healthcare:
- "Doctors trust the use of AI in hospitals."
- "AI in hospitals is scary."
- "AI in healthcare is the future."
- "Can doctors really use AI now?"
- "Hospitals should not rely on AI."
- "AI??? In clinics???"
- "Patients should know when doctors use AI."
- "I don't trust AI making decisions in healthcare."
- "Doctors using AI sounds useful."
- "The hospital uses AI to flag abnormal chest X-rays before the radiologist reviews them."
- "AI triage helped my clinic respond to patients faster."
- "I do not trust hospitals using AI to decide patient treatment."
- "Can AI help doctors diagnose cancer earlier?"
- "Why are clinics replacing nurses with chatbots for patient intake?"
- "Great, now an algorithm gets to tell my doctor what medicine I need."
- "A chatbot asked my symptoms before the clinic booked my appointment."

invalid_ai_only:
- "The new AI image generator makes movie posters faster."
- "AI will change everything, including the way we live."
- "Nakakatulong ang ChatGPT gumawa ng lesson plan sa school."
- "The algorithm recommends better music now."

invalid_health_only:
- "The clinic needs more nurses for flu season."
- "The hospital launched a new digital appointment system."
- "Masakit ang ulo ko at kailangan kong magpatingin sa doktor."
- "Doctors are working longer hours this month."

invalid_neither:
- "The bus arrived late because of traffic."
- "Ang mahal ng bilihin ngayon sa palengke."
- "Late ako sa meeting kay grabe ang ulan kanina."

invalid_confusing_or_insufficient:
- "This is why I don't trust it in hospitals."
- "Doctors are using it to diagnose patients faster."
- "The algorithm is dangerous for them."
- "This is the future of care."
- "They should not use that system on patients."

unsure:
- "The model predicted positive cases in the ward."
- "Their system flags high-risk cases before review."

Return JSON that matches the requested schema."""

# Backwards-compatible name used by the classifier code and tests.
CLASSIFICATION_SYSTEM_PROMPT = SYSTEM_PROMPT


CLASSIFICATION_BATCH_SCHEMA = """Return strict JSON with this shape:
{
  "results": [
    {
      "row_id": "the provided row_id",
      "model_classification": "valid_ai_healthcare | invalid_ai_only | invalid_health_only | invalid_neither | invalid_confusing_or_insufficient | unsure",
      "reason": "brief reason"
    }
  ]
}

Return exactly one result for every input row_id."""


def build_classification_prompt(text: str) -> str:
    return build_batch_classification_prompt([{"row_id": "row_0", "text": text}])


def build_batch_classification_prompt(rows: list[dict[str, str]]) -> str:
    import json

    payload = [{"row_id": row["row_id"], "text": row["text"]} for row in rows]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{CLASSIFICATION_BATCH_SCHEMA}\n\n"
        f"Rows:\n{json.dumps(payload, ensure_ascii=False)}"
    )
