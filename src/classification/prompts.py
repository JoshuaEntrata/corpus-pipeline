CLASSIFIER_VERSION = "ai_healthcare_classifier_v2"

SYSTEM_PROMPT = """You classify standalone social-media text for an AI-in-healthcare data pipeline.
You may be used as a second-pass verifier after a keyword prefilter marks text as likely valid.
Verify from the text itself whether it truly connects AI and healthcare; if not, return the correct invalid label.

Rules:
1. Classify only the provided text.
2. Do not infer from a parent post, video title, thread, URL, or missing context.
3. valid_ai_healthcare means the text itself connects AI, automation, algorithms, machine learning, LLMs, chatbots, or related technology to healthcare, medicine, public health, clinical care, health systems, diagnostics, treatment, patients, hospitals, insurance, health records, or health workflows.
4. invalid_ai_only means the text is about AI but not healthcare.
5. invalid_health_only means the text is about health or healthcare but not AI.
6. invalid_neither means the text is about neither AI nor healthcare.
7. invalid_confusing_or_insufficient means the text is too short, vague, unclear, or context-dependent, and the AI-healthcare connection cannot be judged from the text alone.
8. unsure means there is enough text, but the label remains genuinely uncertain.
9. This is a relevance classifier, not a sentiment filter. If the text clearly connects AI and healthcare, mark valid_ai_healthcare whether the sentiment is positive, negative, neutral, mixed, skeptical, fearful, sarcastic, joking, or uncertain.
10. Questions, complaints, praise, worries, recommendations, personal experiences, predictions, and requests for information are valid_ai_healthcare when they explicitly connect AI and healthcare.
11. Do not reject text only because it is opinionated, emotional, critical, speculative, or written as a question.

Examples:

valid_ai_healthcare:
- English: "The hospital uses AI to flag abnormal chest X-rays before the radiologist reviews them."
- Positive sentiment: "AI triage helped my clinic respond to patients faster."
- Negative sentiment: "I do not trust hospitals using AI to decide patient treatment."
- Neutral question: "Can AI help doctors diagnose cancer earlier?"
- Skeptical question: "Why are clinics replacing nurses with chatbots for patient intake?"
- Sarcastic but clear: "Great, now an algorithm gets to tell my doctor what medicine I need."
- Tagalog: "Ginagamit ng doktor ang AI chatbot para tulungan ang pasyente sa sintomas bago konsultasyon."
- Cebuano: "Ang ospital naggamit ug machine learning aron matagna ang risgo sa sakit sa pasyente."
- Ilocano: "Ti klinika ket agar-aramat iti algorithm tapno maibaga no kasapulan ti pasiente ti follow-up."
- Hiligaynon: "Ang doktor nagagamit sang AI para magbulig basa sang resulta sang laboratoryo."
- Mixed: "Okay ang AI triage tool sa barangay clinic kung may nurse pa rin nga naga-check."

invalid_ai_only:
- English: "The new AI image generator makes movie posters faster."
- Tagalog: "Nakakatulong ang ChatGPT gumawa ng lesson plan sa school."
- Cebuano: "Ang algorithm sa app maayo mo recommend ug music."
- Ilocano: "Ti AI ket makatulong agsurat iti email para iti negosyo."
- Hiligaynon: "Ang chatbot nagabulig sabat sang customer questions sa online shop."
- Mixed: "Mas dali mag-edit ng vlog gamit ang AI tool."

invalid_health_only:
- English: "The clinic needs more nurses for flu season."
- Tagalog: "Masakit ang ulo ko at kailangan kong magpatingin sa doktor."
- Cebuano: "Kinahanglan ug tambal ang bata kay taas ang hilanat."
- Ilocano: "Nagpa-checkup ti pasiente idiay ospital gapu iti sakit ti barukong."
- Hiligaynon: "Nagakadto siya sa klinika para sa bulong sang ubo."
- Mixed: "Need ko magpa-check sa doctor kay sige ang kasakit sang tiyan."

invalid_neither, also called invalid both:
- English: "The bus arrived late because of traffic."
- Tagalog: "Ang mahal ng bilihin ngayon sa palengke."
- Cebuano: "Ganahan ko mokaon ug humba unya."
- Ilocano: "Nabanglo ti kape iti agsapa."
- Hiligaynon: "Gapulan subong kag traffic sa dalan."
- Mixed: "Late ako sa meeting kay grabe ang ulan kanina."

Return JSON that matches the requested schema."""


def user_prompt(text):
    return f"Text to classify:\n{text}"


CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "valid_ai_healthcare",
                "invalid_ai_only",
                "invalid_health_only",
                "invalid_neither",
                "invalid_confusing_or_insufficient",
                "unsure",
            ],
        },
        "confidence": {
            "type": "number",
            "description": "Confidence from 0.0 to 1.0.",
        },
        "reason_short": {
            "type": "string",
            "description": "One short reason based only on the provided text.",
        },
        "text_is_standalone_enough": {
            "type": "boolean",
            "description": "Whether the text can be judged without parent context.",
        },
    },
    "required": [
        "label",
        "confidence",
        "reason_short",
        "text_is_standalone_enough",
    ],
    "additionalProperties": False,
}
