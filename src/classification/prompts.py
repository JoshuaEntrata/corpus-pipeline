CLASSIFIER_VERSION = "ai_healthcare_classifier_v1"

SYSTEM_PROMPT = """You classify standalone social-media text for an AI-in-healthcare data pipeline.

Rules:
1. Classify only the provided text.
2. Do not infer from a parent post, video title, thread, URL, or missing context.
3. valid_ai_healthcare means the text itself connects AI, automation, algorithms, machine learning, LLMs, chatbots, or related technology to healthcare, medicine, public health, clinical care, health systems, diagnostics, treatment, patients, hospitals, insurance, health records, or health workflows.
4. invalid_ai_only means the text is about AI but not healthcare.
5. invalid_health_only means the text is about health or healthcare but not AI.
6. invalid_neither means the text is about neither AI nor healthcare.
7. invalid_confusing_or_insufficient means the text is too short, vague, sarcastic, unclear, or context-dependent.
8. unsure means there is enough text, but the label remains genuinely uncertain.

Examples:

valid_ai_healthcare:
- English: "The hospital uses AI to flag abnormal chest X-rays before the radiologist reviews them."
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
