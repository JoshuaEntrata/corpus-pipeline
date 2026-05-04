RAW_COLLECTION_FIELDS = [
    "source_platform",
    "source_item_id",
    "source_url",
    "collection_method",
    "created_at_utc",
    "title",
    "body_text",
    "description",
    "transcript",
    "comments_json",
]

NORMALIZED_TEXT_ROW_FIELDS = [
    "source_platform",
    "id",
    "preprocessed_text",
    "collection_method",
    "has_ai_keyword",
    "has_health_keyword",
    "needs_classification",
    "needs_language_detection",
]

CLASSIFICATION_FIELDS = [
    "text",
    "ai_healthcare_label",
    "ai_healthcare_confidence",
    "classification_reason_short",
    "classified_at_utc",
    "used_prefilter",
    "prefilter_result",
]

LANGUAGE_DETECTION_FIELDS = [
    "id",
    "text",
    "ai_healthcare_label",
    "language_label",
    "language_confidence",
    "language_detector_version",
    "detector_votes_json",
    "used_openai_fallback",
    "language_detected_at_utc",
]
