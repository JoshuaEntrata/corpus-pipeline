RAW_COLLECTION_FIELDS = [
    "run_id",
    "source_platform",
    "source_item_id",
    "source_url",
    "collection_method",
    "collection_query",
    "collected_at_utc",
    "created_at_utc",
    "author_id_hash",
    "title",
    "body_text",
    "description",
    "transcript",
    "comments_json",
    "engagement_json",
    "raw_json",
    "manual_file_name",
]

NORMALIZED_TEXT_ROW_FIELDS = [
    "text_item_id",
    "run_id",
    "source_platform",
    "source_item_id",
    "conversation_root_id",
    "parent_item_id",
    "text_type",
    "raw_text",
    "clean_text",
    "source_url",
    "created_at_utc",
    "collected_at_utc",
    "collection_method",
    "collection_query",
    "author_id_hash",
    "is_duplicate_text",
    "text_hash",
    "metadata_json",
]

CLASSIFICATION_FIELDS = [
    "text_item_id",
    "ai_healthcare_label",
    "ai_healthcare_confidence",
    "classifier_version",
    "classification_reason_short",
    "classified_at_utc",
    "used_prefilter",
    "prefilter_result",
    "model_name",
]

LANGUAGE_DETECTION_FIELDS = [
    "text_item_id",
    "language_label",
    "language_confidence",
    "language_detector_version",
    "detector_votes_json",
    "used_openai_fallback",
    "language_detected_at_utc",
]

