PAD_TOKEN = "<|pad|>"

UNK_TOKEN = "<|unk|>"

PAD_NER_TAG = "<|pad|>"

NER_ENCODING_MAP = {
    PAD_NER_TAG: -100,  # tag associated with padding token <|pad|>
    "O": 0,
    "B-ORG": 1,
    "I-ORG": 2,
    "B-PER": 3,
    "I-PER": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

NER_DECODING_MAP = {entity_id: entity for entity, entity_id in NER_ENCODING_MAP.items()}

if __name__ == "__main__":
    assert len(set(NER_ENCODING_MAP.values())) == len(NER_ENCODING_MAP.keys())
