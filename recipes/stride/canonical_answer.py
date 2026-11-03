"""Dataset-independent minimal-span formatting from question and answer only.

This is an inference postprocessor, not a change to exact-match scoring.
It does not resolve entities, repair factual mistakes, or consult answer labels.
"""
import re

def format_answer(question, answer):
    if not isinstance(question, str):
        raise TypeError('Question must be text')
    if answer is None:
        return None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Answer must be nonempty text or null')
    value=answer.strip()
    q=question.casefold()
    # Restrict address shortening to birth/death locations, not arbitrary names
    # containing commas or broad geographic questions.
    birthplace=(re.search(r'\bwhere\b',q) and re.search(r'\b(?:born|died|die)\b',q))
    place_of_life_event=re.search(r'\bplace of (?:birth|death)\b',q)
    if birthplace or place_of_life_event:
        parts=[part.strip() for part in value.split(',')]
        if len(parts)>1 and all(parts) and not any(char.isdigit() for char in parts[0]):
            # Keep a dotted name abbreviation attached to the locality.
            if re.fullmatch(r'(?:[A-Za-z]\.){2,}',parts[1]):
                value=', '.join(parts[:2])
            else:
                value=parts[0]
    if re.match(r'^\s*who\b',q):
        shortened=re.sub(r'^(?:Sir|Dame)\s+','',value,flags=re.IGNORECASE)
        if len(shortened.split())>=2:
            value=shortened
    return value
