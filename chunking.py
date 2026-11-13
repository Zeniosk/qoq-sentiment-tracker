import re
from pathlib import Path

from ingest import extract_text_from_pdf

PDF_PATH = "data/raw/infosys_q2_fy25.pdf"

# --- Cleaning patterns ---
PAGE_MARKER = re.compile(r"-{2,}\s*Page\s*\d+\s*-{2,}")
FOOTER = re.compile(r"External Document\s*©[^\n]*", re.IGNORECASE)

# A speaker line is either "Name: text" or a short standalone "Name" / "Name - Title"
INLINE_SPEAKER = re.compile(r"^([A-Z][A-Za-z.'\- ]{2,40}):\s*(.*)$")
STANDALONE_SPEAKER = re.compile(
    r"^([A-Z][A-Za-z.'\- ]{2,40})(?:\s*[-–—]\s*[A-Za-z &,.'()/\-]{2,60})?$"
)

# UPDATED: Catches "open the call for questions" OR "question -and-answer" (with or without spaces)
QA_TRIGGER = re.compile(r"(?:open the call for questions|question\s*-?and-?answer)", re.IGNORECASE)

TARGET_CHARS = 1200
MAX_CHARS = 2000
MIN_CHARS = 80
OVERLAP = 200

# Blacklist common phrases that look like speaker names but aren't
SPEAKER_BLACKLIST = {
    "Offset by", "Tailwinds of", "Headwinds from", "Driven by",
    "Due to", "As a result", "In addition", "For example"
}

def clean_raw(text):
    text = PAGE_MARKER.sub("\n", text)
    text = FOOTER.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text

def looks_like_speaker(line):
    """Guard against headings being mistaken for names."""
    if len(line.split()) > 6:
        return False
    if any(ch.isdigit() for ch in line):
        return False
    if line in SPEAKER_BLACKLIST:
        return False
    return True

def split_turns(text):
    """Return [(speaker, utterance), ...] in document order."""
    turns = []
    speaker = "Unknown"
    buffer = []

    def flush():
        if buffer:
            utterance = " ".join(buffer).strip()
            if utterance:
                turns.append((speaker, utterance))
            buffer.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        m = INLINE_SPEAKER.match(line)
        if m and looks_like_speaker(m.group(1)):
            flush()
            speaker = m.group(1).strip()
            if m.group(2):
                buffer.append(m.group(2))
            continue

        m = STANDALONE_SPEAKER.match(line)
        if m and looks_like_speaker(m.group(1)):
            flush()
            speaker = m.group(1).strip()
            continue

        buffer.append(line)

    flush()
    return turns

def chunk_transcript(text):
    text = clean_raw(text)

    # Locate the start of Q&A so we can tag sections
    qa_match = QA_TRIGGER.search(text)
    qa_pos = qa_match.start() if qa_match else None

    turns = split_turns(text)

    chunks = []
    char_cursor = 0
    current_speaker = None
    current_text = ""

    def emit(speaker, body, section):
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < MIN_CHARS:
            return
        if len(body) <= MAX_CHARS:
            pieces = [body]
        else:
            pieces = []
            start = 0
            while start < len(body):
                pieces.append(body[start:start + MAX_CHARS])
                start += MAX_CHARS - OVERLAP
        for piece in pieces:
            chunks.append({
                "text": piece,
                "speaker": speaker,
                "section": section,
                "char_start": char_cursor,
            })

    for speaker, utterance in turns:
        section = "qa" if (qa_pos is not None and char_cursor >= qa_pos) else "prepared"
        char_cursor += len(utterance)

        if speaker == current_speaker and len(current_text) + len(utterance) < TARGET_CHARS:
            current_text += " " + utterance
            continue

        if current_speaker is not None:
            emit(current_speaker, current_text, section)

        current_speaker = speaker
        current_text = utterance

    if current_speaker is not None:
        emit(current_speaker, current_text, section)

    for i, c in enumerate(chunks):
        c["chunk_id"] = i

    return chunks

if __name__ == "__main__":
    raw = extract_text_from_pdf(PDF_PATH)

    # Dump cleaned text so you can eyeball the real format in VS Code
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    dump_path = Path("data/processed/infosys_q2_fy25.txt")
    dump_path.write_text(clean_raw(raw), encoding="utf-8")
    print(f"Cleaned text written to {dump_path} -- open it in VS Code\n")

    chunks = chunk_transcript(raw)

    lengths = [len(c["text"]) for c in chunks]
    print("=" * 60)
    print(f"Chunks produced : {len(chunks)}")
    if lengths:
        print(f"Avg length      : {sum(lengths) // len(lengths)} chars")
        print(f"Min / Max       : {min(lengths)} / {max(lengths)} chars")
    print(f"Section split   : "
          f"{sum(c['section'] == 'prepared' for c in chunks)} prepared / "
          f"{sum(c['section'] == 'qa' for c in chunks)} qa")

    print("\nSpeakers detected:")
    speakers = {}
    for c in chunks:
        speakers[c["speaker"]] = speakers.get(c["speaker"], 0) + 1
    for name, count in sorted(speakers.items(), key=lambda x: -x[1]):
        print(f"  {count:3d}  {name}")

    print("\nFirst 3 Q&A chunks:")
    qa_chunks = [c for c in chunks if c['section'] == 'qa']
    for c in qa_chunks[:3]:
        print(f"\n[{c['chunk_id']}] {c['speaker']} ({c['section']})")
        print(f"  {c['text'][:220]}...")
    print("=" * 60)