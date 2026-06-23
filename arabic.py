"""
Arabic Diacritization (Tashkeel) + POS Tagging - Simple Streamlit Interface
----------------------------------------------------------------------------
Setup:
    pip install streamlit camel-tools
    camel_data -i disambig-mle-calima-msa-r13

Run:
    streamlit run arabic_tashkeel_pos_app.py
"""

import streamlit as st
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize

@st.cache_resource
def download_camel_data():
    """Checks if data folder exists on Streamlit Cloud; if not, triggers installation."""
    data_path = os.path.expanduser('~/.camel_tools_data')
    if not os.path.exists(data_path):
        # Triggers camel_data command-line downloader safely via background process
        subprocess.run(["camel_data", "-i", "disambig-mle-calima-msa-r13"], check=True)
    return True

# Run the data downloader automatically before executing any analytical imports
_ = download_camel_data()

st.set_page_config(page_title="Arabic Diacritization & POS Tagger", layout="wide")

# Strings CAMeL Tools uses when a feature has no real value
_EMPTY_VALS = {"", "NOAN", "backoff", "NO_ANALYSIS", None}


@st.cache_resource
def load_model():
    return MLEDisambiguator.pretrained()


mle = load_model()


def _clean(val):
    """Return val as-is if it is meaningful, otherwise return None."""
    return None if val in _EMPTY_VALS else val

def convert_pattern_to_fa3al(pattern: str) -> str:
    """Converts numeric pattern placeholders safely to traditional Arabic weights."""
    if not pattern or pattern in ["—", "PUNC", "UNK"]:
        return "—"
    
    # 1. Temporarily protect real Arabic characters if any exist
    # 2. Map CAMeL's exact structural characters safely
    converted = (pattern.replace("1", "ف")
                        .replace("2", "ع")
                        .replace("3", "ل")
                        .replace("#", "ـ")) # Replaces pattern hashtags with a clean Arabic elongation line
    return converted

def clean_root_display(raw_root: str) -> str:
    """Cleans up the root string by removing hashtags or tracking placeholders."""
    if not raw_root or raw_root in ["PUNC", "UNK"]:
        return "—"
    
    # Clean up tracking characters and replace internal weak-letter hashtags with an empty slot or dash
    cleaned = raw_root.replace(".", " ").replace("-", " ").replace("#", "_")
    return cleaned

def process(paragraph: str):
    """
    Returns a list of dicts for each token:
      word    – original surface form
      diac    – diacritized form (tashkeel)
      pos     – part-of-speech tag
      root    – Arabic root (جذر), dot-separated in DB e.g. k.t.b → displayed spaced
      pattern – morphological pattern (وزن), e.g. فَعَلَ
      _raw    – full analysis dict (for debug expander)
    """
    tokens = simple_word_tokenize(paragraph)
    disambiguated = mle.disambiguate(tokens)

    results = []
    for d in disambiguated:
        if d.analyses:
            top = d.analyses[0].analysis

            # Extract and clean Root using our new function
            raw_root = _clean(top.get("root"))
            root_display = clean_root_display(raw_root)

            # Extract and clean Pattern using our updated function
            raw_pattern = _clean(top.get("pattern"))
            pattern_display = convert_pattern_to_fa3al(raw_pattern)

            results.append({
                "word":    d.word,
                "diac":    _clean(top.get("diac")) or d.word,
                "pos":     _clean(top.get("pos")) or "UNK",
                "root":    root_display,
                "pattern": pattern_display,
                "_raw":    dict(top),
            })
        else:
            results.append({
                "word":    d.word,
                "diac":    d.word,
                "pos":     "UNK",
                "root":    "—",
                "pattern": "—",
                "_raw":    {},
            })
    return results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🕮 Arabic Diacritization & POS Tagger")
st.write(
    "Enter an Arabic paragraph below. "
    "The tool shows diacritized text, POS tags, Arabic root (جذر), "
    "and morphological pattern (وزن) for each word."
)

text = st.text_area("Arabic paragraph:", height=150, placeholder="اكتب الفقرة هنا...")

if st.button("Process", type="primary"):
    if not text.strip():
        st.warning("Please enter some Arabic text.")
    else:
        results = process(text)

        # ---------- Box 1: Diacritized paragraph ----------
        st.subheader("📝 Diacritized Text (Tashkeel)")
        diac_text = " ".join(r["diac"] for r in results)
        st.markdown(
            '<div style="direction:rtl; font-size:26px; line-height:2.2; '
            'border:2px solid #4CAF50; border-radius:12px; padding:24px; '
            'background-color:#f6fff6; '
            'font-family:\'Traditional Arabic\',\'Arial\',sans-serif;">'
            f"{diac_text}</div>",
            unsafe_allow_html=True,
        )

        st.write("")

        # ---------- Box 2: Per-word cards ----------
        st.subheader("🏷️ Word Analysis")
        st.caption(
            "Each card: **diacritized word** · "
            "<span style='color:#1565C0'>POS</span> · "
            "<span style='color:#2E7D32'>جذر root</span> · "
            "<span style='color:#6A1B9A'>وزن pattern</span>",
            unsafe_allow_html=True,
        )

        cards_html = (
            "<div style='direction:rtl; display:flex; flex-wrap:wrap; gap:14px; margin-top:10px;'>"
        )
        for r in results:
            cards_html += (
                "<div style='border:2px solid #2196F3; border-radius:12px; "
                "padding:12px 16px; text-align:center; min-width:120px; "
                "background-color:#f3f8ff;'>"

                # ① Diacritized word
                f"<div style='font-size:22px; font-weight:bold; "
                f"font-family:Traditional Arabic,Arial,sans-serif;'>{r['diac']}</div>"

                # ② POS
                f"<hr style='margin:6px 0; border-color:#cde;'>"
                f"<div style='font-size:12px; color:#1565C0; font-weight:600;'>"
                f"POS: {r['pos']}</div>"

                # ③ Root
                f"<div style='font-size:13px; color:#2E7D32; margin-top:4px; "
                f"font-family:Traditional Arabic,Arial,sans-serif;'>"
                f"جذر: {r['root']}</div>"

                # ④ Pattern
                f"<div style='font-size:13px; color:#6A1B9A; margin-top:4px; "
                f"font-family:Traditional Arabic,Arial,sans-serif;'>"
                f"وزن: {r['pattern']}</div>"

                "</div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        st.write("")

        # ---------- Table view ----------
        with st.expander("📊 View as table"):
            st.table([
                {k: v for k, v in r.items() if k != "_raw"}
                for r in results
            ])

        # ---------- Debug: raw analysis dict ----------
        #with st.expander("🔬 Debug: raw analysis dicts"):
        #    for r in results:
        #        st.write(f"**{r['word']}**", r["_raw"])