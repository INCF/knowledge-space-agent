# agents.py
import os
import re
import json
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, TypedDict, Any

from langgraph.graph import StateGraph, END

from ks_search_tool import general_search, general_search_async, global_fuzzy_keyword_search
from retrieval import Retriever

#  LLM (Gemini) client setup
try:
    from google import genai
    from google.genai import types as genai_types
except Exception as _e:
    raise RuntimeError("google-genai is required. Install with: pip install google-genai") from _e


def _use_vertex() -> bool:
    """
    Use Vertex AI if GCP_PROJECT_ID is present (unless GEMINI_USE_VERTEX explicitly disables it).
    """
    flag = os.getenv("GEMINI_USE_VERTEX")
    if flag is not None:
        return flag.strip().lower() in {"1", "true", "yes", "y"}
    return bool(os.getenv("GCP_PROJECT_ID"))


def _ensure_google_creds_for_vertex() -> None:
    """
    Prefer Application Default Credentials (ADC) on GCE:
      - If GOOGLE_APPLICATION_CREDENTIALS is already set
      - Else, if backend/service-account.json exists, use it.
      - Else, do nothing and let ADC from the metadata server be used.
    """
    if not _use_vertex():
        return
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    sa_path = os.path.join(os.path.dirname(__file__), "service-account.json")
    if os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path


def _require_llm_creds() -> None:
    """
    Validate that we have the minimum inputs for the selected mode.
    """
    if _use_vertex():
        if not os.getenv("GCP_PROJECT_ID"):
            raise RuntimeError("GCP_PROJECT_ID must be set for Vertex mode.")
        _ensure_google_creds_for_vertex()
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY must be set for API-key mode.")


_GENAI_CLIENT = None


def _get_genai_client():
    """
    Build a google.genai client for either Vertex (ADC or creds file) or API-key mode.
    """
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT

    if _use_vertex():
        project = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_REGION") or "europe-west4"
        _ensure_google_creds_for_vertex()
        _GENAI_CLIENT = genai.Client(vertexai=True, project=project, location=location)
    else:
        _GENAI_CLIENT = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    return _GENAI_CLIENT


FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
FLASH_LITE_MODEL = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-2.5-flash-lite")


# Query intent/types
class QueryIntent(Enum):
    DATA_DISCOVERY = "data_discovery"
    ACCESS_DOWNLOAD = "access_download"
    METADATA_QUERY = "metadata_query"
    QUALITY_CHECK = "quality_check"
    TOOLING_FORMAT = "tooling_format"
    INSTITUTION = "institution"
    GREETING = "greeting"


LLM_TOKEN_LIMITS = {
    "keywords": 256,
    "rewrite": 256,
    "synthesis": 8192,
    "intents": 256,
}


def _is_more_query(text: str) -> Optional[int]:
    t = (text or "").strip().lower()
    if not t:
        return None
    if t in {"more", "next", "continue", "more please", "show more", "keep going"}:
        return None
    m = re.match(r"^(?:next|more|show)\s+(\d{1,3})\b", t)
    return int(m.group(1)) if m else (None if any(w in t for w in ["more", "next", "continue"]) else None)


#  LLM calls using google.genai
async def call_gemini_for_keywords(query: str) -> List[str]:
    """
    Extract raw keywords/phrases from the user's text using the LLM only.
    No local greeting filters — prompt handles exclusions. Minimal trim+dedupe here.
    """
    _require_llm_creds()
    client = _get_genai_client()
    prompt = (
        "Extract important search keywords and multi-word phrases from a neuroscience *data* query.\n"
        "Return STRICT JSON only:\n"
        "{ \"keywords\": [\"...\"] }\n"
        "\n"
        "CRITICAL RULES:\n"
        "1) Output ONLY tokens/phrases that appear verbatim in the user text (case-insensitive). No stemming/synonyms.\n"
        "2) EXCLUDE greetings/small talk and their misspellings/elongations (hi, hello, hellow, helo, hey, yo, hola, "
        "   hallo, howdy, greetings, sup, heyya, heyyy, hiii). Treat /h+e*l+l*o+w*/, /he+y+/, /hi+/ as greetings.\n"
        "3) Do NOT include words just because they *look like* a greeting (e.g., 'yellow' from 'hellow').\n"
        "4) Keep multi-word technical phrases intact (e.g., 'medial prefrontal cortex', 'two-photon imaging').\n"
        "5) Preserve license/format tokens exactly (PDDL, CC0, NWB, BIDS, NIfTI, DICOM, HDF5).\n"
        "6) If the message is only a greeting/small talk, return {\"keywords\": []}.\n"
        f"\nQuery: {query}\n"
    )
    cfg = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=LLM_TOKEN_LIMITS["keywords"],
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(model=FLASH_LITE_MODEL, contents=[prompt], config=cfg)
    out = json.loads(resp.text or "{}")
    kws = out.get("keywords", []) or []
    normalized: List[str] = []
    for k in kws:
        if not isinstance(k, str):
            continue
        t = k.strip()
        if ":" in t:
            t = t.split(":", 1)[1].strip()
        if t:
            normalized.append(t)
    return list(dict.fromkeys(normalized))[:20]


async def call_gemini_rewrite_with_history(query: str, history: List[str]) -> str:
    """
    Rewrite the user's query using short chat history if necessary.
    Keeps exact tokens and multi-word phrases intact.
    """
    _require_llm_creds()
    client = _get_genai_client()
    last_user_turns = [h for h in history if h.startswith("User: ")]
    ctx = "\n".join(last_user_turns[-6:])
    prompt = (
        "Rewrite the user's latest search query to be SELF-CONTAINED using prior context only if needed.\n"
        "Return ONLY the rewritten query text (no JSON, no prose).\n"
        "\n"
        "Rules:\n"
        "• If the latest message is a pure greeting (hi/hello/hellow/hey etc.), return it unchanged.\n"
        "• Otherwise remove salutations/filler and keep only the research intent.\n"
        "• Do NOT invent entities. Keep exact license/format tokens (PDDL, CC0, BIDS, NWB, NIfTI, DICOM, HDF5) and numbers.\n"
        "• Preserve multi-word technical phrases intact (e.g., 'medial prefrontal cortex').\n"
        "• Do NOT transform or replace domain terms with synonyms.\n"
        "\n"
        f"Context:\n{ctx}\n"
        f"Latest: {query}\n"
    )
    cfg = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=LLM_TOKEN_LIMITS["rewrite"],
    )
    resp = client.models.generate_content(model=FLASH_LITE_MODEL, contents=[prompt], config=cfg)
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini rewrite returned empty text.")
    return text


async def call_gemini_detect_intents(query: str, history: List[str]) -> List[str]:
    """
    Multi-label intent detection via LLM.
    - 'greeting' only when message is purely small talk.
    - If any data-related tokens exist, prefer data_discovery.
    """
    _require_llm_creds()
    client = _get_genai_client()
    allowed = [i.value for i in QueryIntent]
    last_user_turns = [h for h in history if h.startswith("User: ")]
    ctx = "\n".join(last_user_turns[-6:])
    prompt = (
        "Detect which intents apply to the user's message. MULTI-LABEL allowed.\n"
        f"Allowed intents: {allowed}\n"
        "Return STRICT JSON only: {\"intents\": [\"...\"]} using only allowed values.\n"
        "\n"
        "Decisions:\n"
        "• Choose 'greeting' ONLY if the message is essentially small talk/a salutation (e.g., hi/hello/hellow/hey etc.).\n"
        "• If the message ALSO contains any dataset/data terms (e.g., EEG, fMRI, BIDS, NWB, hippocampus, rat, dataset, download, license), "
        "  DO NOT include 'greeting'. Prefer data-related intents instead.\n"
        "• Typos/elongations still count as greetings (hellow/helo/heyyy/hiii). If message is ONLY that, return ['greeting'].\n"
        "• If unsure and there is any data-related token, prefer data_discovery.\n"
        "\n"
        f"Context:\n{ctx}\n"
        f"Query: {query}\n"
    )
    cfg = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=LLM_TOKEN_LIMITS["intents"],
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(model=FLASH_LITE_MODEL, contents=[prompt], config=cfg)
    out = json.loads(resp.text or "{}")
    intents = [i for i in out.get("intents", []) if i in allowed]
    return list(dict.fromkeys(intents or [QueryIntent.DATA_DISCOVERY.value]))[:6]




async def call_gemini_for_final_synthesis(
    query: str,
    search_results: List[dict],
    intents: List[str],
    start_number: int = 1,
    previous_text: Optional[str] = None,
) -> str:

    _require_llm_creds()
    client = _get_genai_client()

    extras = []
    if QueryIntent.ACCESS_DOWNLOAD.value in intents:
        extras.append("Focus on access methods, download links, APIs, and license information.")
    if QueryIntent.METADATA_QUERY.value in intents:
        extras.append("Emphasize technical specs, preprocessing, collection methods, parameters.")
    if QueryIntent.QUALITY_CHECK.value in intents:
        extras.append("Highlight sample sizes, completeness, QC metrics, known issues.")
    if QueryIntent.TOOLING_FORMAT.value in intents:
        extras.append("Focus on file formats, tool compatibility, and pipelines.")
    if QueryIntent.INSTITUTION.value in intents:
        extras.append("Highlight the institution/organization and collaborations.")

    key_details_spec = (
        "- Under **Key Details**, produce a compact, multi-line bullet list in THIS EXACT ORDER if values exist:\n"
        "  - Species: ...\n"
        "  - Brain Regions: ...\n"
        "  - Technique or Modality: ...\n"
        "  - Data Formats: ...\n"
        "  - License: ...\n"
        "  - Subjects/Samples: ...\n"
        "  - Tasks/Conditions: ...\n"
        "  - Recording Specs: ...  (e.g., sampling rate, channels, TR, resolution)\n"
        "  - Authors: ...\n"
        "  - Year: ...\n"
        "- Each label MUST be on its own line as a sub-bullet under **Key Details**; do not merge labels on one line.\n"
        "- Show at most 6 values per field; if more, show the first 6 and append '…'.\n"
        "- For Authors, show up to 5 names; if more, append 'and {N} more'.\n"
        "- Never print empty labels; omit fields that are missing.\n"
        "- Preserve exact tokens for licenses and formats (PDDL, CC0, BIDS, NWB).\n"
        "- Include a **Relevance** section that explains specifically how this dataset matches the user's query.\n"
    )
    extra_rules = ("\n" + "\n".join(extras)) if extras else ""

    base_prompt = (
        "You are a neuroscience data expert helping researchers find and understand datasets.\n"
        "Use the raw candidate objects below; fields may appear at the top level, inside `metadata`, `_source`, or `detailed_info`.\n"
        "RULES:\n"
        "- Show up to 15 datasets without truncation mid-item.\n"
        "- Only mention fields that actually exist.\n"
        "- If few exact matches exist, include closely related datasets.\n"
        "- Never claim lack of memory; continue the list naturally if asked for 'more'.\n"
        f"{key_details_spec}{extra_rules}\n\n"
        "OUTPUT FORMAT:\n"
        "### 🔬 Neuroscience Datasets Found\n\n"
        "####  {number}. {Title}\n"
        "- **Source:** {datasource_name or institution}\n"
        "- **Description:** {1-2 sentences}\n"
        "- **Key Details:** (follow the exact order and format specified above)\n"
        "- **Relevance:** {Explain how this dataset matches the user's query}\n"
        "- **🔗 Access:** [{source name}]({primary_link})\n\n"
        f"Start numbering at {start_number} and continue sequentially. Do not repeat earlier items.\n"
    )

    def _cfg():
        return genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=LLM_TOKEN_LIMITS["synthesis"],
        )

    prompt = (
        f"{base_prompt}\nUser Query: {json.dumps(query, ensure_ascii=False)}\n"
        f"Intents: {json.dumps(intents)}\n\nRaw candidates (JSON):\n"
        f"{json.dumps(search_results, ensure_ascii=False)}"
    )

    resp = client.models.generate_content(model=FLASH_MODEL, contents=[prompt], config=_cfg())
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini synthesis returned empty text.")
    return text


#  Agent state and search/fuse/response pipeline
class AgentState(TypedDict):
    session_id: str
    query: str
    history: List[str]
    keywords: List[str]
    effective_query: str
    intents: List[str]
    ks_results: List[dict]
    vector_results: List[dict]
    final_results: List[dict]
    all_results: List[dict]
    final_response: str


class KSSearchAgent:
    async def run(self, query: str, keywords: List[str], want: int = 45) -> dict:
        try:
            print("  -> Using parallel enrichment in KS search")
            general = await general_search_async(query, top_k=min(want, 50), enrich_details=True)
            general = general.get("combined_results", [])
        except Exception as e:
            print(f"Async general search error, falling back to sync: {e}")
            try:
                general = general_search(query, top_k=min(want, 50), enrich_details=True).get("combined_results", [])
            except Exception as e2:
                print(f"Sync general search error: {e2}")
                general = []
        try:
            print(f"  -> Running fuzzy search with keywords: {keywords}")
            fuzzy = global_fuzzy_keyword_search(keywords, top_k=min(want, 50))
            print(f"  -> Fuzzy search returned {len(fuzzy)} results")
        except Exception as e:
            print(f"Fuzzy config search error: {e}")
            fuzzy = []
        return {"combined_results": (general + fuzzy)[: max(want, 15)]}


class VectorSearchAgent:
    def __init__(self):
        self.retriever = Retriever()
        self.is_enabled = self.retriever.is_enabled

    async def run(self, query: str, want: int, context: Optional[Dict] = None) -> List[dict]:
        if not self.is_enabled:
            return []
        try:
            # Run the synchronous search in a thread to make it async
            # Use enhanced search with hybrid and re-ranking
            results = await asyncio.to_thread(
                self.retriever.search,
                query=query,
                top_k=min(want, 50),
                context={"raw": True},
                use_hybrid=True,
                use_rerank=True
            )
            # Convert to dict format and include enhanced scores
            result_dicts = []
            for item in results:
                item_dict = item.__dict__.copy()
                # Use the best available score for ranking (rerank > hybrid > similarity)
                best_score = item.rerank_score or item.hybrid_score or item.similarity
                item_dict['final_score'] = best_score
                item_dict['similarity'] = item.similarity  # Keep original similarity
                result_dicts.append(item_dict)
            return result_dicts
        except Exception as e:
            print(f"Vector search error: {e}")
            import traceback
            traceback.print_exc()
            return []


async def extract_keywords_and_rewrite(state: AgentState) -> AgentState:
    print("--- Node: Keywords, Rewrite, Intents ---")
    # Detect intents on the raw input first
    intents0 = await call_gemini_detect_intents(state["query"], state.get("history", []))
    if intents0 == [QueryIntent.GREETING.value]:
        print("Pure greeting detected; skipping search.")
        return {**state, "effective_query": state["query"], "keywords": [], "intents": intents0}

    effective = await call_gemini_rewrite_with_history(state["query"], state.get("history", []))
    keywords = await call_gemini_for_keywords(effective)
    # Re-evaluate intents after rewrite (usually drops greeting if mixed)
    intents = await call_gemini_detect_intents(effective, state.get("history", []))
    print(f"  -> Effective query: {effective}")
    print(f"  -> Keywords: {keywords}")
    print(f"  -> Intents: {intents}")
    return {**state, "effective_query": effective, "keywords": keywords, "intents": intents}


# Global vector agent instance - initialized once per process
_global_vector_agent = None

def get_vector_agent():
    global _global_vector_agent
    if _global_vector_agent is None:
        _global_vector_agent = VectorSearchAgent()
    return _global_vector_agent

async def execute_search(state: AgentState) -> Dict[str, Any]:
    print("--- Node: Search Execution ---")
    if set(state.get("intents", [])) == {QueryIntent.GREETING.value}:
        print("Pure greeting; skipping search.")
        return {"ks_results": [], "vector_results": []}
    want_pool = 60  # collect enough for several pages (15 per page)

    # Run both searches simultaneously using shared vector agent
    ks_agent = KSSearchAgent()
    vec_agent = get_vector_agent()  # Reuse the same instance

    ks_task = asyncio.create_task(
        ks_agent.run(state["effective_query"], state.get("keywords", []), want=want_pool)
    )
    vec_task = asyncio.create_task(
        vec_agent.run(query=state["effective_query"], want=want_pool, context={"raw": True})
    )

    # Wait for both searches to complete
    ks_results_data, vec_results = await asyncio.gather(ks_task, vec_task)
    all_ks_results = ks_results_data.get("combined_results", [])

    print(f"Search completed: KS results={len(all_ks_results)}, Vector results={len(vec_results)}")
    return {"ks_results": all_ks_results, "vector_results": vec_results}


def calculate_advanced_score(result: dict, query: str, intents: List[str]) -> float:
    """
    Calculate an advanced score based on multiple factors including:
    - Original similarity/relevance score
    - Metadata relevance to query
    - Content relevance to query
    - Source reliability (if available)
    - Temporal factors (if available)
    """
    import re
    from typing import Any

    # Base score from original system
    base_score = result.get("similarity", 0) if "similarity" in result else result.get("_score", 0)

    # Normalize base score to 0-1 range if needed
    if base_score > 1:
        base_score = min(base_score / 10.0, 1.0)  # Assuming scores are typically 0-10 scale

    # Extract text content for relevance checking
    title = result.get("title_guess", "") or result.get("title", "") or ""
    content = result.get("content", "") or ""
    metadata_text = ""

    # Extract metadata for relevance
    metadata = result.get("metadata", {}) or {}
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, (str, list)):
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value if isinstance(v, str))
                metadata_text += f" {value}"

    # Combine all text for relevance analysis
    all_text = f"{title} {content} {metadata_text}".lower()
    query_lower = query.lower()

    # Calculate query term matching
    query_terms = query_lower.split()
    matched_terms = 0
    total_terms = len(query_terms)

    for term in query_terms:
        if term and term in all_text:
            matched_terms += 1

    # Term matching score (0-1)
    term_match_score = matched_terms / total_terms if total_terms > 0 else 0.0

    # Exact phrase matching bonus
    phrase_bonus = 0.0
    if query_lower in all_text:
        phrase_bonus = 0.2  # Bonus for exact phrase match

    # Intent-specific scoring
    intent_bonus = 0.0
    if intents:
        # Check if result contains intent-related keywords
        if QueryIntent.ACCESS_DOWNLOAD.value in intents:
            access_keywords = ["download", "access", "license", "api", "format"]
            if any(keyword in all_text for keyword in access_keywords):
                intent_bonus += 0.1

        if QueryIntent.METADATA_QUERY.value in intents:
            metadata_keywords = ["metadata", "format", "spec", "parameter", "method"]
            if any(keyword in all_text for keyword in metadata_keywords):
                intent_bonus += 0.1

    # Combine scores with weights
    # Base score: 50%, Term matching: 30%, Phrase bonus: 15%, Intent bonus: 5%
    final_score = (base_score * 0.5) + (term_match_score * 0.3) + (phrase_bonus * 0.15) + (intent_bonus * 0.05)

    return min(final_score, 1.0)  # Cap at 1.0


def fuse_results(state: AgentState) -> AgentState:
    print("--- Node: Result Fusion ---")
    ks_results = state.get("ks_results", [])
    vector_results = state.get("vector_results", [])
    query = state.get("effective_query", "")
    intents = state.get("intents", [])

    combined: Dict[str, dict] = {}

    # Process vector results with advanced scoring
    for res in vector_results:
        if isinstance(res, dict):
            doc_id = res.get("id") or res.get("_id") or f"vec_{len(combined)}"
            # Calculate advanced score for vector result
            advanced_score = calculate_advanced_score(res, query, intents)
            combined[doc_id] = {**res, "final_score": advanced_score, "source_type": "vector"}

    # Process KS results with advanced scoring
    for res in ks_results:
        if isinstance(res, dict):
            doc_id = res.get("_id") or res.get("id") or f"ks_{len(combined)}"
            if doc_id in combined:
                # Combine scores if result exists from both sources
                existing_score = combined[doc_id]["final_score"]
                new_score = calculate_advanced_score(res, query, intents)
                # Average the scores when result appears in both sources
                combined[doc_id].update({**res, "final_score": (existing_score + new_score) / 2.0, "source_type": "combined"})
            else:
                # Calculate advanced score for KS result
                advanced_score = calculate_advanced_score(res, query, intents)
                combined[doc_id] = {**res, "final_score": advanced_score, "source_type": "knowledge_space"}

    # Sort by final score in descending order
    all_sorted = sorted(combined.values(), key=lambda x: x.get("final_score", 0), reverse=True)
    print(f"Results summary: KS={len(ks_results)}, Vector={len(vector_results)}, Combined={len(all_sorted)}")
    page_size = 15
    return {**state, "all_results": all_sorted, "final_results": all_sorted[:page_size]}


async def generate_final_response(state: AgentState) -> AgentState:
    print("--- Node: Response Generation ---")
    intents = state.get("intents", [QueryIntent.DATA_DISCOVERY.value])
    if set(intents) == {QueryIntent.GREETING.value}:
        response = (
            "Hey! I can help you find neuroscience datasets.\n\n"
            "Try examples:\n"
            "- rat electrophysiology in hippocampus\n"
            "- human EEG visual stimulus\n"
            "- fMRI datasets with CC0 or PDDL license\n"
            "- datasets from EBRAINS \n"
        )
        return {**state, "final_response": response}
    raw_results = state.get("final_results", [])
    start_number = state.get("__start_number__", 1)
    prev_text = state.get("__previous_text__", "")
    print(f"Generating response for {len(raw_results)} final results, start={start_number}, intents={intents}")
    response = await call_gemini_for_final_synthesis(
        state["effective_query"], raw_results, intents, start_number=start_number, previous_text=prev_text
    )
    return {**state, "final_response": response}


class NeuroscienceAssistant:
    def __init__(self):
        self.chat_history: Dict[str, List[str]] = {}
        self.session_memory: Dict[str, Dict[str, Any]] = {}
        self.user_feedback: Dict[str, List[Dict]] = {}  # Store user feedback for learning
        self.entity_memory: Dict[str, Dict[str, Any]] = {}  # Track entities mentioned in conversation
        self.research_context: Dict[str, Dict[str, Any]] = {}  # Track research threads and objectives
        self.citation_tracker: Dict[str, Dict[str, List]] = {}  # Track citations and references
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("prepare", extract_keywords_and_rewrite)
        workflow.add_node("search", execute_search)
        workflow.add_node("fusion", fuse_results)
        workflow.add_node("generate_response", generate_final_response)
        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "search")
        workflow.add_edge("search", "fusion")
        workflow.add_edge("fusion", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def reset_session(self, session_id: str):
        self.chat_history.pop(session_id, None)
        self.session_memory.pop(session_id, None)
        self.entity_memory.pop(session_id, None)
        self.research_context.pop(session_id, None)
        self.citation_tracker.pop(session_id, None)

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract key entities from user query using simple pattern matching"""
        import re

        # Common neuroscience entities and patterns
        patterns = [
            r'\b(human|mouse|rabbit|rat|monkey|drosophila|c\.elegans)\b',  # species
            r'\b(hippocampus|cortex|amygdala|thalamus|striatum|cerebellum|brainstem)\b',  # brain regions
            r'\b(EEG|fMRI|MRI|PET|MEG|DTI|NIRS)\b',  # imaging techniques
            r'\b(electrophysiology|patch clamp|calcium imaging|optogenetics)\b',  # techniques
            r'\b(BIDS|NWB|NIfTI|DICOM|HDF5)\b',  # formats
            r'\b(CC0|PDDL|license|open access)\b',  # licenses
            r'\b(gene|protein|receptor|channel)\b',  # molecular
        ]

        entities = []
        query_lower = query.lower()
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            entities.extend(matches)

        return list(set(entities))  # Remove duplicates

    def _update_entity_memory(self, session_id: str, query: str, results: List[dict]):
        """Update entity memory based on query and results"""
        if session_id not in self.entity_memory:
            self.entity_memory[session_id] = {"entities": [], "preferences": {}, "context": {}}

        # Extract entities from query
        query_entities = self._extract_entities_from_query(query)
        self.entity_memory[session_id]["entities"].extend(query_entities)

        # Track user preferences based on results they engage with
        for result in results[:3]:  # Focus on top results
            source = result.get("datasource_name", result.get("source", "unknown"))
            if source not in self.entity_memory[session_id]["preferences"]:
                self.entity_memory[session_id]["preferences"][source] = 0
            self.entity_memory[session_id]["preferences"][source] += 1

    def _update_research_context(self, session_id: str, query: str, results: List[dict]):
        """Update research context with objectives and thread information"""
        if session_id not in self.research_context:
            self.research_context[session_id] = {
                "research_objectives": [],
                "research_threads": [],
                "current_focus": "",
                "research_summary": "",
                "next_steps": []
            }

        # Update research objectives based on query
        if query.strip():
            objectives = self.research_context[session_id]["research_objectives"]
            if query not in objectives:
                objectives.append(query)
                # Keep only the most recent 5 objectives
                if len(objectives) > 5:
                    objectives.pop(0)

        # Track research threads
        threads = self.research_context[session_id]["research_threads"]
        if results:
            # Add top result as part of current research thread
            top_result = results[0] if results else {}
            if top_result:
                thread_entry = {
                    "query": query,
                    "result_id": top_result.get("id", ""),
                    "title": top_result.get("title_guess", ""),
                    "timestamp": datetime.now().isoformat()
                }
                threads.append(thread_entry)
                # Keep only the most recent 10 thread entries
                if len(threads) > 10:
                    threads.pop(0)

        # Update current focus
        self.research_context[session_id]["current_focus"] = query

    def _update_citation_tracker(self, session_id: str, results: List[dict]):
        """Track citations and references mentioned in conversation"""
        if session_id not in self.citation_tracker:
            self.citation_tracker[session_id] = {
                "citations": [],
                "referenced_datasets": [],
                "accessed_datasets": []
            }

        tracker = self.citation_tracker[session_id]

        for result in results:
            dataset_id = result.get("id", "")
            title = result.get("title_guess", "")
            source = result.get("datasource_name", "")

            # Add to referenced datasets if not already there
            ref_key = f"{dataset_id}:{title}"
            if ref_key not in tracker["referenced_datasets"]:
                tracker["referenced_datasets"].append(ref_key)

            # Add to accessed datasets (with timestamp)
            access_entry = {
                "dataset_id": dataset_id,
                "title": title,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }
            tracker["accessed_datasets"].append(access_entry)

            # Keep only the most recent 20 accessed datasets
            if len(tracker["accessed_datasets"]) > 20:
                tracker["accessed_datasets"] = tracker["accessed_datasets"][-20:]

    def _generate_research_summary(self, session_id: str) -> str:
        """Generate a research summary for the current session"""
        if session_id not in self.research_context:
            return ""

        context = self.research_context[session_id]
        summary_parts = []

        # Add research objectives
        if context["research_objectives"]:
            summary_parts.append("Research Objectives:")
            for i, obj in enumerate(context["research_objectives"][-3:], 1):  # Last 3 objectives
                summary_parts.append(f"  {i}. {obj}")

        # Add current focus
        if context["current_focus"]:
            summary_parts.append(f"\nCurrent Focus: {context['current_focus']}")

        # Add accessed datasets
        if session_id in self.citation_tracker:
            accessed = self.citation_tracker[session_id]["accessed_datasets"]
            if accessed:
                summary_parts.append(f"\nAccessed Datasets ({len(accessed)}):")
                for i, dataset in enumerate(accessed[-5:], 1):  # Last 5 accessed
                    summary_parts.append(f"  {i}. {dataset['title']} [{dataset['source']}]")

        return "\n".join(summary_parts)

    def _generate_next_steps_suggestions(self, session_id: str, query: str) -> List[str]:
        """Generate next-step suggestions based on research context"""
        suggestions = []

        # Get research context
        context = self.research_context.get(session_id, {})
        tracker = self.citation_tracker.get(session_id, {})

        # If there are accessed datasets, suggest related exploration
        if tracker.get("accessed_datasets"):
            last_dataset = tracker["accessed_datasets"][-1] if tracker["accessed_datasets"] else {}
            if last_dataset:
                suggestions.append(f"Explore related datasets to '{last_dataset['title']}'")
                suggestions.append(f"Look for datasets with similar methodology to '{last_dataset['title']}'")

        # If there are research objectives, suggest continuation
        if context.get("research_objectives"):
            last_objective = context["research_objectives"][-1] if context["research_objectives"] else ""
            if last_objective:
                suggestions.append(f"Continue research on '{last_objective}'")
                suggestions.append(f"Find datasets that complement '{last_objective}'")

        # General suggestions based on query
        if "human" in query.lower():
            suggestions.append("Explore human-specific datasets")
        elif "mouse" in query.lower():
            suggestions.append("Look for mouse model datasets")

        if "fMRI" in query.upper() or "MRI" in query.upper():
            suggestions.append("Find related neuroimaging datasets")

        # Limit to 3 suggestions
        return suggestions[:3]

    def _get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation context for current session"""
        context = {
            "entities": self.entity_memory.get(session_id, {}).get("entities", []),
            "preferences": self.entity_memory.get(session_id, {}).get("preferences", {}),
            "recent_queries": [],
            "research_context": self.research_context.get(session_id, {}),
            "citations": self.citation_tracker.get(session_id, {}).get("referenced_datasets", []),
            "research_summary": self._generate_research_summary(session_id)
        }

        # Get recent queries from chat history
        if session_id in self.chat_history:
            recent_chats = self.chat_history[session_id][-6:]  # Last 3 exchanges
            for chat in recent_chats:
                if chat.startswith("User:"):
                    context["recent_queries"].append(chat[6:])  # Remove "User: " prefix

        return context

    def _resolve_pronouns(self, query: str, session_id: str) -> str:
        """Simple pronoun resolution to improve query understanding"""
        context = self._get_conversation_context(session_id)

        # Replace pronouns with relevant entities if available
        query_lower = query.lower()

        # If user says "these" or "those", try to map to recent entities
        if "these" in query_lower or "those" in query_lower:
            if context["entities"]:
                # Replace with most recent entity
                most_recent_entity = context["entities"][-1] if context["entities"] else ""
                if most_recent_entity:
                    query = re.sub(r'\b(these|those)\b', most_recent_entity, query, flags=re.IGNORECASE)

        # If user says "it" or "this", try to map to recent entities
        if "it" in query_lower or "this" in query_lower:
            if context["entities"]:
                most_recent_entity = context["entities"][-1] if context["entities"] else ""
                if most_recent_entity:
                    query = re.sub(r'\b(it|this)\b', most_recent_entity, query, flags=re.IGNORECASE)

        return query

    def collect_feedback(self, session_id: str, query: str, response: str, result_ids: List[str], rating: int, feedback_text: str = ""):
        """Collect user feedback to improve future results"""
        feedback_entry = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "result_ids": result_ids,
            "rating": rating,  # 1-5 scale
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }

        if session_id not in self.user_feedback:
            self.user_feedback[session_id] = []

        self.user_feedback[session_id].append(feedback_entry)

        # Keep only recent feedback (last 100 entries per session)
        if len(self.user_feedback[session_id]) > 100:
            self.user_feedback[session_id] = self.user_feedback[session_id][-100:]

    async def handle_chat(self, session_id: str, query: str, reset: bool = False) -> str:
        try:
            if reset:
                self.reset_session(session_id)
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []

            # Resolve pronouns in query for better understanding
            resolved_query = self._resolve_pronouns(query, session_id)

            more_count = _is_more_query(query)
            mem = self.session_memory.get(session_id, {})
            if more_count is not None or (query.strip().lower() in {"more", "next", "continue", "more please", "show more", "keep going"}):
                all_results = mem.get("all_results", [])
                if not all_results:
                    return "There are no earlier results to continue. Ask me for a dataset (e.g., 'human EEG BIDS')."
                page_size = more_count or mem.get("page_size", 15)
                page = mem.get("page", 1) + 1
                start = (page - 1) * page_size
                batch = all_results[start:start + page_size]
                if not batch:
                    return "You've reached the end of the results. Try refining the query."
                intents = mem.get("intents", [QueryIntent.DATA_DISCOVERY.value])
                effective_query = mem.get("effective_query", "")
                prev_text = mem.get("last_text", "")

                text = await call_gemini_for_final_synthesis(
                    effective_query, batch, intents, start_number=start + 1, previous_text=prev_text
                )
                mem.update({
                    "page": page,
                    "page_size": page_size,
                    "last_text": f"{prev_text}\n\n{text}"[-12000:],
                })
                self.session_memory[session_id] = mem
                self.chat_history[session_id].extend([f"User: {query}", f"Assistant: {text}"])
                if len(self.chat_history[session_id]) > 20:
                    self.chat_history[session_id] = self.chat_history[session_id][-20:]
                return text

            initial_state: AgentState = {
                "session_id": session_id,
                "query": resolved_query,  # Use resolved query
                "history": self.chat_history[session_id][-10:],
                "keywords": [],
                "effective_query": "",
                "intents": [],
                "ks_results": [],
                "vector_results": [],
                "final_results": [],
                "all_results": [],
                "__start_number__": 1,
                "__previous_text__": "",
                "final_response": "",
            }
            final_state = await self.graph.ainvoke(initial_state)
            response_text = final_state.get("final_response", "I encountered an unexpected empty response.")

            # Update entity memory with query and results
            all_results = final_state.get("all_results", [])
            self._update_entity_memory(session_id, query, all_results)

            # Update research context
            self._update_research_context(session_id, query, all_results)

            # Update citation tracker
            self._update_citation_tracker(session_id, all_results)

            self.session_memory[session_id] = {
                "all_results": all_results,
                "page": 1,
                "page_size": 15,
                "effective_query": final_state.get("effective_query", initial_state["query"]),
                "keywords": final_state.get("keywords", []),
                "intents": final_state.get("intents", [QueryIntent.DATA_DISCOVERY.value]),
                "last_text": response_text,
            }

            self.chat_history[session_id].extend([f"User: {query}", f"Assistant: {response_text}"])
            if len(self.chat_history[session_id]) > 20:
                self.chat_history[session_id] = self.chat_history[session_id][-20:]
            return response_text
        except Exception as e:
            print(f"Error in handle_chat: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
