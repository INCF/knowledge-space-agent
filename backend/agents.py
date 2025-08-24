# agents.py
import os
import re
import json
from enum import Enum
from typing import Dict, List, Optional, TypedDict, Any

from langgraph.graph import StateGraph, END

from ks_search_tool import general_search, global_fuzzy_keyword_search
from retrieval import Retriever

# --- LLM (Gemini) client setup ----------------------------------------------
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


def _require_llm_creds():
    if _use_vertex():
        if not os.getenv("GCP_PROJECT_ID"):
            raise RuntimeError("GCP_PROJECT_ID must be set for Vertex mode.")
        # Ensure service account path is set relative to this file (if not provided)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "service-account.json")

    else:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY must be set for API-key mode.")


_GENAI_CLIENT = None


def _get_genai_client():
    global _GENAI_CLIENT
    if _GENAI_CLIENT is not None:
        return _GENAI_CLIENT
    if _use_vertex():
        project = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_REGION") or "europe-west4"
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            sa_path = os.path.join(os.path.dirname(__file__), "service-account.json")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        _GENAI_CLIENT = genai.Client(vertexai=True, project=project, location=location)
    else:
        _GENAI_CLIENT = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _GENAI_CLIENT


FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
FLASH_LITE_MODEL = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-2.5-flash-lite")


# --- Query intent/types ------------------------------------------------------
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


# --- LLM calls using google.genai -------------------------------------------
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
    """
    Final rendering with gemini-2.5-flash. Shows up to 15 datasets per call.
    Removes 'Relevance' and 'Matched on' sections.
    """
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
        "- **Key Details:**\n"
        "  - Species: ...\n"
        "  - Brain Regions: ...\n"
        "  - Technique or Modality: ...\n"
        "  - Data Formats: ...\n"
        "  - License: ...\n"
        "  - Subjects/Samples: ...\n"
        "  - Tasks/Conditions: ...\n"
        "  - Recording Specs: ...\n"
        "  - Authors: ...\n"
        "  - Year: ...\n"
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
    # Best-effort continuation if the model stopped early; SDK may not expose finish_reason consistently.
    if not text:
        raise RuntimeError("Gemini synthesis returned empty text.")
    return text


# --- Agent state and search/fuse/response pipeline ---------------------------
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
        # We’ll gather more than one page to fill several "more" requests; first page is capped to 15 later.
        try:
            general = general_search(query, top_k=min(want, 50), enrich_details=True).get("combined_results", [])
        except Exception as e:
            print(f"General search error: {e}")
            general = []
        try:
            fuzzy = global_fuzzy_keyword_search(keywords, top_k=min(want, 50))
        except Exception as e:
            print(f"Fuzzy config search error: {e}")
            fuzzy = []
        # return up to 'want' combined (dedupe can be added if needed)
        return {"combined_results": (general + fuzzy)[: max(want, 15)]}


class VectorSearchAgent:
    def __init__(self):
        self.retriever = Retriever()
        self.is_enabled = self.retriever.is_enabled

    def run(self, query: str, want: int, context: Optional[Dict] = None) -> List[dict]:
        if not self.is_enabled:
            return []
        try:
            results = self.retriever.search(query=query, top_k=min(want, 50), context={"raw": True})
            return [item.__dict__ if hasattr(item, "__dict__") else item for item in results]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []


async def extract_keywords_and_rewrite(state: AgentState) -> AgentState:
    print("--- Node: Keywords, Rewrite, Intents ---")
    # Detect intents on the raw input first (to catch pure greeting)
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


async def execute_search(state: AgentState) -> Dict[str, Any]:
    print("--- Node: Search Execution ---")
    if set(state.get("intents", [])) == {QueryIntent.GREETING.value}:
        print("Pure greeting; skipping search.")
        return {"ks_results": [], "vector_results": []}
    want_pool = 60  # collect enough for several pages (15 per page)
    ks_agent = KSSearchAgent()
    ks_results_data = await ks_agent.run(state["effective_query"], state.get("keywords", []), want=want_pool)
    all_ks_results = ks_results_data.get("combined_results", [])
    vec_agent = VectorSearchAgent()
    vec_results = vec_agent.run(query=state["effective_query"], want=want_pool, context={"raw": True})
    print(f"Search completed: KS results={len(all_ks_results)}, Vector results={len(vec_results)}")
    return {"ks_results": all_ks_results, "vector_results": vec_results}


def fuse_results(state: AgentState) -> AgentState:
    print("--- Node: Result Fusion ---")
    ks_results = state.get("ks_results", [])
    vector_results = state.get("vector_results", [])
    combined: Dict[str, dict] = {}
    for res in vector_results:
        if isinstance(res, dict):
            doc_id = res.get("id") or res.get("_id") or f"vec_{len(combined)}"
            combined[doc_id] = {**res, "final_score": res.get("similarity", 0) * 0.6}
    for res in ks_results:
        if isinstance(res, dict):
            doc_id = res.get("_id") or res.get("id") or f"ks_{len(combined)}"
            if doc_id in combined:
                combined[doc_id]["final_score"] += res.get("_score", 0) * 0.4
            else:
                combined[doc_id] = {**res, "final_score": res.get("_score", 0) * 0.4}
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
            "- human EEG visual stimulus, BIDS format\n"
            "- fMRI datasets with CC0 or PDDL license\n"
            "- datasets from EBRAINS about DWI\n"
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

    async def handle_chat(self, session_id: str, query: str, reset: bool = False) -> str:
        try:
            if reset:
                self.reset_session(session_id)
            if session_id not in self.chat_history:
                self.chat_history[session_id] = []

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
                    return "You’ve reached the end of the results. Try refining the query."
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
                "query": query,
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

            self.session_memory[session_id] = {
                "all_results": final_state.get("all_results", []),
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
