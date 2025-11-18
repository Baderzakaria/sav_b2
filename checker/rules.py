"""A2A checker rules for inter-agent consistency validation."""

import json
from typing import Dict, Any, Tuple, List
from models.labels import Labels, CheckerOutput


def load_taxonomy() -> Dict[str, Any]:
    """Load sentiment taxonomy from prompts."""
    try:
        with open("prompts/freemind_prompts_v0.3.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("taxonomy", {})
    except Exception:
        return {}


def get_sentiment_range(sentiment: str, taxonomy: Dict[str, Any]) -> Tuple[int, int]:
    """Get allowed gravity range for a sentiment label."""
    sentiment_opts = taxonomy.get("sentiment", {}).get("options", [])
    for opt in sentiment_opts:
        if opt.get("label") == sentiment:
            return tuple(opt.get("allowed_range", [-10, 10]))
    return (-10, 10)


def get_base_gravity(sentiment: str, taxonomy: Dict[str, Any]) -> int:
    """Get base gravity for a sentiment label."""
    sentiment_opts = taxonomy.get("sentiment", {}).get("options", [])
    for opt in sentiment_opts:
        if opt.get("label") == sentiment:
            return opt.get("base_gravity", 0)
    return 0


def clamp_gravity(score: int, min_val: int, max_val: int) -> int:
    """Clamp gravity score within allowed range."""
    return max(min_val, min(max_val, score))


def apply_checker_rules(
    results: Dict[str, Any],
    context: Dict[str, str],
    taxonomy: Dict[str, Any]
) -> CheckerOutput:
    """Apply A2A consistency rules and return final labels with trace."""
    
    # Extract individual agent results
    utile = results.get("A1", {}).get("utile", True)
    categorie = results.get("A2", {}).get("categorie", "retour_client")
    sentiment = results.get("A3", {}).get("sentiment", "neutre")
    type_probleme = results.get("A4", {}).get("type_probleme", "autre")
    score_gravite = results.get("A5", {}).get("score_gravite", 0)
    affect = results.get("A6", {}).get("affect")
    
    rules_fired = []
    corrections = []
    status = "ok"
    
    # Rule 1: utile=false normalization
    if not utile:
        if categorie != "retour_client":
            corrections.append(f"categorie: {categorie} → retour_client")
            categorie = "retour_client"
        if type_probleme != "autre":
            corrections.append(f"type_probleme: {type_probleme} → autre")
            type_probleme = "autre"
        if score_gravite != 0:
            corrections.append(f"score_gravite: {score_gravite} → 0")
            score_gravite = 0
        rules_fired.append("UTILE_FALSE_NORMALIZE")
    
    # Rule 2: categorie=probleme with positive sentiment → warn
    if categorie == "probleme" and sentiment in ["legerement_positif", "positif", "tres_positif"]:
        rules_fired.append("CATEGORIE_PROBLEME_SENTIMENT_POSITIF_WARN")
        status = "warn"
    
    # Rule 3: sentiment vs gravity alignment with taxonomy
    min_grav, max_grav = get_sentiment_range(sentiment, taxonomy)
    if score_gravite < min_grav or score_gravite > max_grav:
        old_score = score_gravite
        score_gravite = clamp_gravity(score_gravite, min_grav, max_grav)
        corrections.append(f"score_gravite: {old_score} → {score_gravite} (clamped to {sentiment} range)")
        rules_fired.append(f"GRAVITY_CLAMP_{sentiment.upper()}")
    
    # Rule 4: outrage_critique must have high gravity
    if sentiment == "outrage_critique" and score_gravite < 7:
        old_score = score_gravite
        score_gravite = max(7, score_gravite)
        corrections.append(f"score_gravite: {old_score} → {score_gravite} (outrage_critique minimum)")
        rules_fired.append("OUTRAGE_CRITIQUE_MIN_7")
        if status == "ok":
            status = "warn"
    
    # Rule 5: context mentions incident → type=panne
    ctx_text = " ".join([
        context.get("ctx_before", ""),
        context.get("ctx_refs", "")
    ]).lower()
    
    incident_keywords = ["panne", "incident", "rupture", "coupure", "défaillance", "problème technique"]
    if any(kw in ctx_text for kw in incident_keywords):
        if type_probleme != "panne" and categorie == "probleme":
            corrections.append(f"type_probleme: {type_probleme} → panne (context indicates incident)")
            type_probleme = "panne"
            rules_fired.append("CTX_INCIDENT_TYPE_PANNE")
    
    # Rule 6: affect-based adjustments
    if affect:
        emotion = affect.get("emotion_primary")
        sarcasm = affect.get("sarcasm", False)
        
        # Emotion-gravity consistency
        if emotion == "colere" and score_gravite < 3:
            rules_fired.append("EMOTION_COLERE_GRAVITY_LOW_WARN")
            if status == "ok":
                status = "warn"
        
        if emotion == "tristesse" and score_gravite > 4:
            # Tristesse is often less urgent than colere
            rules_fired.append("EMOTION_TRISTESSE_GRAVITY_HIGH_NOTE")
        
        # Sarcasm softens conflicts
        if sarcasm and status == "fail":
            status = "warn"
            rules_fired.append("SARCASM_SOFTEN_FAIL_TO_WARN")
    
    # Rule 7: Multiple warnings → fail
    if len(corrections) >= 3:
        status = "fail"
        rules_fired.append("MULTIPLE_CORRECTIONS_FAIL")
    
    # Build final labels
    try:
        final_labels = Labels(
            utile=utile,
            categorie=categorie,
            sentiment=sentiment,
            type_probleme=type_probleme,
            score_gravite=score_gravite,
            affect=affect
        )
    except Exception as e:
        # Validation failed
        status = "fail"
        rules_fired.append(f"VALIDATION_ERROR: {str(e)}")
        # Fallback to safe defaults
        final_labels = Labels(
            utile=False,
            categorie="retour_client",
            sentiment="neutre",
            type_probleme="autre",
            score_gravite=0,
            affect=None
        )
    
    # Build trace
    a2a_trace = {
        "inputs": {
            "A1": {"utile": results.get("A1", {}).get("utile")},
            "A2": {"categorie": results.get("A2", {}).get("categorie")},
            "A3": {"sentiment": results.get("A3", {}).get("sentiment")},
            "A4": {"type_probleme": results.get("A4", {}).get("type_probleme")},
            "A5": {"score_gravite": results.get("A5", {}).get("score_gravite")},
            "A6": {"affect": results.get("A6", {}).get("affect")}
        },
        "rules_fired": rules_fired,
        "corrections": corrections
    }
    
    return CheckerOutput(
        final=final_labels,
        checker_status=status,
        a2a_trace=a2a_trace
    )

