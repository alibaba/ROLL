import json
from pathlib import Path
PERSONALITY_KEY_MEANINGS = {
    "values": "reflects my values or cultural perspectives",
    "creativity": "produces responses that are creative and inspiring",
    "fluency": "produces responses that are well-written and coherent",
    "factuality": "produces factual and informative responses",
    "diversity": "summarises multiple viewpoints or different worldviews",
    "safety": "produces responses that are safe and do not risk harm",
    "personalisation": "learns from our conversations and feels personalised to me",
    "helpfulness": "produces responses that are helpful and relevant to my requests",
}


def build_profile_text(record: dict) -> str:
    """将 Survey 第 15‑30 项整理成可读的人物档案文本"""
    get = lambda k: record.get(k) or ""          # 避免 KeyError，同时过滤 None
    parts = []

    # 1. 基本人口属性 15‑20
    basic = {
        "Age": get("age"),
        "Education": get("education"),
        "Employment": get("employment_status"),
        "Marital status": get("marital_status"),
        "English proficiency": get("english_proficiency"),
        "Gender": get("gender"),
    }
    parts.extend(f"{k}: {v}" for k, v in basic.items() if v)

    # 2. 宗教 21‑24
    religion = get("religion_simplified") or get("religion_categorised") or get("religion_self_described")
    if religion:
        parts.append(f"Religion: {religion}")

    # 3. 种族 25‑28
    ethnicity = get("ethnicity_simplified") or get("ethnicity_categorised") or get("ethnicity_self_described")
    if ethnicity:
        parts.append(f"Ethnicity: {ethnicity}")

    # 4. 出生地 30（如需，可加上 residence 等 33+ 字段）
    birth_country = get("location_birth_country")
    if birth_country:
        parts.append(f"Country of birth: {birth_country}")
        
    # 5. 现居地 33‑35
    reside_country = get("location_reside_country")          # 33
    reside_iso     = get("location_reside_countryISO")       # 34
    reside_sub     = get("location_reside_subregion")        # 35
    if reside_country:
        parts.append(f"Country of residence: {reside_country}")
    if reside_sub:
        parts.append(f"Sub‑region of residence: {reside_sub}")
    if reside_iso:        # 可选：只在有值时附上三字母 ISO
        parts.append(f"Country ISO code: {reside_iso}")
        
    # c) Stated preferences（人格相关滑条）
    stated = get("stated_prefs")
    if stated:
        pref_lines = []
        for key, meaning in PERSONALITY_KEY_MEANINGS.items():
            if key in stated:
                score = stated[key]
                pref_lines.append(
                    f"{key.capitalize()} ({meaning}): {score}/100 importance"
                )
        if pref_lines:
            parts.append("Stated LLM preferences:\n" + "\n".join(pref_lines))
        
        
    # 6. 受访者自述（第 13 项）可保留在最后
    if get("self_description"):
        parts.append("\nSelf‑description:\n" + get("self_description").strip())

    return "\n".join(parts)



def main():
    output = {}
    with Path("prism-data/survey.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            output[rec["user_id"]] = build_profile_text(rec)

    Path("profile_with_perference.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")



if __name__ == "__main__":
    main()
