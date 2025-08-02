import re


def default_parser_action_func(text, action_pattern, action_lookup, special_token_list):
    action = None
    match = re.search(action_pattern, text, re.DOTALL)
    if not match:
        action_info = {
            "action": action,
            "action_content": "",
            "think_content": ""
        }
        return action_info
    if len(match.groups()) == 1:
        think_content, action_content = "", match.group(1)
    else:
        think_content, action_content = match.group(1), match.group(2)
    action_content_strip = action_content.strip()
    think_content_strip = think_content.strip()

    if special_token_list is not None:
        for special_token in special_token_list:
            action_content_strip = action_content.replace(special_token, "").strip()
            think_content_strip = think_content.replace(special_token, "").strip()

    action = action_content_strip
    if action_lookup is not None:
        action = None
        rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
        if action_content_strip.lower() in rev_action_lookup:
            action = rev_action_lookup[action_content_strip.lower()]

    action_info = {
        "action": action,
        "action_content": action_content,
        "think_content": think_content,
        "action_content_strip": action_content_strip,
        "think_content_strip": think_content_strip
    }
    return action_info