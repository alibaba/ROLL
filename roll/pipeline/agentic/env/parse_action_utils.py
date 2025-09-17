import re


def default_parser_action_func(text, action_pattern, action_lookup, special_token_list):
    if special_token_list is not None:
        for special_token in special_token_list:
            text = text.replace(special_token, "").strip()

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
        think_content, action_content = "", match.group(1).strip()
    else:
        think_content, action_content = match.group(1).strip(), match.group(2).strip()
    action_content = action_content.strip()
    think_content = think_content.strip()

    action = action_content
    if action_lookup is not None:
        action = None
        rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
        if action_content.lower() in rev_action_lookup:
            action = rev_action_lookup[action_content.lower()]

    action_info = {
        "action": action,
        "action_content": action_content,
        "think_content": think_content,
    }
    return action_info