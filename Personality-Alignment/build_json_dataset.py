import pandas as pd
import json

def load_data():
    """Load all data splits from JSONL files"""
    conversations = pd.read_json("conversations.jsonl", lines=True)
    metadata = pd.read_json("metadata.jsonl", lines=True)
    return {
        "conversations": conversations,
        "metadata": metadata
    }

def main():
    # Load all data splits
    data = load_data()
    conversations_df = data["conversations"]
    metadata_df = data["metadata"]

    # Filter metadata for English language
    english_convo_ids = set(
        metadata_df[metadata_df["en_flag"] == True]["conversation_id"]
    )
    # Filter conversations for English only
    english_conversations = conversations_df[
        conversations_df["conversation_id"].isin(english_convo_ids)
    ]

    # Build output dataset
    output = []
    for _, convo in english_conversations.iterrows():
        user_id = convo["user_id"]
        conversations = convo["conversation_history"]
        conversation_type = convo.get("conversation_type", "")
        output.append(
            {
                "user_id": user_id,
                "conversation_type": conversation_type,
                "opening_prompt": convo["opening_prompt"],
                "conversations": conversations,
            }
        )

    # Write to file
    with open("roleplay_dataset_en.json", "w") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
