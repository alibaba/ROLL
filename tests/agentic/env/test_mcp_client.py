import pytest
import json
from roll.pipeline.agentic.env.mcp.mcp_client import MCPClient

@pytest.mark.asyncio
async def test_sokoban_mcp_server_interaction():
    async with MCPClient("http://sokoban-mcp.alibaba-inc.com/sse") as client: 
        tools_list = await client.tools()
        tool_names = [tool.name for tool in tools_list]
        assert "reset" in tool_names, "reset tool not found in server tools"
        assert "play" in tool_names, "play tool not found in server tools"
        # call reset without seed
        raw_reset_result = await client.call_tool("reset")
        reset_result = parse_call_tool_result(raw_reset_result)
        assert "Observation" in reset_result
        print("Reset observation:\n", reset_result["Observation"])
        # call reset with seed=2
        seed = 2
        raw_reset_seed_result = await client.call_tool("reset", {"seed": seed})
        reset_seed_result = parse_call_tool_result(raw_reset_seed_result)
        assert "Observation" in reset_seed_result
        assert reset_seed_result["Observation"] == "######\n#_#_P#\n#_#X_#\n#___O#\n#____#\n######"
        print(f"Reset with seed={seed} observation:\n", reset_seed_result["Observation"])
        
        # call play with action=3 （left）
        await call_play_and_parse(client, 3,
            expected_obs="######\n#_#P_#\n#_#X_#\n#___O#\n#____#\n######"
        )
        # call play with action=2 （down）
        await call_play_and_parse(client, 2,
            expected_obs="######\n#_#__#\n#_#P_#\n#__XO#\n#____#\n######"
        )
        # call play with action=4 （right）
        await call_play_and_parse(client, 4,
            expected_obs="######\n#_#__#\n#_#_P#\n#__XO#\n#____#\n######"
        )
        # call play with action=2 （down）
        await call_play_and_parse(client, 2,
            expected_obs="######\n#_#__#\n#_#__#\n#__XS#\n#____#\n######"
        )
        # call play with action=2 （down）
        await call_play_and_parse(client, 2,
            expected_obs="######\n#_#__#\n#_#__#\n#__XO#\n#___P#\n######"
        )
        # call play with action=3 （left）
        await call_play_and_parse(client, 3,
            expected_obs="######\n#_#__#\n#_#__#\n#__XO#\n#__P_#\n######"
        )       
        # call play with action=3 （left）
        await call_play_and_parse(client, 3,
            expected_obs="######\n#_#__#\n#_#__#\n#__XO#\n#_P__#\n######"
        )     
        # call play with action=1 （up）
        await call_play_and_parse(client, 1,
            expected_obs="######\n#_#__#\n#_#__#\n#_PXO#\n#____#\n######"
        )               
        # call play with action=4 （right）
        await call_play_and_parse(client, 4,
            expected_obs="######\n#_#__#\n#_#__#\n#__P√#\n#____#\n######",
            reward=10.9,
            done=True, 
            success=True
        )   
        
def parse_call_tool_result(call_tool_result):
    """
    Extract the JSON string from CallToolResult
    """
    content_list = getattr(call_tool_result, "content", [])
    text_json_str = None
    for content_item in content_list:
        if hasattr(content_item, "type") and content_item.type == "text":
            text_json_str = content_item.text
            break
    if not text_json_str:
        raise ValueError("No 'text' content found in CallToolResult")
    return json.loads(text_json_str)

async def call_play_and_parse(client, action_code, expected_obs, reward=-0.1, done=False, success=False, effective=True):
    raw = await client.call_tool("play", {"action": action_code})
    res = parse_call_tool_result(raw)
    assert res["Observation"] == expected_obs
    assert res["Reward"] == reward
    assert res.get("Game End") is done
    server_info = res.get("info", {})
    assert server_info.get("action_is_effective") is effective
    assert server_info.get("success") is success
    print(f"Action {action_code} Observation:\n{res['Observation']}")
    print(f"Game ended: {res['Game End']} \ninfo: {res['info']}")
    return res

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_sokoban_mcp_server_interaction())
