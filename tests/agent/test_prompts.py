from arc.agent.prompts import get_reliability_block


def test_voice_mode_uses_friendly_spoken_summary_instructions():
    prompt = get_reliability_block("main", voice_mode=True)

    assert "friendly assistant speaking out loud" in prompt
    assert "warm, natural, and conversational" in prompt
    assert "under 25 words" in prompt
    assert 'avoid robotic words like "implemented", "completed", or "addressed"' in prompt
    assert "[spoken]I found the issue and fixed it. Voice mode should work properly now.[/spoken]" in prompt
