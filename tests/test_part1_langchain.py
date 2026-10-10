"""Part 1 lessons (LangChain essentials), each driven offline through its `run()` with a scripted model."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .conftest import FakeChatModel, tool_call


def test_1_1_models_and_messages(lesson, capsys):
    llm = FakeChatModel(replies=["RAG means ...", "Hi Ada", "Your name is Ada.", "a", "b", "c"])
    fast = FakeChatModel(replies=["A notebook groups notes."])

    result = lesson("1.1").run(llm, fast)

    assert result["reply"].usage_metadata["total_tokens"] == 15
    # The model is stateless: the third call only "remembers" the name because we resent the history.
    third_call = llm.seen[2]
    assert [type(m) for m in third_call] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert result["batch"] == ["a", "b", "c"]
    assert result["fast"] == "A notebook groups notes."
    assert "tokens" in capsys.readouterr().out


def test_1_2_prompts_and_lcel(lesson):
    llm = FakeChatModel(replies=["Yes, offline works.", "Yes, both apps.", "Answer", "Offline access"])

    result = lesson("1.2").run(llm)

    first_prompt = llm.seen[0]
    assert isinstance(first_prompt[0], SystemMessage) and "Nimbus Notes" in first_prompt[0].content
    # system + 2 few-shot pairs + 2 history messages (MessagesPlaceholder) + the new question
    second_prompt = llm.seen[1]
    assert len(second_prompt) == 8
    assert "Category: billing" in second_prompt[2].content
    assert second_prompt[-1].content == "And Linux?"
    # The two parallel branches run concurrently, so which scripted reply each gets is not fixed.
    assert set(result["parallel"]) == {"answer", "title"}
    assert set(result["parallel"].values()) == {"Answer", "Offline access"}


def test_1_3_structured_output(lesson):
    module = lesson("1.3")  # import once: each import defines a fresh Triage class
    t = module.Triage(category="sync", sentiment="negative", urgent=True, summary="Sync broken")
    llm = FakeChatModel(structured={module.Triage: [t, t, t, t]})

    result = module.run(llm)

    assert result["single"].urgent is True
    assert len(result["batch"]) == 3


def test_1_4_multimodal_sends_text_and_image_blocks(lesson):
    llm = FakeChatModel(replies=["The screenshot shows error SYNC-403."])

    answer = lesson("1.4").run(llm)

    blocks = llm.seen[0][-1].content
    assert {b["type"] for b in blocks} == {"text", "image"}
    image = next(b for b in blocks if b["type"] == "image")
    assert image["mime_type"] == "image/png" and len(image["base64"]) > 100
    assert "SYNC-403" in answer


def test_1_5_rag_pipeline(lesson, fake_retriever):
    llm = FakeChatModel(replies=["Within 30 days [02-plans-and-billing.md]"])

    result = lesson("1.5").run(llm, fake_retriever)

    assert result["chunks"] > result["files"] == 8
    system = llm.seen[0][0].content
    assert "[source:" in system  # retrieved context was stuffed into the prompt
    assert result["answer"].startswith("Within 30 days")


def test_1_6_tools_and_agent_calls_both_tools(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            tool_call("quote_price", {"plan": "team", "members": 12, "billing": "annual"}),  # bind_tools demo
            tool_call("search_docs", {"query": "SSO Okta plan"}, "c1"),
            tool_call("quote_price", {"plan": "team", "members": 12, "billing": "annual"}, "c2"),
            "You need Team; it costs $921.60/year.",
        ]
    )

    result = lesson("1.6").run(llm, fake_retriever)

    assert result["manual_call"]["name"] == "quote_price"
    assert result["tools_called"] == ["search_docs", "quote_price"]
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "$921.60" in tool_results[-1].content
    assert result["answer"].endswith("$921.60/year.")


def test_1_7_streaming_modes(lesson, fake_retriever):
    llm = FakeChatModel(
        replies=[
            "Notes sync every few seconds",
            tool_call("search_docs", {"query": "offline"}),
            "Yes, offline works on desktop.",
        ]
    )

    result = lesson("1.7").run(llm, fake_retriever)

    assert len(result["tokens"]) > 1  # several chunks, not one blob
    assert "".join(result["tokens"]) == "Notes sync every few seconds"
    assert result["steps"] == ["model", "tools", "model"]
    assert any("Searching" in c for c in result["custom"])  # emitted by the tool via get_stream_writer


def test_1_8_short_term_memory_is_per_thread(lesson):
    llm = FakeChatModel(replies=["Noted.", "You have 12 seats.", "I don't know your seat count."])

    result = lesson("1.8").run(llm)

    second_turn = llm.seen[1]
    assert any("12 seats" in m.text for m in second_turn if isinstance(m, HumanMessage))
    other_thread = llm.seen[2]
    assert not any("12 seats" in m.text for m in other_thread)
    assert result["saved_messages"] == 4


def test_1_9_dynamic_prompt_depends_on_runtime_context(lesson):
    llm = FakeChatModel(replies=["Upgrade to Team for SSO.", "Your account manager can enable SSO."])

    lesson("1.9").run(llm)

    free_prompt, enterprise_prompt = llm.seen[0][0].content, llm.seen[1][0].content
    assert "Free" in free_prompt and "Enterprise" in enterprise_prompt
    assert free_prompt != enterprise_prompt


def test_1_10_capstone_support_agent(lesson, fake_retriever):
    capstone = lesson("1.10")
    answer = {
        "answer": "Deleted notes stay in the trash for 30 days.",
        "sources": ["07-troubleshooting.md"],
        "needs_ticket": False,
    }
    llm = FakeChatModel(
        replies=[
            tool_call("search_docs", {"query": "trash retention"}),
            tool_call("SupportAnswer", answer, "c2"),
            tool_call("SupportAnswer", {**answer, "answer": "Admins can restore them for 30 more days."}, "c3"),
        ]
    )

    results = capstone.run(llm, fake_retriever, conversation=["How long is trash kept?", "Can an admin restore them?"])

    assert [r.answer for r in results] == [answer["answer"], "Admins can restore them for 30 more days."]
    # Turn 2 saw turn 1 (memory) and the plan-aware system prompt.
    assert any("How long is trash kept?" in m.text for m in llm.seen[-1])
    assert "Team" in llm.seen[0][0].content
