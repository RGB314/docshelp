"""Part 4 lessons (LangGraph at scale and long-term memory), driven offline with scripted models."""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, SystemMessage

from .conftest import FakeChatModel, tool_call


def test_4_1_parallel_fan_out_and_fan_in(lesson, fake_retriever):
    llm = FakeChatModel(replies=["It's a known Windows issue after sleep; sync works."])

    result = lesson("4.1").run(llm, fake_retriever)

    assert {line.split("]")[0] + "]" for line in result["context"]} == {"[docs]", "[changelog]", "[status]"}
    assert len(llm.seen) == 1  # the answer node ran once, after all three branches
    assert result["steps"][0] == {"search_docs", "search_changelog", "check_status"}  # one parallel step
    assert result["answer"].startswith("It's a known Windows issue")


def test_4_2_subgraphs_share_keys_and_keep_private_state(lesson):
    llm = FakeChatModel(replies=["REPORT", "FAQ"])  # the two subgraphs run in parallel: order not fixed

    result = lesson("4.2").run(llm)

    assert {result["failure_report"], result["faq_suggestions"]} == {"REPORT", "FAQ"}
    assert len(result["reviewed_ids"]) == 7  # 5 open + 2 solved, merged by the parent's reducer
    assert "open_tickets" not in result and "questions" not in result  # private subgraph state
    assert {"failures", "faq"} <= set(result["namespaces"])  # events streamed from inside the subgraphs


def test_4_3_map_reduce_one_send_per_document(lesson, fake_retriever):
    llm = FakeChatModel(replies=["a note", "a note", "a note", "Yes: offline edits sync later."])

    state = lesson("4.3").run(llm, fake_retriever)

    assert len(state["docs"]) == 3 and len(state["notes"]) == 3  # one mapped call per document
    assert all(note.endswith("a note") for note in state["notes"])
    assert state["answer"] == "Yes: offline edits sync later."  # the reduce step ran last, once
    assert "a note" in llm.seen[-1][-1].text  # and it read the notes


def test_4_4_research_assistant(lesson, fake_retriever):
    from docshelp.research import Analyst, Perspectives

    first = Perspectives(analysts=[Analyst(name="Sam", role="student", focus="offline notes")] * 2)
    second = Perspectives(
        analysts=[
            Analyst(name="Sam", role="student", focus="offline notes"),
            Analyst(name="Ada", role="workspace admin", focus="sync conflicts across a team"),
        ]
    )
    # 2 interviews x (question, answer, section) + introduction, body, conclusion = 9 text calls
    llm = FakeChatModel(replies=["text"] * 9, structured={Perspectives: [first, second]})

    result = lesson("4.4").run(llm, fake_retriever)

    assert result["feedback_rounds"] == 2  # feedback once (re-planned), then approved
    assert sorted(s.splitlines()[0] for s in result["sections"]) == ["### For students", "### For workspace admins"]
    assert result["article"].startswith("# Working offline")
    assert "## Sources" in result["article"] and "changelog.md" in result["article"]
    assert not llm.replies  # every interview and writer ran exactly once


def test_4_5_store_remembers_across_threads(lesson):
    llm = FakeChatModel(
        replies=[
            "Nice to meet you, Ada.", "Ada is a workspace admin; the team uses Windows laptops.",  # thread 1
            "You use Windows.", "Ada is a workspace admin; the team uses Windows laptops.",  # thread 2
            tool_call("remember_preference", {"preference": "bullet points"}), "Noted!",  # create_agent
        ]
    )

    result = lesson("4.5").run(llm)

    thread_2_system = llm.seen[2][0]
    assert isinstance(thread_2_system, SystemMessage) and "Windows" in thread_2_system.text  # recalled
    assert len(llm.seen[2]) == 2  # ...although thread 2 itself had only the new question
    assert result["preferences"] == ["bullet points"]


def test_4_6_semantic_search_and_filters(lesson):
    llm = FakeChatModel(replies=["I see this happened before; let's fix it."])

    result = lesson("4.6").run(llm, DeterministicFakeEmbedding(size=32))

    assert len(result["hits"]) == 2
    assert all(hit["account_id"] == "acct-1001" for hit in result["hits"])  # namespaces isolate accounts
    assert result["open_hits"] and all(hit["status"] == "open" for hit in result["open_hits"])
    assert all(hit.get("summary") != "internal note" for hit in result["open_hits"])  # not indexed
    assert result["hits"][0]["summary"] in llm.seen[0][-1].text


def patch(doc_id, *ops, call_id="p1"):
    """The tool call trustcall asks for when updating an existing document: JSON patch operations."""
    return tool_call("PatchDoc", {"json_doc_id": doc_id, "planned_edits": "update", "patches": list(ops)}, call_id)


def test_4_7_profile_is_patched_not_regenerated(lesson):
    first = {
        "name": "Ada", "role": "workspace admin", "operating_systems": ["Windows", "iOS"], "devices": ["laptop", "iPhone"]
    }
    llm = FakeChatModel(
        replies=[
            tool_call("CustomerProfile", first),
            patch(
                "CustomerProfile",
                {"op": "replace", "path": "/operating_systems/0", "value": "macOS"},
                {"op": "add", "path": "/preferences/-", "value": "short answers"},
            ),
        ]
    )

    profiles = lesson("4.7").run(llm)

    assert profiles[0]["operating_systems"] == ["Windows", "iOS"]
    assert profiles[1]["operating_systems"] == ["macOS", "iOS"]  # patched in place
    assert profiles[1]["role"] == "workspace admin"  # untouched fields survive the update
    assert profiles[1]["preferences"] == ["short answers"]


def test_4_8_collection_updates_one_item_and_inserts_another(lesson):
    llm = FakeChatModel(
        replies=[
            tool_call("PastIssue", {"summary": "iPhone notes reach the laptop 20 minutes late", "platform": "Windows"}),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "PatchDoc", "id": "p1", "type": "tool_call", "args": {
                        "json_doc_id": "issue-1", "planned_edits": "solved",
                        "patches": [{"op": "replace", "path": "/status", "value": "solved"},
                                    {"op": "replace", "path": "/resolution", "value": "updated to 4.12"}]}},
                    {"name": "PastIssue", "id": "n1", "type": "tool_call",
                     "args": {"summary": "PDF export drops images", "platform": "Windows"}},
                ],
            ),
        ]
    )

    snapshots = lesson("4.8").run(llm)

    assert [i["key"] for i in snapshots[0]] == ["issue-1"]
    after = {i["key"]: i for i in snapshots[1]}
    assert after["issue-1"]["status"] == "solved" and after["issue-1"]["resolution"] == "updated to 4.12"
    assert after["issue-2"]["summary"] == "PDF export drops images"  # inserted as a new item


def test_4_9_capstone_memory_agent_remembers_across_threads(lesson, fake_retriever):
    from docshelp.memory import ns

    llm = FakeChatModel(
        replies=[
            # thread 1: the agent saves the profile, then the issue, then answers
            tool_call("UpdateMemory", {"update_type": "profile"}),
            tool_call("CustomerProfile", {"name": "Ada", "role": "workspace admin", "devices": ["iPhone", "laptop"]}),
            tool_call("UpdateMemory", {"update_type": "issues"}, "c2"),
            tool_call("PastIssue", {"summary": "iPhone notes slow to reach the Windows laptop", "platform": "Windows"}),
            "Thanks Ada, let's look into that delay.",
            # thread 2: nothing to save, it just answers from memory
            "Version 4.12 speeds up sync, which should fix your iPhone delay.",
        ]
    )

    result = lesson("4.9").run(llm, fake_retriever)

    store = result["store"]
    assert store.get(ns("profile", "acct-1001"), "profile").value["role"] == "workspace admin"
    assert [i.key for i in store.search(ns("issues", "acct-1001"))] == ["issue-1"]
    thread_2_prompt = llm.seen[-1]
    assert "iPhone notes slow" in thread_2_prompt[0].text  # recalled into a brand-new thread
    assert len(thread_2_prompt) == 2  # system prompt + the new question: no old messages
    assert result["answer"].startswith("Version 4.12")
