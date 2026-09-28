from docshelp.ingest import load_docs, split_docs
from docshelp.tools import quote_price


def test_docs_load_and_split_with_source_metadata():
    docs = load_docs()
    assert len(docs) == 8
    chunks = split_docs(docs)
    assert len(chunks) > len(docs)
    assert all("source" in c.metadata and "title" in c.metadata for c in chunks)


def test_retriever_returns_documents(fake_retriever):
    results = fake_retriever.invoke("How long do deleted notes stay in trash?")
    assert len(results) == 3


def test_quote_price_tool():
    assert quote_price.invoke({"plan": "team", "members": 12, "billing": "annual"}).startswith(
        "Team plan, 12 members, annual billing: $921.60"
    )
    assert "$40.00 per month" in quote_price.invoke({"plan": "team", "members": 5})
    assert "custom" in quote_price.invoke({"plan": "enterprise", "members": 500})
