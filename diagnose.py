"""
Diagnostic script to identify RAG system issues.
Run with: python diagnose.py
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if OpenAI API key is set and valid."""
    print("\n=== 1. Checking API Key ===")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False

    print(f"✓ API key found (starts with: {api_key[:8]}...)")

    # Test API key validity
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Simple test - list models (minimal cost)
        models = client.models.list()
        print("✓ API key is valid - can connect to OpenAI")
        return True
    except Exception as e:
        print(f"❌ API key error: {e}")
        return False


def check_vector_database():
    """Check if vector database loads correctly."""
    print("\n=== 2. Checking Vector Database ===")

    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        persist_dir = "combined_DB"

        if not os.path.exists(persist_dir):
            print(f"❌ Database directory '{persist_dir}' not found")
            return None

        print(f"✓ Database directory '{persist_dir}' exists")

        # Load embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("✓ Embeddings model loaded")

        # Load vector database
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

        # Check document count
        collection = vectordb._collection
        doc_count = collection.count()
        print(f"✓ Vector database loaded with {doc_count} documents")

        if doc_count == 0:
            print("⚠️ WARNING: Database is empty!")
            return None

        return vectordb

    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_retrieval(vectordb):
    """Test document retrieval."""
    print("\n=== 3. Testing Document Retrieval ===")

    if vectordb is None:
        print("⚠️ Skipping - vector database not available")
        return False

    try:
        # Test queries
        test_queries = [
            "recubrimiento mínimo",
            "instrumentación sísmica",
            "concrete cover",
            "seismic design"
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")

            # Try NSR-10
            try:
                nsr_docs = vectordb.similarity_search(
                    query,
                    k=2,
                    filter={"code": "NSR-10"}
                )
                print(f"  NSR-10: Found {len(nsr_docs)} documents")
                if nsr_docs:
                    print(f"    First doc page: {nsr_docs[0].metadata.get('page', 'N/A')}")
            except Exception as e:
                print(f"  NSR-10: Error - {e}")

            # Try ACI-318
            try:
                aci_docs = vectordb.similarity_search(
                    query,
                    k=2,
                    filter={"code": "ACI-318"}
                )
                print(f"  ACI-318: Found {len(aci_docs)} documents")
                if aci_docs:
                    print(f"    First doc page: {aci_docs[0].metadata.get('page', 'N/A')}")
            except Exception as e:
                print(f"  ACI-318: Error - {e}")

        return True

    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_embedding_generation():
    """Test if embeddings can be generated (costs a small amount)."""
    print("\n=== 4. Testing Embedding Generation ===")

    try:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Generate a test embedding
        test_text = "test embedding generation"
        result = embeddings.embed_query(test_text)

        print(f"✓ Embedding generated successfully (dimension: {len(result)})")
        return True

    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str:
            print("❌ Rate limit exceeded - too many requests")
        elif "quota" in error_str or "insufficient" in error_str:
            print("❌ API quota exhausted - no credits remaining")
        elif "authentication" in error_str or "api key" in error_str:
            print("❌ API key authentication failed")
        else:
            print(f"❌ Embedding error: {e}")
        return False


def check_llm():
    """Test if LLM can respond."""
    print("\n=== 5. Testing LLM Response ===")

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.0,
            max_tokens=50,
        )

        response = llm.invoke("Responde solo 'OK' si puedes leer esto.")
        print(f"✓ LLM responded: {response.content[:100]}")
        return True

    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str:
            print("❌ Rate limit exceeded")
        elif "quota" in error_str or "insufficient" in error_str:
            print("❌ API quota exhausted - no credits remaining")
        else:
            print(f"❌ LLM error: {e}")
        return False


def main():
    print("=" * 60)
    print("RAG System Diagnostic Tool")
    print("=" * 60)

    results = {}

    # Run checks
    results['api_key'] = check_api_key()

    if results['api_key']:
        results['embedding'] = check_embedding_generation()
        results['llm'] = check_llm()
        vectordb = check_vector_database()
        results['database'] = vectordb is not None
        results['retrieval'] = check_retrieval(vectordb)
    else:
        print("\n⚠️ Skipping remaining tests - API key issue")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not results.get('api_key'):
        print("→ Check your .env file contains OPENAI_API_KEY")
    elif not results.get('embedding') or not results.get('llm'):
        print("→ Your API key may have exhausted its credits")
        print("→ Check your OpenAI usage at: https://platform.openai.com/usage")
        print("→ Add billing/credits at: https://platform.openai.com/account/billing")
    elif not results.get('database'):
        print("→ Check that combined_DB folder exists and contains data")
    elif not results.get('retrieval'):
        print("→ Database exists but retrieval failed - may need to rebuild index")
    else:
        print("→ All systems appear operational")
        print("→ If still having issues, check Streamlit logs for specific errors")


if __name__ == "__main__":
    main()
