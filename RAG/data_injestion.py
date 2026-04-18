from langchain_core.documents import Document


"""
This is the structure of the Document that is used for the chunking purpose
"""
doc = Document(
    page_content="this is main text content used to create the RAG",
    metadata={
        "source" : "example.txt",
        "page":1,
        "author":"Siva Kavindra",
        "date_created":"2026-04-19"
    }
)




