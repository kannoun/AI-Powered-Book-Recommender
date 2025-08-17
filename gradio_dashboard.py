import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import gradio as gr



books = pd.read_csv("books_with_emotions_categories.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


class BookSearchFAISS:
    def __init__(self, index, metadata, model):
        self.index = index
        self.metadata = metadata
        self.model = model
    
    def similarity_search(self, query: str, k: int = 10):
        # Encode query
        query_embedding = self.model.encode([query])
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_normalized.astype('float32'), k)
        
        # Format results
        docs = []
        for i, idx in enumerate(indices[0]):
            doc = self.metadata[idx].copy()
            doc['similarity_score'] = float(scores[0][i])
            docs.append(doc)
        
        return docs
    


# Load FAISS index from disk
index = faiss.read_index("book_faiss_index.index")

# Load metadata from disk
with open("book_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Loaded FAISS index with {index.ntotal} vectors and {len(metadata)} metadata entries")

# Initialize model (still needed for encoding queries)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize search
db_books = BookSearchFAISS(index, metadata, model)



def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Debug: Print the first recommendation to see the structure
    if recs:
        print("First recommendation structure:", recs[0].keys())
    
    # Try different possible key names for the content
    # You'll need to adjust this based on your actual metadata structure
    try:
        # Option 1: If the content is stored under 'page_content' key
        books_list = [int(rec['page_content'].strip('"').split()[0]) for rec in recs]
    except KeyError:
        try:
            # Option 2: If the content is stored under 'content' key
            books_list = [int(rec['content'].strip('"').split()[0]) for rec in recs]
        except KeyError:
            try:
                # Option 3: If the content is stored under 'text' key
                books_list = [int(rec['text'].strip('"').split()[0]) for rec in recs]
            except KeyError:
                # Option 4: If the ISBN is directly stored as a key
                try:
                    books_list = [int(rec['isbn13']) for rec in recs]
                except KeyError:
                    # Print available keys to help debug
                    print("Available keys in recommendation:", list(recs[0].keys()) if recs else "No recommendations")
                    raise KeyError("Could not find the correct key for book content/ISBN in metadata")

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Origin()) as dashboard:
    gr.Markdown("# Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about ...")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()