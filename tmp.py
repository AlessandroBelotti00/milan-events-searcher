# retrieval.py

from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.pipeline.standard_pdf_pipeline import StandardPdfPipelineOptions
# from docling.pipeline.accelerator import AcceleratorOptions
# from docling.pipeline.format_options import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
import re
import os
from dotenv import load_dotenv
from src.retrieval.utils import convert_pdf_to_markdown
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.retrieval.chunk_embed import ChunkList, chunking_llm, EmbedData

load_dotenv(override=True)

class Chunk(BaseModel):
    """A food recipe chunk with structured fields."""
    title: str = Field(..., description="The title of the recipe")
    ingredients: str = Field(..., description="The list of ingredients needed for the recipe and their quantities")
    preparation: str = Field(..., description="The preparation steps for the recipe")
    cooking_mode: str = Field(..., description="The list of cooking modes to be used (e.g., baking, frying, boiling, oven)")

class ChunkList(BaseModel):
    chunks: list[Chunk]

def chunk_to_embedding_text(chunk: ChunkList) -> str:
    return f"""
        Recipe title: {chunk.title}

        Ingredients:
        {chunk.ingredients}

        Preparation:
        {chunk.preparation}

        Cooking mode:
        {chunk.cooking_mode}
    """.strip()


def chunking_llm(paragraph):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(model=os.getenv("OPENAI_DEPLOYMENT_NAME"))

    model_with_structure = model.with_structured_output(ChunkList)

    prompt = f"""
        You are extracting structured recipe data from a document.

        Split the text into multiple chunks.
        Each chunk MUST correspond to exactly ONE recipe.

        Rules:
        - Do not merge recipes
        - Do not invent information
        - If a field is missing, return an empty string
        - Return ONLY structured data

        Text:
        {paragraph}
        """

    response = model_with_structure.invoke(prompt)
    return response


def clean_and_split_markdown(text):
    """
    Pipeline:
    1) Remove all <!-- image --> tags
    2) Keep only level-2 headings with ALL CAPS titles (## TITLE)
    3) Split text into chunks at each heading
    """
    
    # 1️⃣ Remove all image tags
    cleaned_text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
    
    # 2️⃣ Find all level-2 headings in all caps
    # Regex: '##' followed by optional spaces, then uppercase letters, numbers, spaces, commas, hyphens, accented letters
    heading_pattern = re.compile( r"^##\s+(.+)$", flags=re.MULTILINE)
    all_headings = [(m.start(), m.group(1)) for m in heading_pattern.finditer(cleaned_text)]

    # 3️⃣ Filter headings using zero-shot classifier
    recipe_headings = [(idx, h) for idx, h in all_headings if is_recipe_title_zero_shot(h)]

    # 4️⃣ Split text into chunks at each recipe title
    chunks = []
    for i, (start_idx, heading) in enumerate(recipe_headings):
        # Determine end of chunk
        end_idx = recipe_headings[i+1][0] if i+1 < len(recipe_headings) else len(cleaned_text)
        chunk = cleaned_text[start_idx:end_idx].strip()
        chunks.append(chunk)
    return chunks



def ocr_pdf_to_markdown(pdf_path: str) -> str:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,                  # enable OCR
        force_ocr=False,              # only OCR if needed
        images_scale=1,               # faster OCR
        do_table_structure=False,     # disable tables
        do_formula_enrichment=False,  # disable formulas
        generate_picture_images=False,
        generate_page_images=False,
        accelerator_options=AcceleratorOptions(
            num_threads=4              # adjust for your CPU
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,    
                pipeline_options=pipeline_options,
            )
        }
    )

    result = converter.convert(pdf_path)

    # Export clean markdown (best for chunking)
    return result.document.export_to_markdown()

#TODO
#TESTARE convert_pdf_to_markdown() CON IL PDF
# print(ocr_pdf_to_markdown("C:/Users/al.belotti/personal_projects/milan-events-searcher/docs/ricettario1.pdf"))
# output_path = "ricettario3_ocr.txt"
# with open(output_path, "w", encoding="utf-8") as f:
#     f.write(ocr_pdf_to_markdown("C:/Users/al.belotti/personal_projects/milan-events-searcher/docs/ricettario3.pdf"))

# print(f"Markdown saved to: {output_path}")

raw_text = open("C:/Users/al.belotti/personal_projects/milan-events-searcher/ricettario2_ocr.txt", "r", encoding="utf-8").read()



# for i, chunk in enumerate(recipe_chunks):
#     print(f"--- Chunk {i+1} ---")
#     print(chunk[:50])  # print first 500 characters
#     print("\n")

chunk_list = chunking_llm(raw_text) 
embeddata = EmbedData(batch_size=8)
embeddata.embed(chunk_list)
for el in [chunk_to_embedding_text(chunk) for chunk in chunk_list.chunks]:
    print(el)
    print("--------------------------")

# title_emb = embed(chunk.title)
# ingredients_emb = embed(chunk.ingredients)
# prep_emb = embed(chunk.preparation)
# mode_emb = embed(chunk.cooking_mode)

# # Weighted average
# final_emb = normalize(0.5*ingredients_emb + 0.3*prep_emb + 0.2*title_emb)

