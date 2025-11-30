from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions, AcceleratorDevice, AcceleratorOptions
import re
from collections import OrderedDict


#Replace each base64 image with its corresponding summary
def replace_base64_images(md_text, summary_dict):
    pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

    def replacement(match):
        # Get next unused key from the summaries dict
        if summary_dict:
            key, value = summary_dict.popitem(last=False)  # pop the first item
            return f"\n\n{value}\n\n"
        else:
            return "\n\n[Image removed - no summary available]\n\n"

    return re.sub(pattern, replacement, md_text)


def convert_pdf_to_markdown(pdf_path: str) -> str:  
    # Configura pipeline PDF (OCR + estrazione immagini)
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
        generate_page_images=True,
        do_formula_enrichment=True,
        images_scale=2,
        table_structure_options={"do_cell_matching": True},
        accelerator_options=AcceleratorOptions(),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: 
                PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
        }
    )

    # Converte PDF in Docling Document
    result = converter.convert(pdf_path)
    document = result.document
   
    markdown_text = document.export_to_markdown(image_mode="embedded")

    return markdown_text


