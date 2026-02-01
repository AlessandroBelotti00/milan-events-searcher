from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
import re
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


#Replace each base64 image with its corresponding summary
def replace_base64_images(md_text):
    pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

    return re.sub(pattern, "", md_text)


def convert_pdf_to_markdown(pdf_path: str) -> str:  
    # Configura pipeline PDF (OCR + estrazione immagini)
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

    # Converte PDF in Docling Document
    result = converter.convert(pdf_path)
    document = result.document
   
    markdown_text = document.export_to_markdown()

    

    return markdown_text


