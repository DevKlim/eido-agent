import pytesseract
from PIL import Image
import logging
from typing import Optional, Union, IO
import os

logger = logging.getLogger(__name__)

# --- Tesseract Configuration ---
# You might need to set the TESSERACT_CMD_PATH if tesseract is not in your system's PATH
# Example for Windows:
# TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux (if installed in a non-standard location):
# TESSERACT_CMD_PATH = '/usr/local/bin/tesseract'

# Attempt to set TESSERACT_CMD_PATH if an environment variable is set
# or if a common non-PATH location is known for the OS.
tesseract_cmd_env = os.getenv('TESSERACT_CMD')
if tesseract_cmd_env:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_env
    logger.info(f"Tesseract path set from TESSERACT_CMD env var: {tesseract_cmd_env}")
# else:
    # Add platform-specific checks for common installation paths if needed
    # if os.name == 'nt' and os.path.exists(TESSERACT_CMD_PATH_WINDOWS_DEFAULT):
    #     pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH_WINDOWS_DEFAULT
    #     logger.info(f"Tesseract path set to Windows default: {TESSERACT_CMD_PATH_WINDOWS_DEFAULT}")


def extract_text_from_image(image_input: Union[str, IO[bytes], Image.Image]) -> Optional[str]:
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        image_input: Path to the image file, a file-like object (e.g., BytesIO),
                     or a PIL Image object.

    Returns:
        The extracted text as a string, or None if extraction fails.
    """
    try:
        # Open the image if a path or file-like object is provided
        if isinstance(image_input, str) or hasattr(image_input, 'read'):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            logger.error(f"Invalid image input type: {type(image_input)}")
            return None

        # Convert image to grayscale for potentially better OCR results, though Tesseract handles color
        # img = img.convert('L')

        # Perform OCR
        # You can specify language e.g., lang='eng'
        # You can specify page segmentation mode (psm) or OCR engine mode (oem) for advanced control
        # Default psm is 3 (Fully automatic page segmentation, but no OSD)
        # Default oem is 3 (Default, based on what is available)
        custom_config = r'--oem 3 --psm 6' # Assume a single uniform block of text. Adjust as needed.
        text = pytesseract.image_to_string(img, config=custom_config)
        
        logger.info(f"OCR successful. Extracted {len(text)} characters.")
        # logger.debug(f"OCR Extracted Text (first 100 chars): {text[:100]}")
        return text.strip()

    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract is not installed or not in your PATH. "
            "Please install Tesseract OCR and ensure it's accessible. "
            "You might need to set pytesseract.tesseract_cmd to your Tesseract installation path."
        )
        return None
    except FileNotFoundError: # If image_input is a path and file not found
        logger.error(f"Image file not found: {image_input}")
        return None
    except Exception as e:
        logger.error(f"An error occurred during OCR: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing OCR Processor...")
    
    # Create a dummy image file for testing if you don't have one
    # For a real test, replace 'path/to/your/test_image.png' with an actual image path
    test_image_path = "test_ocr_image.png" # Needs to be created or replaced

    if os.path.exists(test_image_path):
        print(f"\n--- Testing with image file: {test_image_path} ---")
        extracted_text = extract_text_from_image(test_image_path)
        if extracted_text:
            print("Extracted Text:")
            print(extracted_text)
        else:
            print("Failed to extract text from the image.")
    else:
        print(f"Test image '{test_image_path}' not found. Skipping image file test.")
        print("To run this test effectively, create an image named 'test_ocr_image.png' with some text in the 'utils' directory,")
        print("or provide a path to an existing image.")

    # Example of how to test with an in-memory image (e.g., from an upload)
    try:
        from io import BytesIO
        # This requires a real image to be encoded into bytes_io
        # For a simple test, we'll skip this unless a real image path is available
        if os.path.exists(test_image_path):
            with open(test_image_path, "rb") as f:
                img_bytes = f.read()
            bytes_io = BytesIO(img_bytes)
            print(f"\n--- Testing with BytesIO object from: {test_image_path} ---")
            extracted_text_bytesio = extract_text_from_image(bytes_io)
            if extracted_text_bytesio:
                print("Extracted Text (from BytesIO):")
                print(extracted_text_bytesio)
            else:
                print("Failed to extract text from BytesIO.")
    except ImportError:
        print("BytesIO not available, skipping in-memory image test.")
    except Exception as e:
        print(f"Error during BytesIO test setup: {e}")