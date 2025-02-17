def save_text_to_file(filename, text):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        return f"Text successfully saved to {filename}"
    except Exception as e:
        return f"Error: {str(e)}"