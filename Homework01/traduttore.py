import json
from deep_translator import GoogleTranslator

def translate_notebook(input_path: str, output_path: str, target_language: str = 'it', chunk_size: int = 1000):
    # Carica il file notebook
    with open(input_path, 'r', encoding='utf-8') as file:
        notebook_content = json.load(file)
    
    translator = GoogleTranslator(source='auto', target=target_language)

    # Funzione per tradurre in blocchi
    def translate_in_chunks(text_lines, chunk_size):
        combined_text = ''.join(text_lines)
        # Dividi il testo in blocchi più grandi
        chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
        translated_chunks = []
        
        # Traduci ogni blocco
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk)
                # Se la traduzione fallisce e restituisce None, usa il chunk originale
                if translated_chunk is None:
                    translated_chunk = chunk
                translated_chunks.append(translated_chunk)
            except Exception as e:
                print(f"Errore durante la traduzione del blocco: {chunk[:100]}... Errore: {e}")
                translated_chunks.append(chunk)  # Mantieni il testo originale se la traduzione fallisce
                
        return ''.join(translated_chunks).splitlines(keepends=True)

    # Traduci tutte le celle di tipo 'markdown' o 'code'
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'markdown' or cell['cell_type'] == 'code':
            source_lines = cell['source']
            # Traduci il contenuto in blocchi
            translated_lines = translate_in_chunks(source_lines, chunk_size)
            cell['source'] = translated_lines

    # Salva il notebook tradotto
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(notebook_content, file, ensure_ascii=False, indent=4)

# Specifica i percorsi del file di input e output
input_file_path = 'HW1.ipynb'
output_file_path = 'HW1_translated.ipynb'

# Esegui la traduzione
translate_notebook(input_file_path, output_file_path)
