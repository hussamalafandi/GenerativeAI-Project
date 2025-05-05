# Teil 1: Decoder Language Model 

**Inhalt**:
- **Titel**: Decoder Language Model: Tiny Shakespeare
- **Untertitel**: Ein autoregressives Sprachmodell mit PyTorch
- **Autor**: Sakina Ahmadi
- **Datum**: 5. Mai 2025
- **Dozent**: M.Sc. Hussam Alafandi


## Projektübersicht
**Inhalt**:
- **Ziel**: Entwicklung eines kleinen autoregressiven Sprachmodells
- **Datensatz**: Tiny Shakespeare (~500k Zeichen)
- **Technologien**:
  - PyTorch für Modell und Training
  - wandb für Logging
  - Hugging Face für Modell-Upload
  - GitHub für Code-Abgabe
- **Anforderungen erfüllt**: Modellimplementierung, Training, Textgenerierung, Logging, Upload


## Modellarchitektur
**Inhalt**:
- **Transformer Decoder**:
  - `d_model=128`, `num_layers=2`, `nhead=4`
  - ~500k–700k Parameter
- **Code-Snippet** (aus `src/model.py`):
  ```python
  class DecoderLanguageModel(nn.Module):
      def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
          super().__init__()
          self.embedding = nn.Embedding(vocab_size, d_model)
          self.pos_encoder = PositionalEncoding(d_model, max_len)
          self.transformer_decoder = nn.TransformerDecoder(...)
  ```

## Training und Ergebnisse
**Inhalt**:
- **Training**:
  - 5 Epochen, benutzerdefinierter PyTorch-Loop
  - GPT2Tokenizer für Daten
  - wandb für Logging
  - wandb: `https://wandb.ai/ahmadi-sakina-hochschule-hannover/decoder-language-model/runs/mcr5bqb4?nw=nwuserahmadisakina` 
- **Textgenerierung**:
  - Beispiel: „To be or not to be not not not ...“
  - Repetitiv, aber akzeptabel


## Herausforderungen und Lösungen
**Inhalt**:
- **Dimensionsfehler**:
  - `RuntimeError: The size of tensor a (33) must match the size of tensor b (32)`
  - Lösung: Sequenzlängen in `src/generate.py` angepasst
- **Hugging Face-Upload**:
  - Syntaxfehler in `src/upload_hf.py` behoben
  - API-Token mit `.env` verwaltet
- **GitHub**:
  - Fehler: `src refspec ahmadiCSE-final`, fehlende `README.md`
  - Lösung: `README.md` erstellt, Branch gepusht


## Demo und Hugging Face
**Inhalt**:
- **Textgenerierung**:
  - Ausgabe: „To be or not to be not not not ...“
  - Code: `python src/generate.py`
- **Hugging Face**:
  - Modell: `https://huggingface.co/ahmadisakina/decoder-language-model`
  - Model-Card mit Metriken und Nutzung
- **Bild**: Screenshot der Hugging Face-Seite oder Textgenerierung


## Fazit 
**Inhalt**:
- **Zusammenfassung**:
  - Alle Anforderungen erfüllt: Modell, Training, Generierung, wandb, Hugging Face, GitHub
- **Learnings**:
  - Transformer-Architekturen, Debugging, Tool-Integration
- **Ausblick**:
  - größeres Modell

# Teil 2: Finetuning eines Sprachmodells mit HuggingFace Transformers

## Systeminformationen

- Pfad zur Umgebung: `C:\Users\ahmad\finetuning\venv\Lib\site-packages\`
- Python-Bibliothek: `transformers`
- Trainingsart: Sprachmodellierung (Causal Language Modeling)


## Trainingsverlauf

| Epoche | Durchschnittlicher Loss | Lernrate       | Grad-Norm | Bemerkung                          |
|--------|-------------------------|----------------|-----------|------------------------------------|
| 0.08   | 4.3157                  | 4.875e-05       | 9.04      | Trainingsbeginn                    |
| 0.15   | 4.0841                  | 4.748e-05       | 7.83      |                                    |
| 0.3    | 3.8335                  | 4.496e-05       | 6.00      | Starker Abfall der Verlustwerte    |
| 0.76   | 3.8441                  | 3.738e-05       | 4.56      | Leichter Anstieg nach Tiefpunkt    |
| 1.14   | 3.5266                  | 3.107e-05       | 4.39      | Stetige Verbesserung               |
| 1.97   | 3.4443                  | 1.718e-05       | 4.05      |                                    |
| 2.35   | 3.3055                  | 1.087e-05       | 4.24      | Tiefpunkt                          |
| 2.95   | 3.4156                  | 7.702e-07       | 4.40      | Letzter gemeldeter Stand           |


## Gesamtstatistiken

- **Trainingsdauer:** 5287 Sekunden (~1.47 Stunden)
- **Durchschnittlicher Loss:** 3.5766
- **Samples/Sekunde:** 1.498
- **Schritte/Sekunde:** 0.749
- **Trainings-Epochen:** 3.0
- **Gesamtschritte:** 3960


## Fazit

Das Finetuning verlief stabil und zeigte eine deutliche Verbesserung des Verlustwerts über die Zeit. Insgesamt zeigt das Modell ein gutes Lernverhalten, und weitere Optimierungsschritte (z. B. Hyperparameter-Tuning, GPU-Beschleunigung) könnten die Leistung weiter verbessern.


## Empfehlungen

- Migration der Datensätze zur `datasets`-Bibliothek
- Einsatz einer GPU für effizienteres Training
- Optimierung der Lernrate und Batchgröße für bessere Konvergenz
- Loss-Typ in der Konfigurationsdatei korrekt setzen



# Vielen Dank
