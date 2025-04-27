# Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

#### Autor: Santiago1712
#### Datum: 25.04.2025

## Projektbeschreibung

Dieses Projekt umfasst die Entwicklung eines autoregressiven Decoder-only-Sprachmodells unter Verwendung von PyTorch. Das Modell wurde auf dem Tiny Shakespeare-Datensatz trainiert und basiert auf der Transformer-Architektur, wobei die Schichten `nn.TransformerDecoderLayer` und `nn.TransformerDecoder` von PyTorch verwendet wurden. Das Hauptziel war die Implementierung eines funktionsfähigen Modells, das lernen konnte, Text mit einer ähnlichen Struktur wie der des Trainingsdatensatzes zu generieren und die Anforderungen des Kurses zu erfüllen.

## Modell

* **Architektur:** Decoder-only Transformer
* **Implementierung:** Unter Verwendung von Modulen aus `torch.nn` (`nn.TransformerDecoder`, `nn.TransformerDecoderLayer`).
* **Schlüsselparameter:**
    * `embed_size`: 8
    * `num_layers`: 1
    * `heads`: 1
    * `dropout`: 0.1
    * `forward_expansion`: 1
    * `max_len`: 128
* Hugging Face Model Hub: (https://huggingface.co/Santiago1712/Modell_TinySchakespeare)

## Tokenizer

Verwendet: `GPT2Tokenizer` aus der `transformers`-Bibliothek von Hugging Face.

## Training

* Datensatz: Tiny Shakespeare
* Anzahl der Epochen: 5
* Finaler Trainingsverlust: 4.96
* Finaler Validierungsverlust: 4.76
* Trainingsschleife: Vollständig in PyTorch implementiert (ohne Verwendung der `Trainer`-Klasse von Hugging Face).
* Logging: `wandb` wurde zur Verfolgung der Trainings- und Validierungsverluste verwendet.

## Evaluation

* Der Verlust wurde nach jeder Epoche auf einem Validierungsdatensatz berechnet, wobei während des Trainings eine sichtbare Abnahme zu beobachten war.
* Eine qualitative Bewertung wurde durch die Generierung von Beispieltext durchgeführt (die Ergebnisse können aufgrund der begrenzten Modellgröße variieren).

## Wie man den Code ausführt

1.  Stelle sicher, dass du die erforderlichen Bibliotheken installiert hast: `torch`, `torch.nn`, `transformers`, `requests`, `tqdm`, `wandb`. Du kannst sie mit `pip install torch transformers requests tqdm wandb` installieren.
2.  Lade das Notebook (`.ipynb`) oder das Python-Skript (`.py`) aus diesem Repository herunter.
3.  Führe das Notebook oder das Skript in einer Python-Umgebung aus. Der Code lädt den Datensatz herunter, lädt den Tokenizer, definiert und trainiert das Modell und versucht schließlich, das Modell zum Hugging Face Hub hochzuladen (erfordert die Authentifizierung mit deinem Token).

## Zusätzliche Hinweise

* Die Größe des Modells wurde klein gehalten (< 1 Million Parameter), um das Training auf der CPU zu ermöglichen.
* Die Qualität des generierten Textes kann aufgrund der reduzierten Modellgröße und der Anzahl der Trainingsepochen begrenzt sein. Das Hauptziel war die Demonstration der Implementierung eines autoregressiven Sprachmodells unter Verwendung der erforderlichen Werkzeuge.

