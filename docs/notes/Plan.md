zapis ruchu - pozycja przed i po w FEN

**Potrzebne konwersje**

* Gra (PGN, chess.Game) -> lista [(ruch, komentarz)]
* ruch w FEN -> ruch-tensor

**Najprostszy preprocessing**
1. Usuń notację
2. Podziel na słowa i znaki "?" i "!"
3. Lemmatyzacja
4. Słowa -> glove embeddings


Sprawdzić model nie-szachowy do sentiment analysis, transfer learning

zobrist hashing

MCTS - z "losowymi" grami z użyciem sieci do oceny ruchu
Symulacja do znanej końcówki - baza danych z końcówkami

github chess gpt3

The chess transformer: Mastering Play using Generative Language Model

GPT3 przez API

BLOOM, Huggingface - mniejsza wersja GPT3

Stockfish - ocena agenta

Ocena reprezentacji - sieć neuronowa ucząca się funkcji stockfisha



Co zrobić dalej:
- przetestować model w porównaniu z losowym na większych danych
- Overfitting: zrobić prostszą sieć
- Wytrenować na większym zbiorze (sztucznych) danych


## Co zrobić po napisaniu 2 rozdziałów pracy

### Sentimate
- Przepisać reprezentację szachownicy na taką jak w papierze tj. 12 kanałów, po jednym na (bierka, kolor) + czyj ruch. Wyrzucić roszadę i bicie w przelocie
- Przemyśleć sentiment analysis komentarzy:
  - Huggingface - wielojęzykowy transformer
- Przepisać trening do Pytorch Lightning
- Sentimate z siecią podobną do DeepChess
- Ocena ruchu w innych wersjach:
  - Tak jak w oryginalnym Sentimate czyli softmax dla n=2
  - Ocena z zakresu [-1, 1] jak w AlphaZero
- Przepisać alpha-beta search do wersji takiej jak w pracy: zwracać najlepszy ruch i ocenę na każdej głębokości
