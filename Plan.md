zapis ruchu - pozycja przed i po w FEN

**Potrzebne konwersje**

* Gra (PGN, chess.Game) -> lista [(ruch, komentarz)]
* ruch w FEN -> ruch-tensor



**Najprostszy preprocessing**
1. Usuń notację
2. Podziel na słowa i znaki "?" i "!"
3. Lemmatyzacja
4. Słowa -> glove embeddings