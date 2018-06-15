<!-- TODO Gedanken -->
# Gedanken
### 15.06.2018
Die Generator Implantierung ist nicht so allgeimein Gültig und man müsste
dann immer viel selber machen. Werde eine Hilfsklasse machen.

Hab die fit funktion im Netzwerk aufgeräumt

Generator classe sollte multithreadning können,
da ich viel Zeit verliere durch http request und
sonst ist es auch nicht schlecht. Probiere es mit einem
[Condition Objects](https://docs.python.org/3/library/threading.html#condition-objects)
und einer Queue.

Implementiert. Probleme: Die Queue liste kann zu gross werden
für den Speicher. Der Generator wird noch nicht
beendet obwohl er schon fertig ist und landed in einem
Deadlock

