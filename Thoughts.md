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
beendet obwohl er schon fertig ist.
### 16.06.2018
Der Generator ist so implementiert, dass man 2 funktionen
immer selber implementieren muss. `get_mini_batches` stellt die mini_batches
her. Die return value von der funktion muss immer in einer liste sein, so dass
man auch mehr als nur Item zurückgeben kann. Also `[Item]` oder `[Item, Item]`,
aber nicht `Item`. Die zweite funktion ist die `__len__()`, damit ich weiss
wie oft ich die `get_mini_batches` aufrufen muss.
Das Multithreadning wird automatisch von der Base classe übernommen.

Queue ist für meine Zwecke nicht geeignet, da ich mehr als ein
Element jeweils aus der Liste haben muss. Eine Liste die ich slicen kann
eignet sich einiges besser für diese Anwendung.

Das Problem, dass die Threads nicht aufhören wenn die liste schon lang genug
ist löse ich am besten mit einem zweiten Condition Object. Wird es
auch schneller machen, da dann die liste mit den Items nicht mehr die
ganze Ziet von den Threads blockiert wird.

Ein Problem könnte noch sein, dass die Threads vom Generator schon
anfangen obwohl das Netzwerk noch nicht initziliesiert ist und es
desewgen ziemlich lange braucht um anzufangen.

Muss noch den Generator richtig beenden(Threads schliessen etc) und
auch wieder reiniliesieren. Bin mir nicht sicher ob ich die Threads
noch joinen muss oder nicht, da alle schon nicht mehr "leben".
 Glaub aber nicht das es schadet

Der Generator funktioniert nicht mit Multicoreprocesse, da
die Liste nicht SharedMemory ist. Fürs Threadning war das egal, da
dort alles SharedMemory ist.

### 17.06.2018

Mithilfe des Managers kann man ziemlich einfach SharedMemory
implementieren, dabei muss man aber sehr aufpassen, da der Processe
alles was es braucht pickelt und wenn man den Manager pickelt gibt
es eine Fehlermeldung, d.h. der Manager darf nicht in der Klasse sein
oder muss man irgenwie Unsichtbar machen vom Processe. Muss noch schauen
wie ich das am besten Löse.

Ich lösch einach die referenz des Manager aus dem dict der Klasse, so
dass es nicht gepickelt werden kann. Aber dadurch habe ich keinen Zugriff
auf den Manager. Was im Moment kein Problem ist. Sonst funktioniert der
Generator ziemlich gut.

#### Generator Vergleich
mini_batch_size = 128
##### Multicore 1 worker
mini_batch num: 0 | time: 5.13

mini_batch num: 1000 | time: 181.22
##### Multicore 2 worker
mini_batch num: 0 | time: 8.62

mini_batch num: 1000 | time: 177.53
##### Multicore 3 worker
mini_batch num: 0 | time: 15.46

mini_batch num: 1000 | time: 181.12
##### Old Generator
mini_batch num: 0 | time: 7.64

mini_batch num: 1000 | time: 231.46
<br><br><br>
Der Multicore Generator ist besser als der Alte. Aber mehr workers
macht ihn nicht viel schneller, was wahrscheinlich darin liegt das ich viele
Locks benutze, die sich gegenseitig blockieren.
