<!-- TODO Gedanken -->
# Gedanken
## 15.06.2018
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
## 16.06.2018
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

## 17.06.2018

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
<br><br>
Der Multicore Generator ist besser als der Alte. Aber mehr workers
macht ihn nicht viel schneller, was wahrscheinlich darin liegt das ich viele
Locks benutze, die sich gegenseitig blockieren.

#### Parallelizing
Ich hab mir verschiedene Arten von Parralelitierung von einem
Neuronalen Netzwek angeschaut. Darunter waren Pipline, large scale distributed und
anderes. Am meisten hat mir der [Hogwild Alogrithmus](https://arxiv.org/abs/1106.5730)
gefallen, da einfach zu implementieren ist und auch gut zu funktioniren scheint

## 18.06.2018
Nach Hogwild muss ich nur in den Layers SharedMemory bereit stellen.
Der Optimizer speichert auch daten, aber ich glaub das ist ok wenn
jeder Processe seinen eigenen Optimizer hat.

Nach langem probieren ist mir aufgefallen, dass der Hogwild Alogrithmus
nicht implementierbar ist in Python, da das SharedMemory auf picklen und
hin und herschicken mit Pipes basiert, aber für Hogwild  bräuchte ich
eine atomi funktion, die es so nicht in Python gibt. Ich weiss gar nicht
ob es sich üperhaupt lohnen kann, das Netzwerk zu parallelisieren, da Python
einfach nicht dafür gemacht worden ist. Ich gebe das Parallelisieren auf und
werden es vielleicht später nochmal probieren

Hab den Convolution Layer implementiert. Hab viel Code von Websiten.

Implementierung von MaxPoolLayer mit einer BasePool Layer Klasse für
einfache implementierung einer weitern Pool Klasse

Flatten Layer hinzugefügt um von einem Convlayer zu einem FullyConnected
zu kommen. Ausserdem hab ich den ndarray vom "alten" Netzwerk umgedreht, d.h
von `[daten, mini_batch_size]` zu `[mini_batch_size, daten]`, da es so
allgemein gültiger ist und mit dem convlayer übereinstimmt, da bei dem `daten`
mehrere Werte sind und durch die Umdrehung bleibt `mini_batch_size`  immer
an der ersten Stelle.(Umstellung noch nicht am Generator und am Batchnormlayer)

## 19.06.2018

Batchnorm mit Convolution kompatibel

Der Convoultion Layer funktiniert gut, aber es braucht ewig zum lernen
mit dem Netzwerk:
```
train_data, train_labels, test_data, test_labels = load_conv()
net = Network()

net.input((1, 28, 28))

net.add(ConvolutionLayer(n_filter=32, width_filter=3, height_filter=3, stride=1, zero_padding=0))
net.add(BatchNorm())
net.add(ReLU())
net.add(ConvolutionLayer(n_filter=64, width_filter=3, height_filter=3, stride=1, zero_padding=0))
net.add(BatchNorm())
net.add(ReLU())
net.add(MaxPoolLayer(width_filter=2, height_filter=2, stride=1))
net.add(Dropout(0.75))
net.add(Flatten())
net.add(FullyConnectedLayer(128))
net.add(ReLU())
net.add(Dropout(0.5))
net.add(FullyConnectedLayer(10))
net.add(SoftMax())

optimizer = Adam(learning_rate=0.01)
net.regression(optimizer=optimizer, cost="cross_entropy")

net.fit(train_data, train_labels, validation_set=(test_data, test_labels), epochs=12, mini_batch_size=256,
        plot=True, snapshot_step=2)
net.evaluate(test_data, test_labels)
```
bräuchte es 16 Stunden.

Deswegen hab ich die numpy libary mit cupy ersetzt. Sie funktiniert
genau gleich(bis auf kleine Ausnahmen), aber sie arbeitet mit der GPU
mithilfe von CUDA, d.h es funktiniert nur wenn CUDA installiert ist und
man eine NVIDIA GPU hat. Ueber die GPU lernt das Netzwerk 10 Mal schneller, d.h
es  braucht "nur" noch 1.6 Stunden. Im Moment geht es aber nur mit CUDA
und nicht mehr über die CPU.

Hab alles kompatible mit cupy gemacht.(Ausser Generator)
