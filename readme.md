## Coordinació de donacions inter-hospitalàries

Aquest projecte aborda el problema de la coordinació de donacions d’òrgans entre hospitals a nivell internacional.

Es treballa amb 40 hospitals que s’han d’assignar a 4 xarxes exclusives de donació, de manera que cada hospital pertanyi només a una xarxa. L’objectiu és trobar una assignació òptima que minimitzi la distància entre hospitals dins de la mateixa xarxa i, alhora, tingui en compte la similitud de les seves poblacions.

## Coordinació de donacions inter-hospitalàries

Aquest projecte aborda el problema de la coordinació de donacions d’òrgans entre hospitals a nivell internacional.

Es treballa amb 40 hospitals que s’han d’assignar a 4 xarxes exclusives de donació, de manera que cada hospital pertanyi només a una xarxa. L’objectiu és trobar una assignació òptima que minimitzi la distància entre hospitals dins de la mateixa xarxa i, alhora, tingui en compte la similitud de les seves poblacions.

## Resolució

El problema es resol mitjançant algorismes de cerca local. Cada estat representa una possible assignació dels 40 hospitals a les 4 xarxes.
S’han implementat i comparat els següents algorismes:
- Cerca local beam
- Hill climbing
- Simulated annealing
- Cerca aleatòria (com a referència)

També s’utilitzen tècniques d’inferència probabilística per treballar amb la xarxa bayesiana i gestionar dades incompletes.



## Estructura del projecte
- bn.py: Implementació bàsica de xarxes bayesianes, factors i distribucions de probabilitat.
- inferencia.py: Conté els algoritmes d'inferencia, Variable Elimination, Rejection Sampling, i Weighted Sampling.
- p1.py: Conté exemples d'úss dels algoritmes amb les dades proporcionades, així com funcions auxiliars per les cerques locals.
- my_bns.py: Conté les definicions específiques de les xarxes bayesianes usades en els experiments.
- data.csv: Fitxer amb les dades utilitzades en els experiments.


##Criteris d'aturada:
Les cerques locals (hill_climbing, simulated_annealing, cerca_local_beam) tenen criteris d'aturada basats en tolerancia de millora (tolerancia). Si no es complís aquest criteri, es podria arribar al màxim d'iteracions sense trobar millores significatives.

##Rendiment:
El rendiment en temps d'execució dels algoritmes pot variar segons la configuració dels paràmetres (per exemple, el nombre d'iteracions i la mida del beam).
