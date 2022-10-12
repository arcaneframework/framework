# Service SimpleCsvOutput {#arcanedoc_services_modules_services}

[TOC]

Ce service permet de créer un tableau 2D de valeur avec des lignes et des colonnes nommées. Aujourd'hui, le format de fichier en sortie est le format CSV.
Ce service peut être utilisé comme service classique à définir dans l'AXL d'un module ou comme singleton pour avoir une instance unique pour tous les modules.

Ce service se veut simple d'utilisation mais permet aussi une utilisation plus avancée.

Exemple de fichier .csv :
```csv
Results_Example3;Iteration 1;Iteration 2;Iteration 3;
Nb de Fissions;36;0;85;
Nb de Collisions;29;84;21;
```
Sous Excel (ou un autre tableur), on obtient ce tableau :
| Results_Example3 | Iteration 1 | Iteration 2 | Iteration 3 |
|------------------|-------------|-------------|-------------|
| Nb de Fissions   | 36          | 0           | 85          |
| Nb de Collisions | 29          | 84          | 21          |

____
## Utilisation
### Singleton

Pour une utilisation en tant que singleton (même objet pour tous les modules) :

Placer ces lignes dans le .config de votre projet :
```xml
<singleton-services>
  <service name="SimpleCsvOutput" need="required" />
</singleton-services>
```
Et dans votre/vos module(s) :
```c
#include "arcane/ISimpleTableOutput.h"

using namespace Arcane;

ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
table->init("Example_Name");
//...
table->writeFile("./example/");
```
____
### Service

Pour une utilisation en tant que service (objet different pour chaque module) : 

Placer ces lignes dans le .axl de votre module :
```xml
<!-- <options> -->
  <service-instance name="simple-table-output" type="Arcane::ISimpleTableOutput">
    <description>Service implémentant ISimpleTableOutput</description>
  </service-instance>
<!-- </options> -->
```
Dans le .arc, vous pouvez configurer les options du services. Par exemple :
```xml
<!-- <mon-module> -->
  <simple-table-output name="SimpleCsvOutput">
    <!-- Le nom du répertoire à créer/utiliser. -->
    <tableDir>./example/</tableDir>
    <!-- Le nom du fichier à créer/écraser. -->
    <tableName>Results_Example</tableName>

    <!-- Au final, on aura un fichier ayant comme chemin : ./example/Results_Example.csv -->
  </simple-table-output>
<!-- </mon-module> -->
```
Et dans votre module :
```c
#include "arcane/ISimpleTableOutput.h"

using namespace Arcane;

options()->simpleCsvOutput()->init();
//...
options()->simpleCsvOutput()->writeFile();
```

Vous pouvez aussi utiliser le service des deux façons en même temps, selon vos besoins.

Pour un exemple plus concret, voir la [mini-app Quicksilver](https://github.com/arcaneframework/arcane-benchs/)

____
### Symboles de nom pour l'exécution parallèle (implémentation CSV)
Dans le nom du repértoire ou dans le nom du tableau, que ce soit en mode singleton ou en mode service, il est possible d'ajouter des "*symboles*" qui seront remplacés lors de l'exécution.

Les *symboles* disponibles sont :
- **@proc_id@** : Sera remplacé par le rank du processus.
- **@num_procs@** : Sera remplacé par le nombre total de processus.

Par exemple, si l'on a :
```xml
<tableDir>./N_@num_procs@/</tableDir>
<tableName>Results_P@proc_id@</tableName>
```
Et que l'on lance le programme avec 2 processus (ID = 0 et 1), on va obtenir deux csv ayant comme chemin :
- ./N_2/Results_P0.csv
- ./N_2/Results_P1.csv

(en séquentiel, on aura ./N_1/Results_P0.csv)

Cela permet, entre autres, de :
- créer un tableau par processus et de les nommer facilement,
- créer des fichiers .arc "générique" où le nombre de processus n'importe pas,
- avoir un nom différent pour chaque tableau, dans le cas où un *cat* est effectué (rappel : *tableName* donne le nom du fichier csv mais est aussi placé sur la première case du tableau).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_parallel_loadbalance
</span> -->
</div>