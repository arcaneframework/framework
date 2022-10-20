# Utilisation {#arcanedoc_services_modules_simplecsvoutput_usage}

[TOC]

## Singleton

Pour une utilisation en tant que singleton (même objet pour tous les modules) :

Placer ces lignes dans le .config de votre projet :

```xml
<singleton-services>
  <service name="SimpleCsvOutput" need="required" />
</singleton-services>
```

Et dans votre/vos module(s) :

```cpp
#include <arcane/ISimpleTableOutput.h>

using namespace Arcane;

ISimpleTableOutput* table = ServiceBuilder<ISimpleTableOutput>(subDomain()).getSingleton();
table->init("Example_Name", "example"); // Ne doit être fait que par un seul module.
// Utilisation du service...
table->writeFile(); // Ne doit être fait que par un seul module (sauf si vous savez ce que vous faites).
```

## Service

Pour une utilisation en tant que service (objet different pour chaque module) : 

Placer ces lignes dans le .axl de votre module :

```xml
<!-- <options> -->
  <service-instance name="simple-table-output" type="Arcane::ISimpleTableOutput">
    <description>Service implémentant ISimpleTableOutput</description>
  </service-instance>
<!-- </options> -->
```

Dans le .arc, vous pouvez configurer les options du service. Par exemple :

```xml
<!-- <mon-module> -->
  <simple-table-output name="SimpleCsvOutput">
    <!-- Le nom du répertoire à créer/utiliser. -->
    <tableDir>example_dir</tableDir>
    <!-- Le nom du fichier à créer/écraser. -->
    <tableName>Results_Example</tableName>

    <!-- Au final, on aura un fichier ayant comme chemin : 
    ./output/csv/example_dir/Results_Example.csv -->
  </simple-table-output>
<!-- </mon-module> -->
```

Et dans votre module :

```cpp
#include <arcane/ISimpleTableOutput.h>

using namespace Arcane;

options()->simpleCsvOutput()->init();
//...
options()->simpleCsvOutput()->writeFile();
```

Vous pouvez aussi utiliser le service des deux façons en même temps, selon vos besoins.

(Pour un exemple plus concret, voir les pages suivantes)


## Symboles de nom pour l'exécution parallèle (implémentation CSV)

Dans le nom du repértoire ou dans le nom du tableau, que ce soit en mode singleton ou en mode service, il est possible d'ajouter des *symboles* qui seront remplacés lors de l'exécution.

Les *symboles* disponibles sont :
- `@proc_id@` : Sera remplacé par le rank du processus.
- `@num_procs@` : Sera remplacé par le nombre total de processus.

Par exemple, si l'on a :

```xml
<tableDir>N_@num_procs@</tableDir>
<tableName>Results_P@proc_id@</tableName>
```

ou lors de l'initialisation du service :

```cpp
...
table->init("Results_P@proc_id@", "N_@num_procs@");
...
```

Et que l'on lance le programme avec 2 processus (ID = 0 et 1), on va obtenir deux csv ayant comme chemin :
- `./output/csv/N_2/Results_P0.csv`
- `./output/csv/N_2/Results_P1.csv`

(en séquentiel, on aura `./output/csv/N_1/Results_P0.csv`)

Cela permet, entre autres, de :
- créer un tableau par processus et de les nommer facilement,
- créer des fichiers .arc "générique" où le nombre de processus n'importe pas,
- avoir un nom différent pour chaque tableau, dans le cas où un *cat* est effectué (rappel : *tableName* donne le nom du fichier csv mais est aussi placé sur la première case du tableau).



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_examples
</span>
</div>
