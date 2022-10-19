# Exemple n°1 {#arcanedoc_services_modules_simplecsvoutput_example1}

[TOC]

Commençons avec l'exemple 1.  
Cet exemple simple permet de sortir un tableau ressemblant à ça :


Results_Example1|Iteration 1|Iteration 2|Iteration 3
----------------|-----------|-----------|-----------
Nb de Fissions  |36         |0          |85
Nb de Collisions|29         |84         |21

Nous avons ici deux lignes et trois colonnes.
Les nombres présents sont générés aléatoirement.

## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample1Module.cc`
\snippet SimpleTableOutputExample1Module.cc SimpleTableOutputExample1_init

Dans cet exemple, on est en mode singleton. Le fichier `.axl` de cette exemple
ne contient donc pas d'options. 

En revanche, le fichier de configuration `.config` réference le singleton.

`csv.config` `<time-loop name="example1">`
\snippet csv.config SimpleTableOutputExample1_config

\note
Pour pouvoir séparer plusieurs cas d'exécutions dans le même code,
j'ai créé plusieurs `time-loop` dans le fichier `.config`.

Revenons dans le fichier `.cc`. On ne fait que récupérer le pointeur vers le
singleton et l'initialiser avec un nom de tableau (`Results_Example1`) et
un nom de sous-dossier (`example1`) (et on print le tableau vide).


## Point d'entrée loop

Voyons le point d'entrée `compute-loop` :

`SimpleTableOutputExample1Module.cc`
\snippet SimpleTableOutputExample1Module.cc SimpleTableOutputExample1_loop

Ici, on a un exemple simple de l'utilisation typique imaginé au départ pour ce service.

On crée une colonne nommé `Iteration X` (donc une nouvelle colonne à chaque itération),
puis on ajoute les valeurs que l'on souhaite sur des lignes.

Les lignes n'existent pas car elles n'ont pas été créées lors de l'init ? Le service
s'occupe de les créer avant d'ajouter la valeur.

Dans cet exemple, les lignes seront donc créées lors de la première itération et après,
le service se contentera de les remplir.

\note
On peut interchanger `row` et `column`, le fonctionnement est identique pour les lignes et
les colonnes (mais le résultat sera évidemment différent).

Exporter des valeurs d'un code, par exemple lors d'une session de débuggage est donc quelque chose
qui est très simple avec ce service.


## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample1Module.cc`
\snippet SimpleTableOutputExample1Module.cc SimpleTableOutputExample1_exit

Dans cette partie, on demande simplement l'écriture du fichier csv dont on a défini
l'emplacement et le nom lors de l'init. Dans l'hypothèse où ce n'aurait pas été fait,
il existe les méthodes \arcane{ISimpleTableOutput::setOutputDirectory()} et 
\arcane{ISimpleTableOutput::setTableName()} pour rectifier (et sinon, le service définit
des valeurs par défaut).




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_examples
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example2
</span>
</div>
