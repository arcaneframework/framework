# Exemple n°2 {#arcanedoc_services_modules_simplecsvoutput_example2}

[TOC]

L'exemple 2 est presque identique à l'exemple 1, mise à part la définition
du nom du tableau (et du fichier) et du nom du sous-répertoire.

En tant que **singleton**, notre service n'a pas accès au jeu de données,
donc il faut passer par l'un des modules pour transférer les informations.

L'exemple 2 montre comment faire ça simplement.

## Fichier .axl -- Partie <options>

Pour commencer, voyons les options du fichier axl :

`SimpleTableOutputExample2.axl`
\snippet SimpleTableOutputExample2.axl SimpleTableOutputExample2_options

Si vous jetez un oeil dans le `.axl` du service (ici : \ref axldoc_service_SimpleCsvOutput_arcane_std),
vous pourrez constater que c'est identique.
En effet, dans le cas du singleton, on utilise notre module principal pour récupérer
les informations dont a besoin notre service.

Autre chose à noter : la valeur par défaut `default=""`.
Mettre une valeur par défaut vide nous permet de déterminer si l'utilisateur
a spécifié une valeur ou non dans le `.arc`.
Plus tard, dans le module, on pourrait dire que s'il n'y a pas de valeurs sur les
deux options, alors c'est que l'utilisateur ne veut simplement pas de sortie csv.
(c'est la méthode utilisée dans QAMA).

## Fichier .arc -- Partie option du module

Voici le `.arc` correspondant :

`SimpleTableOutputExample2.arc`
\snippet SimpleTableOutputExample2.arc SimpleTableOutputExample2_arc

Comme expliqué au-dessus, c'est le module qui gère les deux options,
donc on se retrouve dans la partie "option du module".


## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample2Module.cc`
\snippet SimpleTableOutputExample2Module.cc SimpleTableOutputExample2_init

C'est le module qui gère les deux options du service, y compris les valeurs par
défaut. Si l'utilisateur ne spécifie pas de valeur pour l'option `tableName` dans le
`.arc`, on définit un nom par défaut (sachant que le service le fait aussi si
l'on appelle `table->init()` sans paramètres).

\warning
Un appel à init comme ceci : `table->init()` est différent d'un appel à init
comme cela : `table->init("")` ! L'un prendra une valeur par défaut, l'autre
aura un nom vide et le fichier de sortie n'aura simplement pas de noms (juste
l'extension).


## Point d'entrée loop

Ce point d'entrée est identique à celui de l'exemple 1.


## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample2Module.cc`
\snippet SimpleTableOutputExample2Module.cc SimpleTableOutputExample2_exit


La ligne
```cpp
if(options()->getTableName() != "" || options()->getTableDir() != "")
```
permet de savoir si l'utilisateur a entré au moins une des options.
Si c'est le cas, alors on vérifie la valeur par défaut et on écrit
le fichier.  
Si ce n'est pas le cas, alors on n'écrit pas de fichier.




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example1
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example3
</span>
</div>
