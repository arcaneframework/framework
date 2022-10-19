# Exemple n°2 {#arcanedoc_services_modules_simplecsvoutput_example2}

[TOC]

L'exemple 2 est presque identique à l'exemple 1, mise à part la définition
du nom du tableau (et du fichier) et du nom du sous-répertoire.

En tant que **singleton**, notre service n'a pas accès au jeu de données,
donc il faut passer par l'un des modules pour transférer les informations.

L'exemple 2 montre comment faire ça simplement (et peut être utilisé comme
"template" pour aller plus vite).

## Fichier .axl -- Partie <options>

Pour commencer, voyons les options du fichier axl :

`SimpleTableOutputExample2.axl`
\snippet SimpleTableOutputExample2.axl SimpleTableOutputExample2_options

Si vous jetez un oeil dans le `.axl` du service (ici : \ref axldoc_service_SimpleCsvOutput_arcane_std),
vous pourrez constater que c'est identique.
#ICI

## Fichier .arc -- Partie option du module

`SimpleTableOutputExample2.arc`
\snippet SimpleTableOutputExample2.arc SimpleTableOutputExample2_arc


## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample2Module.cc`
\snippet SimpleTableOutputExample2Module.cc SimpleTableOutputExample2_init




## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample2Module.cc`
\snippet SimpleTableOutputExample2Module.cc SimpleTableOutputExample2_exit





____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_examples
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example2
</span>
</div>
