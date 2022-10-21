# Exemple n°3 {#arcanedoc_services_modules_simplecsvoutput_example3}

[TOC]

Avec l'exemple 3 et les suivants, on n'utilise plus de singleton.
Donc on va voir un exemple simple d'utilisation du service normal.

À noter que cet exemple fait la même chose que les exemples
précedents.


## Fichier .axl -- Partie <options>

Pour commencer, voyons les options du fichier `.axl` :

`SimpleTableOutputExample3.axl`
\snippet SimpleTableOutputExample3.axl SimpleTableOutputExample3_options

Dans le `.axl`, on déclare juste l'utilisation d'un service implémentant
l'interface Arcane::ISimpleTableOutput.

## Fichier .arc -- Partie option du module

Voici le `.arc` correspondant :

`SimpleTableOutputExample3.arc`
\snippet SimpleTableOutputExample3.arc SimpleTableOutputExample3_arc

Ici, par rapport à l'exemple précédent, on remplit les options dans
la partie service.
On demande l'utilisation du service `SimpleCsvOutput` avec les deux options
qu'il demande.


## Point d'entrée initial

Voyons le point d'entrée `start-init` :

`SimpleTableOutputExample3Module.cc`
\snippet SimpleTableOutputExample3Module.cc SimpleTableOutputExample3_init

Par rapport à l'exemple précédent, on n'a pas besoin de récupérer un pointeur
vers un singleton ; ici c'est un service utilisé normalement.

Toujours par rapport à l'exemple précédent, c'est le service qui gère les
valeurs par défaut.

\note
Pour l'instant, il est impossible de demander la non-écriture des fichiers
de sortie directement dans le `.arc` partie service. Si vous ne mettez pas 
de valeurs dans les options `tableDir` et `tableName`, il y aura quand même
écriture. C'est à gérer par le module pour le moment.


## Point d'entrée loop

Voyons le point d'entrée `compute-loop` :

`SimpleTableOutputExample3Module.cc`
\snippet SimpleTableOutputExample3Module.cc SimpleTableOutputExample3_loop

Mis à part le remplacement du pointeur de singleton par l'utilisation
des options du module, il n'y a pas de différences avec les deux précédents
exemples.


## Point d'entrée exit

Enfin, voyons le point d'entrée `exit` :

`SimpleTableOutputExample3Module.cc`
\snippet SimpleTableOutputExample3Module.cc SimpleTableOutputExample3_exit

Ici, on se contente d'écrire le fichier de sortie (et de print le tableau).
Si l'on souhaite agir sur l'écriture ou non du fichier via le `.arc`,
c'est ici qu'on peut le faire en conditionnant l'appel à writeFile()
avec un `if()`.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example2
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvoutput_example4
</span>
</div>
