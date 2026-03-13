# Les fenêtres mémoires en mémoire partagée en multi-processus {#arcanedoc_parallel_shmem}

[TOC]

Cette partie va décrire comment utiliser la mémoire partagée entre les processus d'un même noeud de calcul à l'aide de
fenêtres mémoires.

Une fenêtre mémoire est un espace mémoire alloué dans une partie de la mémoire accessible par tous les processus.
Cette fenêtre sera découpée en plusieurs segments, un par processus.

Deux moyens d'exploiter ces fenêtres sont disponibles :
- via des tableaux et des vues (comme on utiliserait des Arcane::UniqueArray),
- via des variables %Arcane.

<br>

Sommaire de ce sous-chapitre :

1. \subpage arcanedoc_parallel_shmem_winarray <br>
   Présente les classes permettant d'utiliser la mémoire partagée comme des tableaux %Arcane.

2. \subpage arcanedoc_parallel_shmem_winvariable <br>
   Présente comment créer et utiliser des variables %Arcane en mémoire partagée.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_loadbalance
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_shmem_winarray
</span>
</div>
