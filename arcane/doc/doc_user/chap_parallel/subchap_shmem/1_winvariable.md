# Variables en mémoire partagée {#arcanedoc_parallel_shmem_winvariable}

[TOC]

## Introduction {#arcanedoc_parallel_shmem_winvariable_intro}

Les variables %Arcane utilisent habituellement l'allocateur par défaut pour allouer de la mémoire. Sans GPU, on utilise
la mémoire locale de la machine et avec GPU, on utilise la mémoire unifiée.

Un nouvel allocateur mémoire (interne à %Arcane) est disponible et permet d'allouer de la mémoire en mémoire partagée.
Pour cela, en interne, on utilise la classe présentée précédemment : Arcane::MachineShMemWin. On aura donc accès à des
segments non-contigus.

Ce mode est compatible avec l'ensemble des types de variables %Arcane (sauf les variables partielles).

La principale difficulté pour utiliser ce mode mémoire partagée est de s'assurer que tous les appels qui réallouent la
mémoire soit collectifs.

Pour les variables redimensionnées par %Arcane, l'utilisateur n'a pas besoin de se préoccuper de ces appels collectifs.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_shmem_winarray
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
