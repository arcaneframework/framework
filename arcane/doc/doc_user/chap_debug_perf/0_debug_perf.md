# Debug, performance et validation {#arcanedoc_debug_perf}

Ce chapitre est dédié aux méthodes de débuggage et d'analyse
de performance.


Sommaire de ce chapitre :
1. \subpage arcanedoc_debug_perf_check_memory : détection des problèmes mémoire.

2. \subpage arcanedoc_debug_perf_compare_bittobit : comparaison bit à bit de deux exécutions.

3. \subpage arcanedoc_debug_perf_unit_tests : décrit comment réaliser des tests unitaires pour les modules et services.

____
<!-- TODO : Faire un sous-chapitre. -->

Cette page décrit les mécanismes disponibles dans %Arcane pour
obtenir des informations sur les performances.

Ces mécanismes permettent de savoir quelles sont les méthodes qui
prennent le plus de temps dans le code.

\warning Actuellement, le *profiling* ne fonctionne que sur les plateformes **Linux**.

\warning Actuellement, le *profiling* **NE FONCTIONNE PAS** lorsque le *multi-threading*
(que ce soit avec le mécanisme des tâches ou d'échange de message) est actif.

Les différents type d'analyse de performances disponibles sont :

1. \subpage arcanedoc_debug_perf_profiling

2. \subpage arcanedoc_debug_perf_profiling_mpi



____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_debug_perf_check_memory
</span>
</div>
