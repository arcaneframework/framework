# Analyse de performances (*Profiling*)                {#arcanedoc_profiling_toc}


Cette page décrit les mécanismes disponibles dans %Arcane pour
obtenir des informations sur les performances.


Ces mécanismes permettent de savoir quelles sont les méthodes qui
prennent le plus de temps dans le code.


\warning Actuellement, le *profiling* ne fonctionne que sur les plateformes **Linux**.


\warning Actuellement, le *profiling* **NE FONCTIONNE PAS** lorsque le *multi-threading*
(que ce soit avec le mécanisme des tâches ou d'échange de message) est actif.


Les différents type d'analyse de performances disponibles sont :
- \subpage arcanedoc_profiling
- \subpage arcanedoc_profiling_mpi
