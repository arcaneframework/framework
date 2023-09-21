# Analyse de performances par instrumentation interne {#arcanedoc_debug_perf_profiling_mpi}


\warning Actuellement, le *profiling* ne fonctionne que sur les plateformes **Linux**.


\warning Actuellement, le *profiling* ne fonctionne que pour l'implémentation MPI
du *message passing*.


\warning Les fonctionnalités de *profiling* pour le *message passing* décrite ci-dessous
sont encore en développement.


%Arcane dispose d'un mécanisme interne de prise de traces pour l'analyse à
postériori des performances des operations dites de *message passing*.
Ce mécanisme est transparent pour l'utilisateur qui peut activer cette
fonctionnalité *via* la variable d'environnement
**ARCANE_MESSAGE_PASSING_PROFILING**.
Selon la valeur qu'on lui passe, l'un des services sera executé :
- \ref arcanedoc_debug_perf_profiling_mpi_json
- \ref arcanedoc_debug_perf_profiling_mpi_otf2


Chacun de ces services fournira des traces au format associé.


## JSON {#arcanedoc_debug_perf_profiling_mpi_json}


Lorsque l'on positionne la variable d'environnement **ARCANE_MESSAGE_PASSING_PROFILING=JSON**,
le service de profiling interne du *message passing* au format JSON est activé.
Celui-ci va espionner par itération et par point d'entrée les fonctions MPI, indiquant le nombre
d'appels à la fonction, la taille des messages échangés (en octet) et le temps passé dedans (en seconde).
Ces informations sont disponibles dans le sous-répertoire listing de sortie du cas.
Les fichiers sont nommés **message_passing_logs.i.json** où **i** correspond au numéro du
sous-domaine espionné.


Exemple de trace JSON :

```json
{
  "1": {
    "ArcaneTimeLoopBegin": {
      "MPI_Allgather": {
        "Count": 2,
        "MessageSize": 4,
        "TotalTime": 0.0000171661376953125
      }
    },
  },
  "2": {
    "TP_testLoop": {
      "MPI_Allreduce": {
        "Count": 3,
        "MessageSize": 2014,
        "TotalTime": 0.034449100494384769
      },
      "MPI_Recv": {
        "Count": 22,
        "MessageSize": 307625744,
        "TotalTime": 0.00004458427429199219
      },
      "Synchronize": {
        "Count": 82,
        "MessageSize": 82,
        "TotalTime": 0.0026214122772216799
      }
    }
  }
}

```


\note Un plugin pour browser permettant de lire et manipuler les traces fournies à ce format est en cours d'étude.


## OTF2 {#arcanedoc_debug_perf_profiling_mpi_otf2}


Lorsque l'on positionne la variable d'environnement **ARCANE_MESSAGE_PASSING_PROFILING=OTF2**,
le service de profiling interne du *message passing* au format OTF2 est activé.
Le format OTF2 est un format binaire (Open Trace Format 2) open source destiné à être lu par des
outils de profiling comme Vampir, Scalasca, Tau, Score-P, etc.

Chaque appel aux fonctions MPI est instrumenté et va permettre d'obtenir des informations détaillées
sur les communications entre sous-domaines, permettant notamment de détecter les problèmes de déséquilibre
de charge ou d'identifier de mauvais motifs d'échanges.

Ces informations sont disponibles dans le sous-répertoire listing de sortie du cas.
Elles sont stockées dans 2 fichiers et un répertoire. Le répertoire reprend le nom du cas et les deux fichiers
également modulo les extensions *.otf2* et *.def*.
\note Les fichiers et le répertoire sont écrasés à chaque nouveau lancement de la simulation.


Exemple de trace OTF2 visualisée par Vampir :

\image html ex_otf2_vampir.png


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_profiling_sampling
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling_loop
</span>
</div>
