Variables d'environnement {#arcanedoc_env_variables}
==========================

Les variables d'environnement suivantes permettent de modifier le
comportement à l'exécution:

- **ARCANE_PARALLEL_SERVICE** :
  Nom du service utilisé pour gérer le parallélisme. Cette variable
  doit être positionnée si on souhaite un modèle d'exécution
  parallèle. Les valeurs possibles sont: `Mpi` ou `Sequential`. Le mode
  `Mpi` provoque l'appel à MPI_Init() et il faut donc que le programme
  soit lancé par le lanceur mpi de la plateforme (par exemple `mpiexec`,
  `mpirun`, `prun` ...

- **ARCANE_CHECK** :
  Si définie, ajoute des tests de vérification de la validité des
  opérations effectuées. Cela est utile si le code plante dans une
  méthode Arcane. Il est possible de relancer le code avec cette
  variable pour espérer détecter la cause de l'erreur. La version
  'check' de %Arcane inclut automatiquement ces tests ainsi que d'autres
  tests plus coûteux comme les débordements de tableau.

- **ARCANE_PAUSE_ON_ERROR** :
  Si définie, met le code en pause en cas d'erreur détectée par
  %Arcane, comme une exception fatale ou un débordement de tableau.

- **ARCANE_PROFILING** :
  Nom du service utiliser pour avoir des informations de
  profiling. Positionner cette option active le profiling. Cela permet
  en fin d'exécution d'avoir les temps passés dans chaque fonction.
  Les deux valeurs supportées sont 'Papi' et 'Prof'. Pour
  'Papi', la bibliothèque 'papi' doit avoir être installée et le noyau
  linux compatible. 'Prof' utilise les signaux du systeme.

- **ARCANE_PROFILING_PERIOD** :
  Nombre de cycles CPU entre deux échantillons de profiling. Cette
  variable n'est utilisée que si le profiling est actif avec
  'Papi'. Une valeur correcte est 500000, ce qui sur une machine à 3Ghz
  fait environ 6000 évènements par seconde.

- **ARCANE_PROFILING_ONLYFLOPS** :
  Si vrai, utilise uniquement le profiling pour déterminer le
  nombre d'opérations flottantes effectuées. Il faut bien noter qu'il
  s'agit du nombre d'instructions et pas du nombre d'opérations. Sur
  certaines machines comme les Itaniums, une instruction peut effectuer
  deux opérations (multiplication+addition). Cette option n'est
  disponible que si le profiling est actif avec l'option 'Papi'.

- **ARCANE_MESSAGE_PASSING_PROFILING** :
  Permet d'activer le profiling interne pour les opérations de
  message passing. Les valeurs possibles sont:

  - **JSON**: utilise le service de prise
    de trace au format du même nom. Celui-ci contient des informations
    sur le temps passé dans chaque fonctions MPI. Les informations sont
    regroupées par itération et par point d'entrée de la boucle en temps.

  - **OTF2**, utilise le service de prise de trace au format du
    même nom. Une fois ouvert avec un outil adéquat (par exemple Vampir)
    le détail des communications MPI peut être analysé. Les informations
    permettent notamment d'identifier les fonctions MPI mises en oeuvre
    dans chaque point d'entrée de la boucle en temps ainsi que celle
    invoquées par les opérations de synchronisation de variables %Arcane.

- **ARCANE_REDIRECT_SIGNALS** :
  Active (TRUE) ou désactive (FALSE) la redirection des signaux par
  Arcane. Cette redirection est active par défaut dans Arcane et permet
  d'afficher la pile d'appel notamment en cas de plantage. Neammoins,
  cela peut interférer avec d'autres bibliothèques et il est donc
  possible de désactiver la redirection des signaux.

- **ARCANE_PARALLEL_OUTPUT** :
  Active (TRUE) ou désactive (FALSE) les sorties listings de tous
  les sous-domaines en parallèle. Chaque sous-domaine écrira les
  informations de listing dans le fichier 'output#x' avec #x le numéro
  du sous-domaine.

- **ARCANE_TRACE_VARIABLE_CREATION** :
  Si définie, récupère la pile d'appel de chaque création de
  référence à une variable. Ces informations sont ensuite affichées en
  fin d'exécution afin de connaître les variables qui n'ont pas été
  désallouées. A noter que cela peut ralentir sensiblement une
  exécution.

- **STDENV_VERIF, STDENV_VERIF_ENTRYPOINT, STDENV_VERIF_PATH,
  STDENV_VERIF_SKIP_GHOSTS** :
  Voir la rubrique \ref arcanedoc_compare_bittobit

- **ARCANE_CHECK_MEMORY, ARCANE_CHECK_MEMORY_BLOCK_SIZE** :
Voir la rubrique \ref arcanedoc_check_memory

- **ARCANE_DATA_INIT_POLICY** :
  Permet de spécifier la politique d'initialisation des
  variables. L'utilisation de cette variable revient à appeler la
  méthode Arcane::setGlobalDataInitialisationPolicy(). Les valeurs possibles sont:
  - **NONE**: pas d'initialisation
  - **DEFAULT**: valeur par défaut du type, soit 0 pour les entiers et
    0.0 pour les réels
  - **NAN** : initialise les réels avec la valeur NotANumber ce qui
    permet si on active les exceptions flottantes d'arrêter le code si
    on utilise une variables non initialisée explicitement.
  - **LEGACY** : mode antérieur à la version 2.0 de %Arcane. Il ne doit
    plus être utilisé (voir Arcane::DIP_Legacy).

- **ARCANE_LISTENER_TIMEOUT**:

- **ARCANE_GDB_STACK**:

Les variables d'environnement suivantes peuvent être utilisées mais
sont succeptible de provoquer des instabilitées:

- ...

Les autres variables suivantes sont utilisées mais à usage interne à %Arcane:

- ARCANE_PARTICLE_NO_UNIQUE_ID_MAP
- ARCANE_OLD_EXCHANGE
- ARCANE_OLD_VARIABLE_SYNC
- ARCANE_CHANGE_OWNER_ON_INIT
- ARCANE_NB_EXCHANGE
- ARCANE_CHECK_EXCHANGE
- ARCANE_DEBUG_TIED_INTERFACE
- ARCANE_THREAD_IMPLEMENTATION
- ARCANE_LISTENER_TIMEOUT
- ARCANE_VERIF_PARALLEL
- ARCANE_FORCE_PREPARE_DUMP
- ARCANE_SERIALIZE_USE_DERIVED_TYPE
- ARCANE_TRACE_MPI
- ARCANE_NB_THREAD
- ARCANE_PARALLEL_CHECK_SYNC
- ARCANE_TRACE_FUNCTION
