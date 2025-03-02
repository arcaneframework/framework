# Variables d'environnement {#arcanedoc_execution_env_variables}

Les variables d'environnements suivantes permettent de modifier le
comportement à l'exécution :

<table>
<tr>
  <th>Variable</th>
  <th>Description</th>
</tr>

<tr>
  <td>
    ARCANE_PARALLEL_SERVICE (obsolète)
  </td>
  <td>
    Nom du service utilisé pour gérer le parallélisme. Cette variable
    doit être positionnée si on souhaite un modèle d'exécution
    parallèle spécifique. Les valeurs possibles sont : `Mpi` ou `Sequential`. Le mode
    `Mpi` provoque l'appel à MPI_Init() et il faut donc que le programme
    soit lancé par le lanceur mpi de la plateforme (par exemple `mpiexec`,
    `mpirun`, `prun`, ...).
    Cette variable ne doit plus être utilisée car %Arcane détecte
    automatiquement le lancement avec MPI. Si on souhaite surcharger le
    service de parallélisme, il faut le spécifier dans les arguments de
    la ligne de commande (voir \ref arcanedoc_execution_launcher)
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHECK
  </td>
  <td>
    Si définie, ajoute des tests de vérification de la validité des
    opérations effectuées. Cela est utile si le code plante dans une
    méthode Arcane. Il est possible de relancer le code avec cette
    variable pour espérer détecter la cause de l'erreur. La version
    'check' de %Arcane inclut automatiquement ces tests ainsi que d'autres
    tests plus coûteux comme les débordements de tableau.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PAUSE_ON_ERROR
  </td>
  <td>
    Si définie, met le code en pause en cas d'erreur détectée par
    %Arcane, comme une exception fatale ou un débordement de tableau.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING
  </td>
  <td>
    Nom du service utiliser pour avoir des informations de
    profiling. Positionner cette option active le profiling. Cela permet
    en fin d'exécution d'avoir les temps passés dans chaque fonction.
    Les deux valeurs supportées sont 'Papi' et 'Prof'. Pour
    'Papi', la bibliothèque 'papi' doit avoir être installée et le noyau
    linux compatible. 'Prof' utilise les signaux du système.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING_PERIOD
  </td>
  <td>
    Nombre de cycles CPU entre deux échantillons de profiling. Cette
    variable n'est utilisée que si le profiling est actif avec
    'Papi'. Une valeur correcte est 500000, ce qui sur une machine à 3Ghz
    fait environ 6000 évènements par seconde.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PROFILING_ONLYFLOPS
  </td>
  <td>
    Si vrai, utilise uniquement le profiling pour déterminer le
    nombre d'opérations flottantes effectuées. Il faut bien noter qu'il
    s'agit du nombre d'instructions et pas du nombre d'opérations. Sur
    certaines machines comme les Itaniums, une instruction peut effectuer
    deux opérations (multiplication+addition). Cette option n'est
    disponible que si le profiling est actif avec l'option 'Papi'.
  </td>
</tr>
<tr>
  <td>
    ARCANE_LOOP_PROFILING_LEVEL
  </td>
  <td>
    (version 3.8+). Entier indiquant le niveau de profiling des
    boucles. Positionner avec une valeur supérieure ou égale 1 active
    les traces des boucles basées sur les macros fournies par %Arcane
    telles que les ENUMERATE_() (ENUMERATE_CELL(), ENUMERATE_NODE(), ...),
    et les RUNCOMMAND (RUNCOMMAND_LOOP(), RUNCOMMAND_ENUMERATE()). En fin
    de calcul le temps passé dans chaque boucle sera écrit dans le
    listing et dans des fichiers.
  </td>
</tr>

<tr>
  <td>
    ARCANE_MESSAGE_PASSING_PROFILING
  </td>
  <td>
    Permet d'activer le profiling interne pour les opérations de
    message passing. Les valeurs possibles sont :
    - **JSON**: utilise le service de prise
      de trace au format du même nom. Celui-ci contient des informations
      sur le temps passé dans chaque fonction MPI. Les informations sont
      regroupées par itération et par point d'entrée de la boucle en temps.
    - **OTF2**, utilise le service de prise de trace au format du
      même nom. Une fois ouvert avec un outil adéquat (par exemple Vampir)
      le détail des communications MPI peut être analysé. Les informations
      permettent notamment d'identifier les fonctions MPI mises en oeuvre
      dans chaque point d'entrée de la boucle en temps ainsi que celle
      invoquées par les opérations de synchronisation de variables %Arcane.
  </td>
</tr>
<tr>
  <td>
    ARCANE_REDIRECT_SIGNALS
  </td>
  <td>
    Active (TRUE) ou désactive (FALSE) la redirection des signaux par
    Arcane. Cette redirection est active par défaut dans Arcane et permet
    d'afficher la pile d'appel notamment en cas de plantage. Néanmoins,
    cela peut interférer avec d'autres bibliothèques et il est donc
    possible de désactiver la redirection des signaux.
  </td>
</tr>
<tr>
  <td>
    ARCANE_PARALLEL_OUTPUT
  </td>
  <td>
    Active (TRUE) ou désactive (FALSE) les sorties listings de tous
    les sous-domaines en parallèle. Chaque sous-domaine écrira les
    informations de listing dans le fichier 'output#x' avec #x le numéro
    du sous-domaine.
  </td>
</tr>
<tr>
  <td>
    ARCANE_TRACE_VARIABLE_CREATION
  </td>
  <td>
    Si définie, récupère la pile d'appel de chaque création de
    référence à une variable. Ces informations sont ensuite affichées en
    fin d'exécution afin de connaître les variables qui n'ont pas été
    désallouées. A noter que cela peut ralentir sensiblement une
    exécution.
  </td>
</tr>
<tr>
  <td>
    STDENV_VERIF, STDENV_VERIF_ENTRYPOINT, STDENV_VERIF_PATH, STDENV_VERIF_SKIP_GHOSTS
  </td>
  <td>
    Voir la rubrique \ref arcanedoc_debug_perf_compare_bittobit
  </td>
</tr>
<tr>
  <td>
    ARCANE_CHECK_MEMORY, ARCANE_CHECK_MEMORY_BLOCK_SIZE
  </td>
  <td>
    Voir la rubrique \ref arcanedoc_debug_perf_check_memory.
    Cela ne fonctionne qu'avec certaines versions de Linux et la
    glibc. Ce mécanisme utilise des fonctionnalités obsolètes de la
    glibc et n'est donc pas disponible avec des versions récentes
    (>2021) de la glibc. Notamment, ubuntu 22.04 ne supporte pas ce
    mécanisme.
  </td>
</tr>
<tr>
  <td>
    ARCANE_DATA_INIT_POLICY
  </td>
  <td>
    Permet de spécifier la politique d'initialisation des
    variables. L'utilisation de cette variable revient à appeler la
    méthode Arcane::setGlobalDataInitialisationPolicy(). Les valeurs possibles sont :
    - **NONE**: pas d'initialisation
    - **DEFAULT** : valeur par défaut du type, soit 0 pour les entiers et
      0.0 pour les réels
    - **NAN** : initialise les réels avec la valeur NotANumber ce qui
      permet si on active les exceptions flottantes d'arrêter le code si
      on utilise une variable non initialisée explicitement.
    - **LEGACY** : mode antérieur à la version 2.0 de %Arcane. Il ne doit
      plus être utilisé (voir Arcane::DIP_Legacy).
  </td>
</tr>
<tr>
  <td>
    ARCANE_LISTENER_TIMEOUT
  </td>
  <td>
    todo
  </td>
</tr>
<tr>
  <td>
    ARCANE_GDB_STACK
  </td>
  <td>
    todo
  </td>
</tr>
<tr>
  <td>
    ARCANE_REPLACE_SYMBOLS_IN_DATASET
  </td>
  <td>
    Variable d'environnement permettant d'activer le remplacement des symboles dans
    le jeu de données à partir des arguments de la ligne de commande.
    Voir la page \ref arcanedoc_execution_commandlineargs
  </td>
</tr>


</table>


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_direct_execution
</span>
<span class="next_section_button">
\ref arcanedoc_execution_traces
</span>
</div>

