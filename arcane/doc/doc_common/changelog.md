# Nouvelles fonctionnalités {#arcanedoc_news_changelog}

[TOC]

Cette page contient les nouveautés de chaque version de %Arcane v3.X.X.

Les nouveautés successives apportées par les versions de %Arcane
antérieures à la version 3 sont listées ici : \ref arcanedoc_news_changelog20

___

## Arcane Version 3.11 (En cours...) {#arcanedoc_version3110}

### Nouveautés/Améliorations

- Réorganisation interne des synchronisations pour utiliser un seul
  buffer mémoire pour toutes les synchronisations d'un même maillage
  (\pr{861}, \pr{862}, \pr{863}, \pr{866}, \pr{867}, \pr{871},
  \pr{872}, \pr{873}, \pr{874}, \pr{878}, \pr{880})
- Ajoute interface \arcane{IVariableSynchronizerMng} pour gérer toutes
  les instances de \arcane{IVariableSynchronizer} pour un maillage
  donné (\pr{869}, \pr{879}).
- Ajoute mode automatique pour vérifier si une synchronisation a eu un
  effet. Ce mode est activé si la variable d'environnement
  `ARCANE_AUTO_COMPARE_SYNCHRONIZE` est positionnée. Lorsque ce
  mode est actif, des statistiques en fin de calcul permettent de
  connaitre le nombre de synchronisations pour lesquelles les mailles
  fantômes ont été modifiées. La page
  \ref arcanedoc_debug_perf_compare_synchronization indique comment
  utiliser ce mécanisme (\pr{897}, \pr{898}, \pr{900}, \pr{902},
  \pr{910}, \pr{926}).
- Ajoute deux classes expérimentales \arcane{CartesianMeshCoarsening} et
  \arcane{CartesianMeshCoarsening2} pour dé-raffiner le maillage
  cartésien initial. Cela fonctionne pour l'instant uniquement en 2D
  (\pr{912}, \pr{913}, \pr{917}, \pr{918}, \pr{937}, \pr{942},
  \pr{944}, \pr{945}).
- Ajoute itérateur (\arcane{ICartesianMesh::patches()}) sur les patchs
  des maillages cartésiens (\pr{948}).
- Ajoute classe \arcane{CartesianPatch} pour encapsuler les
  \arcane{ICartesianMeshPatch} (\pr{971}).
- Ajoute pour les statistiques d'échange de messages les valeurs
  cumulées sur toutes les exécutions (\pr{852}, \pr{853}).
- Ajoute support en C# des fonctions du jeu de données (\pr{797},
  \pr{800}, \pr{801}, \pr{803}, \pr{804}).
- Ajoute dans la composante PETSc de Aleph le support des
  préconditionneurs ILU et IC en parallèle (\pr{789}, \pr{799}).

#### API Accélérateur

- Dans \arcaneacc{Reducer}, utilise la mémoire `HostPinned`
  au lieu de la mémoire hôte classique pour conserver la valeur
  réduite. Cela permet d'accélérer la recopie mémoire entre GPU et
  GPU (\pr{782})
- Ajoute macro ARCCORE_HOST_DEVICE manquantes pour les méthodes de
  vérification de tailles de tableau (\pr{785}).
- Ajoute méthode \arcaneacc{RunQueue::platformStream()} pour
  récupérer un pointeur sur l'instance native associée (cudaStream ou
  hipStream par exemple) (\pr{796}).
- Ajoute support accélérateur de la version 6 et la version 8 des
  synchronisations des matériaux (\pr{855}).
- Ajoute support pour récupérer le nombre de milieux d'une maille (\pr{860})
- Ajoute vues sur la variable environnement (\pr{904}).
- Ajoute support pour les alogithmes de Scan inclusifs et exclusifs
  via la classe \arcaneacc{Scanner} (\pr{921}, \pr{923}).
- Ajoute accès accélérateur à certaines méthodes de
  \arcanemat{AllEnvCell} et \arcanemat{EnvCell} (\pr{925}).
- Ajoute support pour le filtrage de tableaux (\pr{954}, \pr{955}).
- Ajoute méthode de copie asynchrone pour \arcane{NumArray} et les
  variables sur le maillage (\pr{961}, \pr{962})
- Ajoute méthode de remplissage asynchrone \arcane{NumArrayBase::fill()}
  (\pr{963}, \pr{964}).

### Changements

- Change le type de retour de
  \arcanemat{IMeshMaterialMng::synchronizeMaterialsInCells()} pour
  retourner un `bool` (au lieu d'un `void`) si les matériaux ont été
  modifiées (\pr{827}).

### Corrections

- Dans le format MSH, corrige mauvaise lecture des groupes de noeuds
  lorsqu'il n'y a qu'un seul élément dans ce groupe (\pr{784}).
- Corrige compilation de \arcane{ArrayExtentsValue} pour la dimension
  2 avec les compilateurs Clang et NVCC (\pr{786}).
- Corrige compilation des exemples en C# si l'environnement n'est pas
  disponible. Positionne aussi la variable d'environnement
  `LD_LIBRARY_PATH` si nécessaire (\pr{811}).
- Corrige le mode d'optimisation
  \arcanemat{eModificationFlags::OptimizeMultiMaterialPerEnvironment}
  pour qu'il ait le même comportement que les autres modes
  d'optimisation de mise à jour des valeurs lors du passage d'une
  maille partielle à une maille pure. Le comportement attendu est de
  prendre la valeur partielle du matériau et on prenait la valeur
  partielle du milieu. Pour garder la compatibilité avec l'existant ce
  mode n'est pas actif par défaut. La méthode
  \arcanemat{IMeshMaterialMng::setUseMaterialValueWhenRemovingPartialValue()}
  permet de l'activer (\pr{844}, \pr{957}).
- Corrige bug dans la synchronisation des \arcane{DoF} dans certaines
  conditions (\pr{920}).
- Corrige fuite mémoire dans la gestion accélérateur de
  \arcanemat{AllEnvCell} (\pr{931}).
- Corrige mauvaise prise en compte des maillages additionnels pour les
  options complexes avec occurences multiples (\pr{941}).


### Interne

- Début support des maillages cartésian avec patchs bloc structuré
  (\pr{946}).
- Début refonte de la gestion de la connectivité des matériaux. Cette
  refonte a pour objectif d'optimiser cette gestion des connectivité
  afin d'éviter d'avoir à appeler
  \arcanemat{IMeshMaterialMng::forceRecompute()}
  après une modification. Ces mécanismes ne sont pas actifs par défaut
  (\pr{783}, \pr{787}, \pr{792}, \pr{794}, \pr{795}, \pr{825},
  \pr{826}, \pr{828}, \pr{829}, \pr{831}, \pr{832}, \pr{835},
  \pr{836}, \pr{838}, \pr{839}, \pr{840}, \pr{841}, \pr{842},
  \pr{843}, \pr{847}).
- Nettoyage et réorganisations internes (\pr{781}, \pr{810}, \pr{813},
  \pr{822}, \pr{830}, \pr{834}, \pr{846}, \pr{856}, \pr{857},
  \pr{868}, \pr{908}, \pr{914}, \pr{952})
- Ajoute en test un service pour se connecter à une base Redis pour
  le service \arcane{BasicReaderWriter} (\pr{780})
- Ajoute sauvegarde au format JSON des meta-données pour les
  protections/reprises. Ce format n'est pas utilisé pour l'instant
  mais il remplacera à terme le format XML (\pr{779}, \pr{865}).
- Créé deux classes \arcane{VariableIOReaderMng} et
  \arcane{VariableIOWriterMng} pour gérer la partie entrées/sorties de
  \arcane{VariableMng} (\pr{777}).
- Ajoute méthode \arcane{JSON::value()} pour retourner une
  \arccore{String} (\pr{778})
- Ajoute nouveau module de test pour les matériaux
  (MaterialHeatModule). Ce module permet de mieux tester les méthodes
  d'ajout et de suppression de mailles matériaux (\pr{788}, \pr{790},
  \pr{824}, \pr{848}).
- Améliore l'usage de \arcane{IProfilingService} (\pr{791})
- Intègre les sources de Alien dans le même dépôt que %Arcane (le
  dépôt *framework*) (\pr{798}, \pr{812}, \pr{816}, \pr{817},
  \pr{819}, \pr{820}, \pr{883}, \pr{890}, \pr{891}, \pr{892}).
- Intègre le gestionnaire de maillage *Neo* dans le dépôt (\pr{802},
  \pr{805}, \pr{807}, \pr{808}, \pr{814}, \pr{815}, \pr{854},
  \pr{881}, \pr{882}, \pr{888}).
- Sépare l'interface \arcane{IIncrementalItemConnectivity} en deux
  interfaces \arcane{IIncrementalItemSourceConnectivity} et
  \arcane{IIncrementalItemTargetConnectivity} pour permettre d'avoir
  des connectivités qui n'ont pas de cibles (\pr{846}).
- Ajoute tests pour la version 3 de \arcane{FaceUniqueidBuilder}
  (\pr{850})
- Débute portage MacOS (\pr{884}, \pr{885})
- Optimise les sorties au format EnsightGold lorsque le nombre de
  types de maille est très grand (maillages polyédriques) (\pr{911}).
- Ajoute API interne à %Arcane pour \arcane{ICartesianMesh}
  (\pr{943}).
- Créé les \arcane{MeshHandle} des maillages additionnels avant
  l'appel aux points d'entrée `Build` (\pr{947}).

### Compilation et Intégration Continue (CI)

- Ajoute compilation des exemples dans le CI (\pr{809}).
- Mise à jour des images docker dans le CI (\pr{818})
- Pour Arccore, compile avec l'option *Z7* sous Windows et ajoute les
  symboles de débug. L'option *Z7* permet d'utiliser l'outil *ccache*
  (\pr{821})
- Utilise 'GitHub actions' pour simplifier le CI. Cela permet d'avoir
  des actions composables de haut niveau plutôt que de dupliquer le
  code dans chaque workflow (\pr{849}).
- Utilise une image docker pour le CI avec 'Circle-CI' et ajoute une
  exécution pour les plateformes ARM64 (\pr{887}).
- Ajoute variable CMake `ARCANEFRAMEWORK_BUILD_COMPONENTS` permettant
  de spécifier les composants à compiler. Cette variable est une liste
  pouvant contenir `%Arcane` et `Alien`. Par défaut on compile les deux
  composants (\pr{877})
- Supprime les artefacts après leur récupération pour gagner de la
  place disque (\pr{915}).
- Supprime activation par défaut de la variable CMake
  `ARCANE_ADD_RPATH_TO_LIBS`. Cette variable est obsolète et sera
  supprimée ultérieurement (\pr{919}).
- Supprime le répertoire de sortie des tests dans le CI pour gagner de
  la place sur le disque (\pr{924}).

### Arccore

- Nettoyage interne de la gestion des statistiques pour les échanges
  de message (\pr{851})
- Corrige bug dans les constructeurs de \arccore{SharedArray2} à
  partir de \arccore{UniqueArray2} ou \arccore{Span} (\pr{899}).

### Axlstar

- Améliore le support des vieilles versions de Doxygen (\pr{823}).

## Arcane Version 3.10.11 (30 juin 2023) {#arcanedoc_version3100}

A partir de cette version 3.10, des modifications internes de gestion
des connectivités sont envisagées afin de diminuer la consommation
mémoire pour la pour les maillages cartésiens ou structurés. La page
\ref arcanedoc_new_optimisations_connectivity décrit les évolutions
envisagées et les éventuelles modifications à effectuer dans les codes
utilisateurs de %Arcane.

### Eléments obsolètes

- Supprime dans les matériaux les méthodes obsolètes depuis plusieurs
  années (\pr{652}).
- Rend obsolète la méthode \arcane{ItemConnectedListView::localIds()}
  permettant d'accéder aux tableaux des localId() des entités
  connectées à une autre (\pr{666}).
- Rend privées ou obsolètes les méthodes internes à %Arcane de
  \arcane{ItemInternalConnectivityList} et \arcane{ItemInternal}
  (\pr{787}).
- Rend obsolète \arcane{IDeflateService}. Il faut utiliser
  \arcane{IDataCompressor} à la place (\pr{706}).
- Rend obsolète \arcane{IPostProcessorWriter::setMesh()} qui ne fait
  rien par défaut. Il faut spécifier le maillage souhaité lors de la
  construction du service (via \arcane{ServiceBuilder}) (\pr{748}).
- Rend obsolète \arcane{IHashAlgorithm::computeHash()}. Il faut
  utiliser la version \arcane{IHashAlgorithm::computeHash64()} à la
  place. Ajoute les méthodes \arcane{IHashAlgorithm::hashSize()} et
  \arcane{IHashAlgorithm::name()} pour récupérer les informations sur
  l'algorithme et pouvoir le créer dynamiquement via un service
  (\pr{696}, \pr{707}).
- Rend obsolète les méthodes \arccore{ArrayView::range()},
  \arccore{Span::range()} et \arccore{AbstractArray::range()}. Ces
  méthodes génèrent des objets temporaires ce qui peut poser un problème
  lorsqu'on les utilise dans des boucles `for-range` (voir rubrique
  'Temporary range expression' de [range-for](https://en.cppreference.com/w/cpp/language/range-for)).
  On peut directement utiliser les méthodes `begin()` ou `end()` à la place (\pr{757}).

### Nouveautés/Améliorations

- Création d'une classe \arcane{SmallArray} pour gérer des tableaux de
  petite taille avec la mémoire allouée sur la pile (\pr{615}, \pr{732}).
- Ajoute possibilité dans l'implémentation PETSc de Aleph de passer
  des arguments qui seront utilisés pour initialiser PETSc via l'appel
  à `PetscInitialize()` (\pr{621}).
- Ajoute écrivain de post-traitement au format `VTK HDF V2`. Ce format
  permet de mettre dans le même fichier au format HDF5 plusieurs
  temps de dépouillement (\pr{637}, \pr{638}, \pr{639}).
- Amélioration de la gestion mémoire des connectivités lors de la
  création du maillage. Les tableaux sont pré-alloués pour éviter des
  recopies successives lorsqu'on ajoute les entités une par une
  (\pr{689}, \pr{763}).
- Ajoute méthode \arcane{MDSpan::slice()} pour retourner une vue sur
  une sous-partie de la vue initiale (\pr{690}).
- Possibilité de calculer dynamiquement le nom du répertoire de sortie
  dans \arcane{SimpleTableWriterHelper} (\pr{607}).
- Ajoute fonctions de calcul de hash SHA3 (\pr{695}, \pr{697}, \pr{705}).
- Ajoute classe \arcane{ItemGenericInfoListView} pour rendre accessible
  sur accélérateur les informations génériques sur les entités (comme
  \arcane{Item::owner()}, \arcane{Item::uniqueId()}, ...) (\pr{727}).
- Optimise \arcane{ItemGroup::applyOperation()} pour ne pas passer par
  des sous-groupes et pour directement utiliser le groupe de base si
  toutes les entités sont de même type. Ce mécanisme n'est pas actif
  par défaut. Il faut positionner la variable d'environnement
  `ARCANE_APPLYOPERATION_VERSION` à `2` pour l'utiliser (\pr{728}).
- Ajoute opérateurs `-=`, `*=` et `/=` pour les vues (via
  \arcane{DataSetter}) {\pr{733}).
- Ajoute classe \arcane{Vector3} pour généraliser \arcane{Real3}
  pour d'autres types (\pr{750}).
- Ajoute évènements pour \arcane{IMesh} pour être notifié des appels
  à \arcane{IMesh::prepareForDump()} \pr{771}.

#### API Accélérateur

- Support pour les copies mémoire avec index sur accélérateur
  (\pr{617}, \pr{625}, \pr{658}, \pr{773})
- Intégration partielle de CUPTI (CUDA Profiling Tools Interface)
  permettant de récupérer automatiquement des informations de
  profiling sur les cartes NVIDIA. Par exemple, cela permet de
  récupérer les informations sur les transferts mémoire entre le CPU
  et le GPU (\pr{627}, \pr{632}, \pr{642}).
- Support en CUDA pour tracer les allocations/désallocations de la
  mémoire managée et pour allouer par blocs multiples de la taille de
  page (\pr{641}, \pr{672}, \pr{685}, \pr{693}).
- Support des synchronisations avec le mode 'Accelerator Aware' de
  MPI. Cela permet de faire des synchronisations sans recopie mémoire
  entre le CPU et l'accélérateur. Ce mécanisme est aussi disponible en
  mode échange de message en mémoire partagé et en mode hybride
  (\pr{631}, \pr{644}, \pr{645}, \pr{646}, \pr{654}, \pr{661},
  \pr{680}, \pr{681}, \pr{765}).
- Support sur accélérateur pour savoir si une adresse mémoire donnée
  est accessible sur accélérateur ou sur le CPU ou les deux. Ajoute
  aussi deux macros ARCANE_CHECK_ACCESSIBLE_POINTER() et
  ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS() pour vérifier qu'une zone
  mémoire pour être utilisée dans une
  \arcaneacc{RunQueue} (\pr{660}).
- Support sur accélérateur pour spécifier des informations sur
  l'allocation mémoire des variables
  (\arcane{IVariable::setAllocationInfo()}). Cela permet par exemple
  d'indiquer si une variable sera plutôt accédée sur accélérateur ou
  sur le CPU (\pr{684}).
- Ajoute méthode \arcane{MeshUtils::markMeshConnectivitiesAsMostlyReadOnly()}
  pour indiquer que les variables gérant la connectivité ne seront pas
  modifiées fréquemment. Cela permet d'optimiser la gestion mémoire
  entre accélérateur et CPU pour éviter des recopies. Par défaut les
  groupes d'entités (\arcane{ItemGroup}) utilisent cet attribut
  (\pr{691}, \pr{714}).
- Rend accessible sur accélérateur les informations conservées dans
  \arcane{Materials::AllEnvCell} (\pr{742}).
- Ajoute dans l'API accélérateur des vues à partir des \arccore{Span}.
  Cela permet d'avoir des vues sur des conteneurs autres
  que ceux de %Arcane (par exemple `std::vector`) à partir du moment
  où les données sont contigues en mémoire (\pr{770}).
- Autorise la copie de \arcane{NumArray} depuis différentes zones
  mémoire (\pr{651}).
- Support pour la nouvelle interface \arccore{IMemoryAllocator3} dans
  les allocateurs pour accélérateur (\pr{671}, \pr{674}).

### Changements

- Autorise la copie d'instance de \arcaneacc{Runner} en
  utilisant une sémantique par référence (\pr{623}).
- Sur accélérateur, utilise par défaut un kernel spécifique sur toute
  la grille pour les réductions. Auparavant, on utilisait un kernel qui
  mélangeait opérations atomiques et calcul sur des blocs (\pr{640}).
- Utilise la mémoire de l'accélérateur pour les opérations de
  réduction. Auparavant, on utilisait la mémoire managée (\pr{643}, \pr{683}).
- Change la numérotation des nœuds des mailles hexagonales pour
  `HoneyCombMeshGenerator` (\pr{657}).
- Regroupe dans le namespace \arcane{MeshUtils} les fonctions de
  \arcane{mesh_utils} et \arcane{meshvisitor} (\pr{725}).
- Déplace les méthodes internes à %Arcane de \arcane{IMesh} et
  \arcane{IItemFamily} dans une interface interne (\pr{726}, \pr{738},
  \pr{752}, \pr{768}).
- Dans les comparaisons bit à bit, ne considère pas qu'il y a des
  différences si les deux valeurs à comparer sont des `NaN`.
- Ajoute affichage du temps passé dans l'initialisation MPI et du
  runtime accélérateur (\pr{760}).
- Appelle automatiquement \arcane{ICartesianMesh::computeDirections()}
  après un appel à \arcane{IMesh::prepareForDump()}. Cela permet de
  garantir que les informations cartésiennes sont cohérentes après un
  éventuel compactage (\pr{772}).

### Corrections

- Gère correctement la destruction des instances singleton de
  \arcane{StandaloneSubDomain}. Avant l'instance était détruite dans
  les destructeurs globaux après la fin de `main()` ce qui pouvait
  poser un problème dans certains cas (\pr{619}).
- Corrige erreurs dans le constructeur de recopie de \arcane{NumArray}
  (\pr{717}).
- Dans \arcane{FloatingPointExceptionSentry}, positionne de manière
  inconditionnelle le flag de gestion des exceptions. Auparavant, on
  testait si les exceptions étaient actives et si ce n'était pas le
  cas, on ne faisait rien. Ce mécanisme de détection n'étant pas
  toujours fiable, il est supprimé (\pr{720}).
- Sauvegarde dans les protections le `timestamp` de modification des
  maillages (\arcane{IMesh::timestamp()}) ainsi que l'attribut
  `need-compact` pour avoir le même
  comportement avec ou sans reprise. Notamment, `need-compact` était
  toujours mis à `true` lors d'une reprise ce qui faisait qu'on
  recompactait toujours au moins une après une reprise. Comme les
  entités étaient compactées cela ne changeait pas les résultats, mais
  pouvait provoquer des réallocations qui invalidaient les structures
  calculées telles que les informations de maillage cartésien
  (\pr{739}, \pr{756})
- Corrige utilisation de \arcane{MeshReaderMng::setUseMeshUnit()} qui
  n'était pas pris en compte lorsque la langue du jeu de donnée est
  en francais (\pr{754}).
- Supprime réallocation inutile dans \arccore{AbstractArray} lorsque la
  nouvelle capacité est identique à l'ancienne (commit
  cac7fae3c471f6).

### Interne

- Débute support pour créer dynamiquement des services contenant des
  options du jeu de données (\pr{613}).
- Supprime utilisation de \arcane{ISubDomain} dans certaines parties
  (\pr{620}).
- Ajoute fonction pour récupérer les arguments de la ligne de
  commande (\pr{624}).
- Supprime avertissements coverity (\pr{626}, \pr{692}).
- Rend `const` certaines méthodes de
  \arcane{Materials::IMeshComponent} (\pr{630}).
- Améliorations diverses de la gestion des accélérateurs (\pr{647}).
- Corrige compilation avec PAPI 7.0 et PETSc 3.19 (\pr{648}).
- Ajoute champ de type \arcane{Int32} dans les différentes classes
  gérant les connectivités pour gérer un offset sur les
  localId(). Pour l'instant cela n'est pas utilisé et l'offset vaut
  toujours 0 (\pr{649}, \pr{712}, \pr{723}, \pr{736}, \pr{737}, \pr{744})
- Support pour utiliser un driver parallèle spécifique pour lancer les tests
  (\pr{663}).
- Remplace l'utilisation de `ENUMERATE_*` pour accéder aux
  connectivités par des for-range (\pr{666}, \pr{759}).
- Ajoute interface spécifique pour créer des maillages cartésiens.
  Cela permettra à terme de fournir des méthodes spécialisées pour ces
  maillages pour que la génération soit plus rapide et directement
  avec la bonne numérotation (\pr{694}, \pr{749}, \pr{751}).
- Ajoute typedefs dans \arcane{MDSpan} pour récupérer le type de
  l'élément et du Layout (\pr{699}).
- Ajoute support pour utiliser des hash communs dans
  \arcane{BasicReaderWriter} ce qui pourra être utilisé pour les
  comparaisons bit à bit et rend générique le mécanisme d'accès à
  cette base de hash (\pr{698}, \pr{700}, \pr{701}).
- Ajoute adaptateur pour la base de donnée Redis {\pr{702}).
- Refonte interne du mécanisme de synchronisation pour le rendre
  indépendant du type de données (\pr{704}, \pr{708}, \pr{709}, \pr{711})
- Utilise un seul buffer pour la synchronisation multiple de variables
  au lieu de passer par la sérialisation (\pr{710}).
- Ajoute classe \arcane{MeshKind} pour gérer les propriétés sur la
  structure de maillage (cartésien, non-structuré, amr, ...)
  (\pr{718}).
- Ajoute macro spécifique pour les
  méthodes obsolètes, mais qui ne seront pas supprimées immédiatement
  afin de pouvoir désactiver les avertissements de compilation pour
  ces méthodes. Cela permet au code utilisateur de supprimer les
  avertissements de compilation pour ces méthodes si la macro
  `ARCANE_NO_DEPRECATED_LONG_TERM` est définie lors de la compilation
  (\pr{722}).
- Ajoute possibilité d'afficher l'affinité CPU de tous les rangs
  (\pr{729}).
- Ajoute pour les formats `VTK HDF` les informations sur les
  \arcane{Item::uniqueId()} des noeuds et des mailles (\pr{741}).
- Amélioration de l'intégration continue pour ne pas lancer les tests
  si seulement certains fichiers sont modifiés (par exemple uniquement
  les `.md`) et ajoute vérification de la date et de la licence
  (\pr{743}, \pr{745}).
- Rend privé les méthodes de \arcane{ItemInternalConnectivityList}
  internes à %Arcane et simplifie la gestion de la classe en regroupant
  les informations de connectivités dans une sous-structure
  (\pr{640}).
- Déplace la classe \arcane{ItemGroupImplPrivate} dans son propre
  fichier (\pr{730}).
- Rend `constexpr` la fonction Arcane::arcaneCheckAt() (\pr{746}).

### Arccore (2.5.0)

- Propage l'allocateur de la source dans le constructeur et
  l'opérateur de recopie de \arccore{UniqueArray} et
  \arccore{UniqueArray2} (\pr{635}, \pr{656}).
- Utilise \arccore{Span} au lieu de \arccore{ConstArrayView} pour
  certains arguments pour permettre des vues dont la taille dépasse
  2Go (\pr{635}).
- Évite la construction d'éléments par défaut qui seront ensuite
  écrasés dans \arccore{AbstractArray::_resizeAndCopyView()}. Cele permet aussi
  d'utiliser cette méthode avec des types de données qui n'ont pas de
  constructeur vide (\pr{635}).
- Ne fais plus d'allocation minimale même si un allocateur autre que
  l'allocateur par défaut est utilisé. Auparavant, on allouait toujours
  au moins 4 éléments (\pr{635}).
- Corrige double allocation inutile dans \arccore{Array::operator=()}
  si les deux tableaux n'ont pas le même allocateur (\pr{655}).
- Permet d'afficher dans le constructeur de \arccore{Exception} le
  message de l'exception. Cela est utile pour débugger par exemple les
  exceptions en dehors de `try{ ... } catch` ou les exceptions qui
  lancent d'autres exceptions (\pr{659}).
- Ajoute interface \arccore{IMemoryAllocator3} qui enrichit
  \arccore{IMemoryAllocator} pour passer plus d'informations à
  l'allocateur. Cela permet d'ajouter par exemple la taille allouée ou
  le nom du tableau (\pr{662}, \pr{673}, \pr{677}, \pr{713}, \pr{719}).
- Ajoute type `Int8` et `BFloat16` dans Arccore::eBasicDataType (\pr{669})
- Ajoute différentes fonctions de conversions entre les \arccore{Span}
  et `std::array`. Ajoute aussi les méthodes `subPart` et
  `subPartInterval` communes à \arccore{ArrayView},
  \arccore{ConstArrayView} et \arccore{Span} (\pr{670}).
- Supprime avertissements coverity (\pr{675}).
- Support pour donner un nom aux tableaux \arccore{Array}. Cela est
  utilisé dans \arccore{IMemoryAllocator3} pour afficher les
  informations d'allocation (\pr{676}, \pr{682}).
- Déplace les opérateurs tels que '==', '!=', '<<' et '<' dans les
  classes correspondantes en tant que fonction `friend` (\pr{703}).
- Rend obsolète les méthodes \arccore{ArrayView::range()},
  \arccore{Span::range()} et \arccore{AbstractArray::range()}. Ces
  méthodes génèrent des objets temporaires ce qui peut poser un problème
  lorsqu'on les utilise dans des boucles `for-range`. On peut directement
  utiliser les méthodes `begin()` ou `end()` à la place (\pr{757}).

### Axlstar (2.2.0)

- Ajoute pour expérimentation la possibilité de spécifier plusieurs
  types (`caseoption`, `subdomain`, ...) pour les services
  (\pr{715}).

## Arcane Version 3.9.5 (04 avril 2023) {#arcanedoc_version390}

### Nouveautés/Améliorations

- Ajoute méthode Arcane::geometric::GeomElementViewBase::setValue()
  pour modifier la valeur d'une coordonnée (\pr{598}).
- Optimise la recherche des valeurs dans les tables de marche en
  utilisant une dichotomie (\pr{596}).
- Support de l'API accélérateur sur les milieux via la macro
  RUNCOMMAND_MAT_ENUMERATE() (\pr{595},\pr{593}, \pr{588}, \pr{586}, \pr{577}).
- Ajoute constructeurs explicite entre Arcane::Real2 et Arcane::Real3
  (\pr{591}).

### Changements

- Ajoute information sur la dimension d'une entité dans
  Arcane::ItemTypeInfo et vérifie la cohérence entre la dimension du
  maillage et les entités Arcane::Cell utilisées dans ce maillage
  (\pr{567}).
- Ajoute convertisseurs de Arcane::ItemEnumeratorT vers Arcane::ItemLocalIdT
  (\pr{564}).

### Corrections

- Corrige récupération du groupe du jeu de données dans le cas de
  maillage multiple. On prenait toujours le maillage par défaut pour
  rechercher les groupes même si l'option était associée à un autre
  maillage (\pr{604}).
- Corrige mauvaise dimension de la barre de recherche dans la
  documentation dans certains cas (\pr{597}).
- Corrige mauvais usage du repartitionneur initial dans le cas où plusieurs
  maillages sont présents dans le jeu de données. Le partitionneur
  définissait les variables uniquement sur le premier maillage ce qui
  introduisait des incohérences (\pr{592}).
- Corrige mauvaise détection des connectivités cartésiennes dans la
  direction Y en 3D (\pr{590}).
- Corrige blocage en parallèle dans le lecteur VTK s'il n'y a que des
  connectivités dans le maillage (\pr{589}).
- Corrige mauvais type de maillage lors de l'utilisation de
  sous-maillage allant d'un maillage 2D à un maillage 1D (\pr{587}).
- Corrige ambiguité possible lors de la construction de classes
  dérivées de Arcane::Item (\pr{579}).

### Interne

- Utilise un compteur de référence pour Arcane::ICaseMng (\pr{603}).
- Support prélimaire pour créér un sous-domaine autonome (\pr{599}).
- Refonte de la gestion des buffers lors des synchronisations des
  variables pour préparer le support des accélérateurs (\pr{585},
  \pr{582}, \pr{575}, \pr{572}, \pr{571}, \pr{570}, \pr{569}, \pr{566}).
- Rend privé les constructeurs de
  Arcane::Materials::ComponentItemVectorView (\pr{580}).
- Modifications diverses dans Arcane::ConstMemoryView et
  Arcane::MutableMemoryView (\pr{574}, \pr{573}, \pr{562}).
- Ajoute version préliminaire d'une interface spécifique à un maillage
  pour l'allocation des mailles (\pr{568}).

### Arccore (version 2.2.0)

- Ajoute support générique via la classe
  Arccore::MessagePassing::GatherMessageInfo pour tous les types de
  `MPI_Gather` (\pr{556}).
- Ajoute distinction dans Arccore::MessagePassing::MessageRank entre
  `MPI_ANY_SOURCE` et `MPI_PROC_NULL` (\pr{555}).

## Arcane Version 3.8.15 (22 février 2023) {#arcanedoc_version380}

### Nouveautés/Améliorations

- Support pour spécifier en ligne de commandes les valeurs
  par défaut de Arcane::ParallelLoopOptions en mode multi-thread
  (\pr{420}).
- Support des fichiers Lima, des fichiers MED et des fichiers
  au format `msh` avec les services de maillage (\pr{435}, \pr{439}, \pr{449}).
- Ajoute fonction Arcane::NumArrayUtils::readFromText() pour remplir une
  instance de Arcane::NumArray à partir d'un fichier au format ASCII (\pr{444}).
- Support de la lecture des parallèle des fichiers au format
  MED (\pr{449}).
- Support pour la lecture des groupes de noeuds
  (Arcane::NodeGroup) dans les maillages au format MSH (\pr{475}).
- Support pour la renumérotation des maillages AMR en 3D. Cela
  permet d'avoir la même numérotation en 3D quel que soit le découpage
  (\pr{495}, \pr{514}, \pr{523}).
- Ajoute accès à Arcane::IMeshMng dans Arcane::ICaseMng et
  Arcane::IPhysicalUnitSystem (\pr{461}).
- Support des accélérateurs pour les classes gérant le
  maillage cartésien (Arcane::CellDirectionMng,
  Arcane::FaceDirectionMng et Arcane::NodeDirectionMng) (\pr{474})
- Ajoute classe Arcane::impl::MutableItemBase pour remplacer
  l'utilisation de Arcane::ItemInternal (\pr{499}).
- Ajoute possibilité d'indexer les composantes de Arcane::Real2,
  Arcane::Real3, Arcane::Real2x2 et Arcane::Real3x3 par l'opérateur
  `operator()` (\pr{485}).
- Développements préliminaires pour les variables du maillage à
  plusieurs dimensions (\pr{459}, \pr{463}, \pr{464}, \pr{466}, \pr{471}).
- Ajoute interface Arcane::IDoFFamily pour gérer les
  Arcane::DoF. Auparavant il fallait utiliser directement
  l'implémentation Arcane::mesh::DoFFamily (\pr{480})
- Support dans Aleph des variables qui n'ont pas de familles
  par défaut (comme les Arcane::DoF par exemple) (\pr{468}).
- Support pour compresser les données Arcane::IData via une instance
  de Arcane::IDataCompressor. Ce mécanisme est disponible pour les
  matériaux en appelant la méthode
  Arcane::Materials::IMeshMaterialMng::setDataCompressorServiceName(). Il
  est utilisé lors des appels à
  Arcane::Materials::IMeshMaterialMng::forceRecompute() ou de
  l'utilisation de la classe Arcane::Materials::MeshMaterialBackup
  (\pr{531}, \pr{532}).
- Support des maillages multiples dans les options du jeu de
  données(\pr{453}, \pr{548}).
- Ajoute classe Arcane::MeshHandleOrMesh pour faciliter la transition
  entre Arcane::IMesh et Arcane::MeshHandle lors de l'initialisation
  (\pr{549}).

### Changements

- Utilise toujours une classe à la place d'un entier pour spécifier
  les dimensions (extents) dans les classes Arcane::NumArray et
  Arcane::MDSpan. Cela permet de se rapprocher de l'implémentation
  prévue dans le norme C++23 et d'avoir des dimensions statiques
  (connues à la compilation) (\pr{419}, \pr{425}, \pr{428}).
- Supprime les timers utilisant le temps CPU au lieu du temps écoulé. Le
  type Arcane::Timer::TimerVirtual existe toujours mais se comporte
  maintenant comme le type Arcane::Timer::TimerReal (\pr{421}).
- Supprime paramètre template avec le rang du tableau dans les classes
  Arcane::DefaultLayout, Arcane::RightLayout et Arcane::LefLayout
  (\pr{436}).
- Rend obsolète les méthodes de Arcane::ModuleBuildInfo qui utilisent
  Arcane::IMesh. Il faut utiliser les méthodes qui utilisent
  Arcane::MeshHandle. (\pr{460}).
- Change le type de retour de Arcane::IMeshBase::handle() pour ne pas
  retourner de référence mais une valeur (\pr{489}).
- Utilise des classes de base spécifiques par type de service lors de
  la génération des fichiers `axl` (\pr{472}).
- Utilise une nouvelle classe Arcane::ItemConnectedListView (au lieu
  de Arcane::ItemVectorView) pour gérer les connectivités sur les
  entités. Les méthodes telles que Arcane::Cell::nodes() retournent
  maintenant un objet de ce type. Le but de cette nouvelle classe est
  de pouvoir proposer une structure de données spécique aux
  connectivités et une autre spécifique aux listes d'entités (telles
  que les groupes Arcane::ItemGroup). Des opérateurs de conversion ont
  été ajoutés afin de garantir la compatibilité avec le code source
  existant (\pr{534},\pr{535}, \pr{537}, \pr{539})
- Nouveau service de dépouillement au format VTK HDF. Ce format est
  uniquement supporté par les versions 5.11+ de
  Paraview. L'implémentation actuelle est expérimentale (\pr{510},
  \pr{525}, \pr{527}, \pr{528} \pr{554}, \pr{546}).
- Déplace les fichiers d'en-tête de la composante `arcane_core` et qui
  se trouvaient à la racine de `arcane` dans un sous-répertoire
  `arcane/core`. Afin de rester compatible avec l'existant des
  fichiers d'en-tête faisant références à ces nouveaux fichiers sont
  générés lors de l'installation.
- Ajoute une composante `arcane_hdf5` contenant les classes
  utilitaires (telles que Arcane::Hdf5::HFile, ...). Les fichiers
  d'en-te correspondants sont maintenant dans le répertoire
  `arcane/hdf5` (\pr{505}).
- Nettoyages des classes gérant HDF5: rend obsolète le constructeur de
  recopie des classes de `Hdf5Utils.h` et supprime le support des
  versions de HDF5 antérieures à la version 1.10 (\pr{526}).
- Autorise la création d'instances nulle de Arcane::ItemGroup avant
  l'initialisation de %Arcane. Cela permet par exemple d'avoir des
  variables globales de Arcane::ItemGroup ou des classess dérivées
  (\pr{544}).
- Modifie le comportement de la macro ENUMERATE_COMPONENTITEM pour
  utiliser un type plutôt qu'une chaîne de caractères dans le nom de
  l'énumérateur. Cela permet de l'utiliser avec un paramètre template
  (\pr{540}).

### Corrections

- Corrige inconsistence possible entre les connectivités conservées
  dans Arcane::ItemConnectivityList et
  Arcane::mesh::IncrementalItemConnectivity (\pr{478}).
- Corrige mauvaise valeur de Arcane::HashTableMapT::count() après un
  appel à Arcane::HashTableMapT::clear() (\pr{506}).

### Interne

- Supprime classes internes obsolètes Arcane::IAllocator,
  Arcane::DefaultAllocator, Arcane::DataVector1D,
  Arcane::DataVectorCommond1D et Arcane::Dictionary. Ces classes ne
  sont plus utilisées depuis longtemps (\pr{422}).
- Ajoute classe Arcane::TestLogger pour comparer les résultats des
  tests par rapport à un fichier listing de référence (\pr{418}).
- Ajoute possibilité de conserver les instances de
  Arcane::ItemSharedInfo' dans
  Arcane::ItemInternalConnectivityList'. Cela permettra de supprimer
  une indirection lors des accès aux connectivités. Cette option est
  pour l'instant uniquement utilisée en phase de test (\pr{371})
- Ajoute support pour l'appel Arccore::MessagePassing::mpLegacyProbe()
  pour les différents modes d'échange de message disponibles (\pr{431})
- Refactorisation des classes Arcane::NumArray, Arcane::MDSpan,
  Arcane::ArrayExtents et Arcane::ArrayBounds pour unifier le code
  et supporter des dimensions à la fois statiques et dynamiques. La
  page \ref arcanedoc_core_types_numarray explique l'utilisation de
  ces classes (\pr{426}, \pr{428}, \pr{433}, \pr{437}, \pr{440}).
- Utilise par défaut la Version 2 des synchronisations avec
  MPI. Cette version est la même que la version 1 utilisée auparavant
  mais sans le support des types dérivés (\pr{434}).
- [accelerator] Unifie le lancement des noyaux de calcul créés par les
  macros RUNCOMMAND_LOOP et RUNCOMMAND_ENUMERATE (\pr{438}).
- Unifie l'API de profiling entre les commandes
  (Arcane::Accelerator::RunCommand) et les énumérateurs classiques (via
  Arcane::IItemEnumeratorTracer). En fin de calcul l'affichage est trié
  par ordre décroissant du temps passé dans chaque boucle (\pr{442}, \pr{443}).
- Commence le développement des classes Arcane::NumVector and
  Arcane::NumMatrix pour généraliser les types Arcane::Real2,
  Arcane::Real3, Arcane::Real2x2 et Arcane::Real3x3. Ces classes sont
  pour l'instant à usage interne de %Arcane (\pr{441}).
- Diverses optimisations dans les classes internes gérant la
  connectivités et les itérateurs pour réduire leur taille (\pr{479}, \pr{482},
  \pr{483}, \pr{484})
- Supprime utilisation de Arcane::ItemInternalList dans
  Arcane::ItemVector et Arcane::ItemVectorView (\pr{486}, \pr{487}).
- Supprime utilisation de Arcane::ItemInternalVectorView (\pr{498})
- Supprime utilisation de Arcane::ItemInternal dans de nombreuses
  classes internes (\pr{488}, \pr{492}, \pr{500}, \pr{501}, \pr{502})
- Supprime dans Arcane::MDSpan et Arcane::NumArray les indexeurs
  supplémentaires pour les classes Arcane::Real2, Arcane::Real3,
  Arcane::Real2x2 et Arcane::Real3x3. Ces indexeurs avaient été
  ajoutés à des fin de test mais n'étaient pas utilisés (\pr{490}).
- Autorise de copier les instances de Arcane::StandaloneAcceleratorMng
  et utilise une sémantique par référence (\pr{509}).
- Autorise des instance de Arcane::NumVector et Arcane::NumMatrix avec
  des valeurs quelconques (auparant seules les valeurs 2 ou 3 étaient
  autorisées) (\pr{521})
- Déplace l'implémentation des classes liées à Aleph dans les fichiers
  sources au lieu des fichiers d'en-tête (\pr{504}).
- Fournit une implémentation vide pour les méthodes utilisant
  Arcane::IMultiArray2Data. Cette interface n'est plus utilisée et
  cela permettra au code utilisateur de supprimer les visiteurs
  associés à ce type (\pr{529}).

### Arccore

- Corrige bug si la méthode
  Arccore::MessagePassing::Mpi::MpiAdapter::waitSomeRequestsMPI() est
  appelée avec des requêtes déjà terminées (\pr{423}).
- Ajoute dans Arccore::Span et Arccore::Span un paramètre template
  indiquant le nombre d'éléments de la vue. Cela permettra de gérer
  des vues avec un nombre d'éléments connus à la compilation comme
  cela est possible avec `std::span` (\pr{424}).
- Ajoute fonctions Arccore::MessagePassing::mpLegacyProbe() dont la
  sémantique est similaire à `MPI_Iprobe` and `MPI_Probe` (\pr{430}).
- Corrige détection de la requête vide (\pr{427}, \pr{429}).
- Amélioration diverses du mécanisme d'intégration continue (\pr{503},
  \pr{511}, \pr{512}, \pr{513})

### Axlstar

- Add support for using a specific mesh in service instance (\pr{451})
- Remove support to build with `mono` (\pr{465}).
- Remove support for 'DualNode' and 'Link' items (\pr{524}).
- Various improvements in documentation (\pr{530}).
- Add preliminary support for multi-dimension variables (\pr{520}).
- Fix: Add support of Doxygen commands in AXL descriptions (\pr{538})
- Fix: error with complex options containing more than 30 suboptions
  (\pr{533})

___

## Arcane Version 3.7.23 (17 novembre 2022) {#arcanedoc_version370}

### Nouveautés/Améliorations:

- Refonte complète de la documentation afin qu'elle soit plus
  cohérente et visuellement plus agréable (\pr{378}, \pr{380}, \pr{382}, \pr{384},
  \pr{388}, \pr{390}, \pr{393}, \pr{396})
- Ajoute un service de gestion de sorties au format CSV (voir
  \ref arcanedoc_services_modules_simplecsvoutput) (\pr{277}, \pr{362})
- Ajoute possibilité de spécifier le mot clé `Auto` pour la variable
  CMake `ARCANE_DEFAULT_PARTITIONER`. Cela permet de choisir
  automatiquement lors de la configuration le partitionneur utilisé
  en fonction de ceux disponibles (\pr{279}).
- Ajoute implémentation des synchronisations qui utilise la fonction
  `MPI_Neighbor_alltoallv` (\pr{281}).
- Réduction de l'empreinte mémoire utilisée pour la gestion des
  connectivités suite aux différentes modifications internes
- Optimisations lors de l'initialisation (\pr{302}):
  - Utilise `std::unordered_set` à la place de `std::set` pour les
    vérifications de duplication des uniqueId().
  - Lors de la création de maillage, ne vérifie la non-duplication des
    uniqueId() que en mode check.
- Crée une classe Arcane::ItemInfoListView pour remplacer à terme
  Arcane::ItemInternalList et accéder aux informations des entités à
  partir de leur localId() (\pr{305}).
- [accelerator] Ajoute le support des réductions Min/Max/Sum atomiques
  pour les types `Int32`, `Int64` et `double` (\pr{353}).
- [accelerator] Ajoute nouvel algorithme de réduction sans passer par
  des opérations atomiques. Cet algorithme n'est pas utilisé par
  défaut. Il faut l'activer en appelant
  Arcane::Accelerator::Runner::setDeviceReducePolicy() (\pr{365}, \pr{379})
- [accelerator] Ajoute possibilité de changer le nombre de threads par
  bloc lors du lancement d'une commande via
  Arcane::Accelerator::RunCommand::addNbThreadPerBlock() (\pr{374})
- [accelerator] Ajoute support pour le pre-chargement (prefetching) de
  conseils (advice) de zones mémoire (\pr{381})
- [accelerator] Ajoute support pour récupérer les informations sur les
  accélérateurs disponibles et associer un accélérateur à une instance
  de Arcane::Accelerator::Runner (\pr{399}).
- Début des développements pour pouvoir voir une variable tableau sur
  les entités comme une variable multi-dimensionnelle (\pr{335}).
- Ajoute un observable Arcane::MeshHandle::onDestroyObservable() pour
  pouvoir être notifié lors de la destruction d'une instance de
  maillage Arcane::IMesh (\pr{336}).
- Ajoute méthode Arcane::mesh_utils::dumpSynchronizerTopologyJSON()
  pour sauver au format JSON la topologie de communication pour les
  synchronisation (\pr{360}).
- Ajoute méthode Arcane::ICartesianMesh::refinePatch3D() pour raffiner
  un maillage 3D en plusieurs patchs AMR (\pr{386}).
- Ajoute implémentation de lecture de compteurs hardware via l'API
  perf de Linux (\pr{391}).
- Ajoute support pour le profiling automatiquement des commandes
  lancées via RUNCOMMAND_ENUMERATE (\pr{392}, \pr{394}, \pr{395})

### Changements:

- Modifie les classes associées à Arcane::NumArray
  (Arcane::MDSpan, Arcane::ArrayBounds, ...) pour que le paramètre
  template gérant le rang soit une classe et pas un entier. Le but à
  terme est d'avoir les mêmes paramètres templates que les classes
  `std::mdspan` et `std::mdarray` prévues pour les normes 2023 et 2026
  du C++. Il faut donc maintenant remplacer les dimensions en dur par
  les mots clés Arcane::MDDim1, Arcane::MDDim2, Arcane::MDDim3 ou
  Arcane::MDDim4 (\pr{333})
- La méthode Arcane::NumArray::resize() n'appelle plus le constructeur
  par défaut pour les éléments du tableau. C'était déjà le cas pour
  les types simples (Arcane::Real, Arcane::Real3, ...) mais maintenant
  c'est le cas aussi pour les types utilisateurs. Cela permet
  d'appeler cette méthode lorsque la mémoire est allouée sur l'accélérateur.
- Ajoute classe Arcane::ItemTypeId pour gérer le type de l'entité (\pr{294})
- Le type de l'entité est maintenant conservé sur un Arcane::Int16 au
  lieu d'un Arcane::Int32 (\pr{294})
- Supprime méthodes obsolètes de Arcane::ItemVector, `MathUtils.h`,
  Arcane::IApplication, Arcane::Properties, Arcane::IItemFamily
  (\pr{304}).
- Refonte des classes gérant l'énumération des entités (\pr{308}, \pr{364}, \pr{366}).
  - Supprime la classe de base Arcane::ItemEnumerator de
    Arcane::ItemEnumeratorT. L'héritage est remplacé par un opérateur
    de conversion.
  - Simplifie Arcane::ItemVectorViewConstIterator
  - Simplifie la gestion interne de l'opérateur `operator*` pour ne
    pas utiliser Arcane::ItemInternal.
- Refonte de la gestion du fichier de configuration
  `ArcaneConfig.cmake` géneré (\pr{318}):
  - N'exporte plus par défaut les packages externes dans
    `ArcaneTargets.cmake`. Le fichier `ArcaneConfig.cmake` fait
    maintenant des appels à la commande CMake `find_dependency`. La
    variable CMake `FRAMEWORK_NO_EXPORT_PACKAGES` n'est donc plus
    utilisé par défaut.
  - Ajoute dans `ArcaneConfig.cmake` la variable
    `ARCANE_USE_CONFIGURATION_PATH` pour permettre de charger les
    chemins des packages issus de la configuration de Arcane. Cette
    variable est positionnée à `TRUE` définie par défaut.
- Modifie le prototype de certaines méthodes de classes implémentant
  Arcane::IItemFamily pour utiliser Arcane::Item à la place de
  Arcane::ItemInternal (\pr{311})
- Crée une classe Arcane::ItemFlags pour gérer les flags concernant
  les propriétés des objets qui étaient avant dans
  Arcane::ItemInternal (\pr{312})
- Rend obsolète l'opérateur `operator->` pour la classe Arcane::Item
  et les classes dérivées (\pr{313})
- Change la valeur par défaut de la numérotation des faces dans le
  service de génération cartésien pour utiliser la numérotation
  cartésienne (\pr{315})
- Modification de la signature des méthodes de
  Arcane::IItemFamilyModifier et Arcane::mesh::OneMeshItemAdder
  pour utiliser Arcane::ItemTypeId au lieu de Arcane::ItemTypeInfo
  et Arcane::Item au lieu de Arcane::ItemInternal (\pr{322})
- Supprime méthodes Arcane::Item::activeFaces() et
  Arcane::Item::activeEdges() qui ne sont plus utilisées (\pr{351}).
- [C#] Ajoute la possibilité en fin de calcul de détruire les
  instances des différents gestionnaires comme lorsque le support de
  `.Net` n'est pas activé. Auparavant ces gestionnaires n'étaient
  jamais détruit pour éviter des plantages potentiels lorsque le
  'garbage collector' de l'environnement `.Net` se déclenche. Il est
  possible d'activer cette destruction en positionnant la variable
  d'environnement `ARCANE_DOTNET_USE_LEGACY_DESTROY` à la valeur
  `0`. Cela n'est pas actif par défaut car il peut rester des
  problèmes avec certains services utilisateurs (\pr{337}).
- [configuration] Il est maintenant nécessaire d'utiliser au moins la
  version 3.21 de CMake pour compiler ou utiliser #Arcane (\pr{367}).
- Ajoute constructeur par déplacement (`std::move`) pour
  Arcane::NumArray (\pr{372}).
- [accelerator] Supprime les méthodes obsolètes de création de
  Arcane::Accelerator::RunQueue et Arcane::Accelerator::Runner (\pr{397}).
- Rend obsolète la classe Arcane::AtomicInt32. Il faut utiliser
  la classe std::atomic<Int32> à la place (\pr{408}).

### Corrections:

- Corrige bug lors de la lecture des informations avec le service
  `BasicReaderwriter` lorsqu'une compression est active (\pr{299})
- Corrige bug introduit dans la version 3.6 qui changeait le nom du
  répertoire de sortie pour les comparaisons bit à bit avec le service
  `ArcaneCeaVerifier` (\pr{300}).
- Corrige mauvais recalcul du nombre maximum d'entités connectées à
  une entité dans le cas des particules (\pr{301})

### Interne:

- Simplifie l'implémentation des synchronisations fournissant un
  mécanisme indépendant du type de la donnée (\pr{282}).
- Utilise des variables pour gérer certaines données sur les entités
  telles que Arcane::Item::owner(), Arcane::Item::itemTypeId(). Cela
  permettra à terme de rendre ces informations accessibles sur
  accélérateurs (\pr{284}, \pr{285}, \pr{292}, \pr{295})
- Ajout d'une classe Arcane::ItemBase servant de classe de base pour
  Arcane::Item et Arcane::ItemInternal (\pr{298}, \pr{363}).
- Suppression d'une indirection lorsqu'on accède aux informations des
  connectivités à partir d'une entité (par exemple
  Arcane::Cell::node()) (\pr{298}).
- Simplification de la gestion des informations communes aux entités
  dans une famille pour qu'il n'y ait maintenant plus qu'une seule
  instance commune de Arcane::ItemSharedInfo (\pr{290}, \pr{292}, \pr{297}).
- Supprime certains usages de Arcane::ISubDomain (\pr{327})
  - Ajoute possibilité de créer une instance de Arcane::ServiceBuilder
    à partir d'un Arcane::MeshHandle.
  - Ajoute possibilité de créer une instance de
    Arcane::VariableBuildInfo via un Arcane::IVariableMng.
- Optimise les structures gérant le maillage cartésien pour ne plus
  avoir à conserver les instances de Arcane::ItemInternal*. Cela
  permet de réduire la consommation mémoire et potentiellement
  d'améliorer les performances (\pr{345}).
- Utilise des vues au lieu de Arccore::SharedArray pour les classes
  gérant les directions cartésiennes (Arcane::CellDirectionMng,
  Arcane::FaceDirectionMng et Arcane::NodeDirectionMng) (\pr{347}).
- Utilise un compteur de référence pour gérer
  Arccore::Ref<Arcane::ICaseFunction> (\pr{329}, \pr{356}).
- Ajoute constructeur pour la classe Arcane::Item et ses classes
  dérivées à partir d'un localId() et d'un Arcane::ItemSharedInfo (\pr{357}).
- Mise à jour des références des projets C# pour utiliser les
  dernières version des packages (\pr{359}).
- Nettoyage des classes Arcane::Real2, Arcane::Real3, Arcane::Real2x2
  et Arcane::Real3x3 et ajoute constructeurs à partir d'un
  Arcane::Real (\pr{370}, \pr{373}).
- Refonte partiel de la gestion de la concurrence pour mutualiser
  certaines fonctionnalités (\pr{389}).
- Utilise un Arccore::UniqueArray pour conteneur de
  Arcane::ListImplT. Auparavant le conteneur était un simple tableau C
  (\pr{407}).
- Dans Arcane::ItemGroupImpl, utilise Arcane::AutoRefT pour conserver
  des référence aux sous-groupes à la place d'un simple
  pointeur. Cela permet de garantir que les sous-groupes ne seront pas
  détruits tant que le parent associé existe.
- Corrige divers avertissements signalés par coverity (\pr{402}, \pr{403},
  \pr{405}, \pr{409}, \pr{410} )
- [C#] Indique qu'il faut au moins la version 8.0 du langage.

### Arccon:

Utilise la version 1.5.0:

- Add CMake functions to unify handling of packages  arccon Arccon
  componentbuildBuild configuration (\pr{342}).

### Arccore:

Utilise la version 2.0.12.0:

- Remove some coverity warnings (\pr{400})
- Use a reference counter for IMessagePassingMng (\pr{400})
- Fix method asBytes() with non-const types (\pr{400})
- Add a method in AbstractArray to resize without initializing (\pr{400})
- Make class ThreadPrivateStorage deprecated (\pr{400})

___
## Arcane Version 3.6.13 (06 juillet 2022) {#arcanedoc_news_changelog_version360}

### Nouveautés/Améliorations:

- Ajout d'une interface Arcane::IRandomNumberGenerator pour un service
  de génération de nombre aléatoire (\pr{266})
- Ajoute support des variables matériaux dans les fichiers `axl` pour
  le générateur C# (\pr{273})
- Supprime allocation de la connectivité des noeuds dans les anciennes
  connectivités. Cela permet de réduire l'empreinte mémoire (\pr{231}).
- Ajoute pour les classes Arccore::Span, Arccore::ArrayView,
  Arccore::ConstArrayView ainsi que les vues sur les variable
  l'opérateur 'operator()' qui se comporte comme l'opérateur
  'operator[]'. Cela permet d'uniformiser les écritures entre les
  différents conteneurs et les vues associées (\pr{223}, \pr{222}, \pr{205}).
- Ajoute dans Arcane::ICartesianMeshGenerationInfo les informations
  sur l'origine et la dimension du maillage cartésien (\pr{221}).
- Ajoute en fin d'exécution des statistiques collectives sur les temps
  passés dans les opérations d'échange de message. Ces statistiques
  comprennent le temps minimal, maximal et moyen pour l'ensemble des
  rangs passés dans ces appels (\pr{220})
- Ajoute deux implémentations supplémentaires pour la synchronisation
  des matériaux. La version 7 permet de faire une seule allocation
  lors de cette synchronisation et la version 8 permet de conserver
  cette allocation d'une synchronisation à l'autre (\pr{219}).
- Ajoute implémentation des synchronisations en plusieurs phases
  permettant d'utiliser des tableaux de taille fixe et/ou de faire sur
  une sous-partie des voisins (\pr{214}).
- Ajoute accès pour les accélérateurs à certaines méthodes de
  Arcane::MDSpan (\pr{217}).
- Ajoute accès aux connectivités aux arêtes dans
  Arcane::UnstructuredMeshConnectivityView (\pr{216})
- Ajoute interface accessible via
  'Arcane::IMesh::indexedConnectivityMng()' permet de facilement
  ajouter de nouvelles connectivités (\pr{201}).
- Ajout d'un nouvel algorithme de calcul des uniqueId() des arêtes
  (Edge) pour les maillages cartésiens
- Ajoute support pour les classes de Arccore de l'opérateur
  `operator[]` avec arguments multiples (\pr{241}).
- Possibilité de rendre thread-safe les appels à
  Arcane::Accelerator::makeQueue() en appelant la méthode
  Arcane::Accelerator::Runner::setConcurrentQueueCreation() (\pr{242})

### Changements:

- Scinde en deux composantes les classes gérant les matériaux. Une
  partie est maintenant dans la composante `arcane_core`. Ce
  changement est normalement transparent pour les utilisateurs
  d'%Arcane et il n'y a pas besoin de modifier les sources (\pr{264},\pr{270},\pr{274})
- Compacte les références après un appel à
  Arcane::IItemFamily::compactItems(). Cela permet d'éviter de faire
  grossir inutilement le tableau des contenant les informations
  internes des entités. Comme ce changement peut induire une
  différence sur l'ordre de certaines opérations, il est possible de
  le désactiver en positionnant la variable d'environnement
  `ARCANE_USE_LEGACY_COMPACT_ITEMS` à la valeur `1` (\pr{225}).
- Les types gérant le `localId()` associés aux entités (Arcane::NodeLocalId,
  Arcane::CellLocalId, ...) sont maintenant des `typedef` d'une classe
  template Arcane::ItemLocalIdT.
- Supprime dans les opérateurs d'accès aux variables (operator[])
  les différentes surcharges. Il faut maintenant utiliser un
  'Arcane::ItemLocalIdT' comme indexeur. Des opérateurs de conversion
  vers ce type ont été ajoutés pour que le code source reste compatible.
- Désenregistre automatiquement les variables encores allouées
  lorsqu'on appelle Arcane::IVariableMng::removeAllVariables(). Cela
  permet d'éviter des plantages lorsque des références aux variables
  existaient encore après cet appel. Cela peut notamment être le cas
  avec les extensions C# car les services et modules associés sont
  gérés par un 'Garbage Collector' (\pr{200}).
- Rend obsolète l'utilisation de Arcane::Timer::TimerVirtual. Les timers qui
  utilisent cette propriété se comportent comme s'ils avaient
  l'attribut Arcane::Timer::TimerReal.

### Corrections:

- Corrige mauvaises valeurs de Arcane::IItemFamily::localConnectivityInfos()
  et Arcane::IItemFamily::globalConnectivityInfos() pour les connectivités
  autres que celles aux noeuds. Ce bug avait été introduit lors du
  passage aux nouvelles connectivités (\pr{230}, \pr{27}).
- Corrige divers bugs dans la version 3 de BasicReaderWriter (\pr{238})

### Interne:

- Utilise des variables pour conserver les champs Arcane::ItemInternal::owner() et
  Arcane::ItemInternal::flags() au lieu de conserver l'information dans
  Arcane::ItemSharedInfo. Cela permettra à terme de supprimer le champ
  correspondant dans Arcane::ItemSharedInfo (\pr{227}).

Passage version 2.0.3.0 de Axlstar:

  - Ajoute support dans les fichiers 'axl' des propriétés Arcane::IVariable::PNoExchange,
    Arcane::IVariable::PNoReplicaSync et Arcane::IVariable::PPersistant.

Passage version 2.0.11.0 de %Arccore:

  - Add function `mpDelete()` to destroy `IMessagePassingMng` instances (\pr{258})
  - Optimizations in class `String`(\pr{256},\pr{247})
      - Add move constructor String(String&&) and move copy operator operator=(String&&)
      - Make `String` destructor inline 
      - Make method `String::utf16()` deprecated (replaced by `StringUtils::asUtf16BE()`)
      - Methods `String::bytes()` and `String::format` no longer throws exceptions
      - Add a namespace `StringUtils` to contains utilitarian functions.
  - Add support for multisubscript `operator[]` from C++23 (\pr{241})
  - Add `operator()` to access values of `ArrayView`, `ArrayView2`,
    `ArrayView3`, `ArrayView4`, `Span`, `Span2` and `const` versions
    of these views (\pr{223}).
  - Add `SmallSpan2` implementation for 2D arrays whose `size_type` is an `Int32` (\pr{223}).
  - Add `SpanImpl::findFirst()` method  (\pr{211})
  - Fix build on Ubuntu 22.04




___
## Arcane Version 3.5.7 (07 avril 2022) {#arcanedoc_news_changelog_version350}

### Nouveautés/Améliorations:

- Ajoute classe Arcane::SimdReal3x3 et Arcane::SimdReal2x2 qui sont
  l'équivalent vectorielle des Arcane::Real3x3 et Arcane::Real2x2
- Support de la version 4.2 des fichiers de maillage au format VTK
- Ajoute une nouvelle implémentation des synchronisations qui utilise
  l'appel `MPI_Sendrecv`.
- Ajoute possibilité d'utiliser des messages collectifs
  (MPI_AllToAllv) au lieu de messages point à point lors de l´échange
  des entités suite à un équilibrage de charge. Ce mécanisme est
  temporairement accessible en spécifiant la variable d'environnement
  `ARCANE_MESH_EXCHANGE_USE_COLLECTIVE` (\pr{138},\pr{154}).
- Dans la comparaison bit à bit, ajoute possibilité de ne faire la
  comparaison qu'à la fin de l'exécution au lieu de le faire à chaque
  pas de temps. Cela se fait en spécifiant la variable d'environnement
  STDENV_VERIF_ONLY_AT_EXIT.
- Ajoute générateur de maillages en nid d'abeille en 3D (\pr{149}).
- Ajoute support pour spécifier la disposition des éléments (layout)
  dans la classe Arcane::NumArray. Il existe actuellement deux
  dispositions implémentées: LeftLayout et RightLayout (\pr{151})
- Ajoute méthode Arcane::Accelerator::RunQueue::copyMemory() pour faire des
  copies mémoire asynchrones (\pr{152}).
- Améliore le support ROCM/HIP. Le support des GPU AMD est maintenant
  fonctionnellement équivalent à celui des GPU NVIDIA via Cuda (\pr{158}, \pr{159}).
- Ajoute support pour la mémoire punaisée (Host Pinned Memory) pour
  CUDA et ROCM (\pr{147}).
- Ajoute classe 'Arcane::Accelerator::RunQueueEvent' pour supporter
  les évènements sur les 'Arcane::Accelerator::RunQueue' et permettre
  ainsi de synchroniser entre elle des files différentes (\pr{161}).

### Changements:

- Supprime les macros plus utilisées ARCANE_PROXY et ARCANE_TRACE (\pr{145})

### Corrections:

- Corrige mauvaise détection de la version OneTBB 2021.5 suite à la
  suppression du fichier 'tbb_thread.h' (\pr{146})
- Corrige certaines informations cartésiennes manquantes lorsqu'il n'y
  a qu'une seule couche de maille en Y ou Z (\pr{162}).
- Corrige implémentation manquante de 'Arccore::Span<T>::operator=='
  lorsque le type `T` n'est pas constant (\pr{163}).
- Supprime quelques messages listings trop nombreux.

### Interne:

- Utilise au lieu de Arccore::UniqueArray2 une implémentation
  spécifique pour le conteneur de Arcane::NumArray (\pr{150}).
- Utilise des `Int32` au lieu de `Int64` pour indexer les éléments
  dans Arcane::NumArray (\pr{153})
- Ajoute `constexpr` et `noexcept` à certaines classes de Arccore
  (\pr{156}).
- Passage version 2.0.9.0 de Arccore




___
## Arcane Version 3.4.5 (10 février 2022) {#arcanedoc_news_changelog_version340}

### Nouveautés/Améliorations:

- Dans l'API accélérateur, support dans Arcane::NumArray pour allouer
  directement la mémoire sur l'accélérateur. Auparavant seule la
  mémoire unifiée était disponible. L'énumération
  Arcane::eMemoryRessource et le type Arcane::IMemoryRessourceMng
  permettent de gérer cela (\pr{111}, \pr{113}).
- Amélioration mineures sur la documentation (\pr{117}) :
  - ajout des chemins relatifs pour les fichiers d'en-tête.
  - ajout des classes et type issus de %Arccore
- Ajoute nouvelle méthode de calcul des uniqueId() des faces
  dans le cas cartésien. Cette nouvelle méthode permet une
  numérotation cartésienne des faces qui est cohérente avec celles des
  noeuds et des mailles. Pour l'utiliser, il faut spécifier l'option
  `<face-numbering-version>4</face-numbering-version>` dans le jeu de
  données dans la balise du générateur de maillage (\pr{104}).
- Ajoute option dans le post-processeur %Arcane pour supprimer la
  sortie de dépouillement en fin de calcul.
- Ajoute implémentation de Arcane::IParallelMng::gather() et
  Arcane::IParallelMng::gatherVariable() pour le mode mémoire partagée et le
  mode hybride
- Ajoute dans Arcane::Materials::MeshMaterialInfo la liste des milieux
  dans lequels le matériau est présent
- Support de la compilation avec le compilation NVIDIA HPC SDK.
- Support (partiel) pour désallouer le maillage
  (Arcane::IPrimaryMesh::deallocate()) ce qui permet de le
  réallouer à nouveau par la suite.
- Ajoute générateur de maillage 2D en nid d'abeille.

### Changements:

- Ajoute namespace '%Arcane::' aux cibles `CMake` fournit par
%Arcane. Par exemple, la cible 'arcane_core' devient
'Arcane::arcane_core'. Les anciens noms restent valides (\pr{120}).
- Rend obsolète la conversion de Arcane::ItemEnumerator vers
  Arcane::ItemEnumeratorT. Cela permet d'éviter d'indexer par erreur
  une variable du maillage avec un énumérateur du mauvais type (par
  exemple indexer une variable aux mailles avec un énumérateur aux noeuds).

### Corrections:

- Corrige opérateur 'operator=' pour la classe
  'Arcane::CellDirectionMng' (\pr{109})
- Corrige conversion inutile `Int64` vers
  `Int32` dans la construction des maillages cartésiens ce qui
  empêchait de dépasser 2^31 mailles (\pr{98})
- Corrige mauvais calcul des temps passés dans les
  synchronisations. Seul le temps de la dernière attente était utilisé
  au lieu du cumul (commit cf2cade961)
- Corrige nom prise en compte de l'option 'MessagePassingService' en
  ligne de commande (commit 15670db4)

### Interne:

- Nettoyage de l'API Accélérateur

Passage version 2.0.8.1 de %Arccore:

  - Improve doxygen documentation for types et classes in `message_passing` component.
  - Add functions in `message_passing` component to handle non blocking collectives (\pr{116}, \pr{118})
  - Add some '#defines'  to compile with [hipSYCL](https://github.com/illuhad/hipSYCL).
  - Update '_clang-format' file for version 13 of LLVM/Clang.





___
## Arcane Version 3.3.0 (16 décembre 2021) {#arcanedoc_news_changelog_version330}

### Nouveautés/Améliorations:

- Ajoute possibilité de spécifier le nombre maximum de messages en vol
  lors d'un équilibrage de charge. Pour l'instant cela se fait en
  spécifiant la variable d'environnement
  ARCANE_MESH_EXCHANGE_MAX_PENDING_MESSAGE.
- Ajoute possibilité d'utiliser les Arcane::Real2x2 et Arcane::Real3x3 sur accélérateurs
- Ajoute méthode Arcane::mesh_utils::printMeshGroupsMemoryUsage() pour
  afficher la consommation mémoire associée aux groupes et
  Arcane::mesh_utils::shrinkMeshGroups() pour redimensionner au plus
  juste la mémoire utilisée par les groupes
- Support pour punaiser les threads (voir \ref arcanedoc_execution_launcher)

### Changements:

- Ajoute espace de nom Arcane::ParallelMngUtils pour contenir les fonctions
  utilitaires de Arcane::IParallelMng au lieu d'utiliser les méthodes
  virtuelles de cette interface. Les nouvelles méthodes remplacent
  Arcane::IParallelMng::createGetVariablesValuesOperation(),
  Arcane::IParallelMng::createTransferValuesOperation(),
  Arcane::IParallelMng::createExchanger(),
  Arcane::IParallelMng::createSynchronizer(),
  Arcane::IParallelMng::createTopology().
- Rend obsolète les accès à `Arccore::ArrayView<Array<T>>` dans
  Arcane::CaseOptionMultiSimpleT. Il faut utiliser la méthode
  Arcane::CaseOptionMultiSimpleT::view() à la place.

### Corrections:

- Ajoute une version 4 pour le calcul des couches fantômes qui permet
  d'appeler Arcane::IMeshModifier::updateGhostLayers() même s'il y a
  déjà une ou plusieurs couches de mailles fantomes.

### Interne:

- Nettoyage de la gestion des messages des synchronisations
- Débute support accélérateurs pour la version ROCM/HIP (AMD)
- Support pour la version 2.34 de la glibc qui ne contient plus les
  'hooks' de gestion mémoire (ce mécanisme était obsolète depuis des
  années).
- Ajoute possibilité de compiler avec le standard C++20.

Passage version 2.0.6.0 de %Arccore:
- Update Array views (\pr{76})
  - Add `constexpr` and `noexcept` to several methods of `Arccore::ArrayView`, `Arccore::ConstArrayView` and `Arccore::Span`
  - Add converters from `std::array`
- Separate metadata from data in 'Arccore::AbstractArray' (\pr{72})
- Deprecate `Arccore::Array::clone()`, `Arccore::Array2::clone()` and make `Arccore::Array2`
  constructors protected (\pr{71})
- Add support for compilation with AMD ROCM HIP (e5d008b1b79b59)
- Add method `Arccore::ITraceMng::fatalMessage()` to throw a
  `Arccore::FatalErrorException` in a method marked `[[noreturn]]`
- Add support to compile with C++20 with `ARCCORE_CXX_STANDARD`
  optional CMake variable (665292fce)
- [INTERNAL] Add support to change return type of `IMpiProfiling`
  methods. These methods should return `int` instead of `void`
- [INTERNAL] Add methods in `MpiAdapter` to send and receive messages without gathering statistics
- [INTERNAL] Add methods in `MpiAdapter` to disable checking of requests. These
  checks are disabled by default  if CMake variable
  `ARCCORE_BUILD_MODE` is `Release`





___
## Arcane Version 3.2.0 (15 novembre 2021) {#arcanedoc_news_changelog_version320}

### Nouveautés/Améliorations:

- Ajoute une interface Arcane::IMeshPartitionerBase pour faire
  uniquement le partitionnement sans support pour le
  repartionnement. L'interface Arcane::IMeshPartitioner hérite
  maintenant de Arcane::IMeshPartitionerBase.
- Ajoute dans Arcane::MeshReaderMng la possibilité de créer des
  maillages avec un Arcane::IParallelMng quelconque via la méthode
  Arcane::MeshReaderMng::readMesh().
- Ajoute une interface Arcane::IGridMeshPartitioner pour partitionner
  un maillage suivant une grille. Un service de nom
  `SimpleGridMeshPartitioner` implémente cette interface. La page
  \ref arcanedoc_entities_snippet_cartesianmesh montre un exemple d'utilisation.

### Changements:

- Passage version CMake 3.18 sur les machines Unix et CMake 3.21 sous Windows.
- Rend obsolète dans Arcane::ItemTypeMng la méthode `singleton()`. Les
  instances de cette classe sont attachées au maillage et peuvent être
  récupérées via Arcane::IMesh::itemTypeMng().
- Déplace les classes gérant le maillage cartésian dans le répertoire
  `arcane/cartesianmesh`. Les anciens chemins dans `arcane/cea`
  restent valides.
- Utilise par défaut la version 3 (au lieu de 2) du service de
  création des mailles fantômes. Cette version est plus efficace
  lorsqu'on utilise un grand nombre de sous-domaines car elle utilise
  des communications collectives.
- Supprime la préallocation mémoire pour les anciennes connectivités.
- Rend privé à %Arcane les constructeurs de Arcane::ItemSharedInfo
- Lance une exception fatale si on demande le support des tâches mais
  qu'aucune implémentation n'est disponible. Auparavant, il y avait
  juste un message d'avertissement.

### Corrections:

- Corrige plantage (SEGV) lorsqu'on utilise les tâches et sous-tâches
  en séquentiel.




___
## Arcane Version 3.1.2 (21 octobre 2021) {#arcanedoc_news_changelog_version310}

### Nouveautés/Améliorations:

- Nouvelle implémentation des graphes de maillage utilisant les
  `Arcane::DoF`.
- Ajoute possibilité de renuméroter (via la méthode
  Arcane::ICartesianMesh::renumberItemsUniqueId()) les entités dans les
  maillages AMR par patch pour avoir la même numérotation quel que
  soit le découpage.
- Mise à jour de la documentation pour les accélérateurs

### Changements:

- Le lecteur de maillage au format `GMSH` est maintenant dans la
  bibliothèque `arcane_std` au lieu de `arcane_ios`. Il n'y a donc
  plus besoin de faire l'édition de lien avec cette dernière pour
  pouvoir lire les maillages de ce format.
- Suppression des anciens types d'entités `Link` et `DualNode` et des
  énumérations et classes associées
- Suppression de certaines classes associées aux anciennes connectivités
- Supprime le support pour le système d'exploitation RedHat 6.

### Corrections:

- Corrige plantage lors de la mise à jour des matériaux si la variable
  globale associée à un matériau est désallouée
  (Arcane::IVariable::isUsed()==false)
- Corrige exception flottante (FPE) avec les versions 2.9.9+ de
  `libxml2`. Cette bibliothèque fait explicitement des divisions par 0
  lors de l'initialisation.




___
## Arcane Version 3.0.5 (30 septembre 2021) {#arcanedoc_news_changelog_version305}

### Nouveautés/Améliorations:

- Déplacement de la classe Arcane::NumArray dans la composante
  `utils`. Cela permet de la rendre accessible en dehors de son
  usage pour les accélérateurs
- Modifications diverses dans Arcane::NumArray et les classes
  associées (notamment Arcane::MDSpan) pour les rendre plus génériques
  et pour leur usage future dans les variables %Arcane.
- Simplifie et étend l'utilisation de Arcane::UnstructuredMeshConnectivityView
- Ajoute méthode 'IVariable::dataFactoryMng()' pour récupérer le
  Arcane::IDataFactoryMng àssocié aux données de la variable.
- Ajoute méthodes Arcane::Real2::normL2(), Arcane::Real3::normL2(),
  Arcane::Real2::squareNormL2() et Arcane::Real3::squareNormL3() pour
  remplacer les méthode 'abs()' et 'abs2()' de ces deux classes.
- Ajoute méthodes Arcane::Real2::absolute(), Arcane::Real3::absolute(),
  pour retourner un vecteur avec les valeurs absolues par composante.
- Ajoute support pour la version OneTBB 2021.
- Ajoute macros RUNCOMMAND_ENUMERATE() et RUNCOMMAND_LOOP() pour
  itérer sur les accélérateurs
- Ajoute classe Arcane::Accelerator::IAcceleratorMng pour récupérer
  les informations pour utiliser les accélérateurs. Cette interface
  permet de récupérer l'environnement d'exécution par défaut et la
  file d'exécution par défaut.
- Ajoute classe Arcane::StandaloneAcceleratorMng pour utiliser les
  accélérateurs sans initialiser une application.
- Ajoute support pour paralléliser en multi-thread les boucles
  imbriquées jusqu'à 4 niveaux (\pr{10})

### Changements:

- Rend obsolète la classe interne Arcane::IDataFactory et les méthodes
  correspondantes
- Rend obsolète les méthodes
  Arcane::IDataFactoryMng::createEmptySerializedDataRef() et
  Arcane::IDataFactoryMng::createSerializedDataRef()
- Supprime les méthodes obsolète Arcane::IData::clone() et
  Arcane::IData::cloneTrue().

### Corrections:

- Corrige l'opérateur de recopie lorsqu'on utilise deux vues du même
  type. Comme l'opérateur de recopie n'était pas surchargé, seule la
  référence était modifiée et pas la valeur. Cela se produisait dans
  le cas suivant:
```cpp
using namespace Arcane;
auto v1 = viewInOut(var1);
auto v2 = viewInOut(var2);
ENUMERATE_CELL(icell,allCells()){
  v2[icell] = v1[icell]; // ERREUR: v2 faisait ensuite référence à v1.
}
```
- Corrige erreur de compilation dans le constructeur Span<const T> à
  partir d'un 'ConstArrayView'.
- [arccore] Corrige envoi de message manquant lors de l'appel à
  Arccore::MessagePassing::PointToPointSerializerMng::waitMessages(). Il manquait
  l'appel à Arccore::MessagePassing::PointToPointSerializerMng::processPendingMessages().
  A cause de ce bug, la classe Arcane::TransferValuesParallelOperation
  ne fonctionnait pas et par conséquent la méthode
  Arcane::IItemFamily::reduceFromGhostItems() non plus.
- [config] Supporte le cas où plusieurs versions du SDK pour 'dotnet'
  sont installées. Dans ce cas la version la plus récente est utilisée.




___
## Arcane Version 3.0.3 (Not released) {#arcanedoc_news_changelog_version303}

### Nouveautés/Améliorations:

- Support de l'AMR par patch en parallèle
- Ajout d'une classe Arcane::SimpleSVGMeshExporter pour exporter au
  format SVG un ensemble de mailles
- Support dans l'AMR par patch dans la classe Arcane::DirNode des
  mailles voisines par direction.
- Lors de la synchronisation des groupes, s'assure que tous les
  sous-domaines ont les mêmes groupes et que la synchronisation se
  fait dans le même ordre.

### Changements:

- Rend obsolète les méthodes Arcane::IArrayDataT::value() et
  Arcane::IArray2DataT::value(). On peut à la place utiliser les
  méthodes Arcane::IArrayDataT::view() et
  Arcane::IArray2DataT::view(). Le but de ces changements est de
  pouvoir masquer le conteneur utilisé pour l'implémentation
- Ajoute méthodes Arcane::arcaneParallelFor() et
  Arcane::arcaneParallelForeach() pour remplacer les
  différentes méthodes Arcane::Parallel::For() et Arcane::Parallel::Foreach().

### Corrections:

- Dans l'AMR par patch, s'assure que les entités voisines par
  direction sont toujours dans le même niveau de patch.
- Corrige quelques dépendances manquantes lors de la compilation qui
  pouvaient entrainer des erreurs de compilation dans certains cas.
- Corrige erreurs de compilation des exemples en dehors du répertoire
  des sources.




___
## Arcane Version 3.0.1 (27 mai 2021) {#arcanedoc_news_changelog_version301}

Cette version est la première version 'open source' de %Arcane.

### Nouveautés/Améliorations:

- Nouvelle version du service de lecture/écriture
  Arcane::BasicReaderWriter pour générer moins de fichier et supporter
  la compression. Ce service peut être utilisé à la fois pour les
  protections/reprises et la comparaison bit à bit de
  variables. L'utilitaire C# de comparaison de variables a été mis à
  jour pour supporter cette nouvelle version.
- Support des fichiers de maillage au format 'msh' version 4.1. Cette
  version permet de spécifier des groupes de faces ou de mailles dans
  le fichier de maillage.
- En interne, utilise un seul exécutable pour l'ensemble des
  utilitaires C#.

### Changements:

- Ajoute possibilité lors de la compilation de %Arcane de spécifier les
  packages requis et de ne pas chercher de packages par défaut.

