# Nouvelles fonctionnalités {#arcanedoc_news_changelog}

[TOC]

Cette page contient les nouveautés de chaque version de %Arcane v3.X.X.

Les nouveautés successives apportées par les versions de %Arcane
antérieures à la version 3 sont listées ici : \ref arcanedoc_news_changelog20

___

## Arcane Version 3.8... (janvier 2023) {#arcanedoc_version380}

Work In Progress...

### Nouveautés/Améliorations:

- Ajoute support pour spécifier en ligne de commandes les valeurs
  par défaut de Arcane::ParallelLoopOptions en mode multi-thread
  (\issue{420}).
- Ajoute support des fichiers Lima, des fichiers MED et des fichiers
  au format `msh` avec les services de maillage (\issue{435}, \issue{439}, \issue{449}).
- Ajoute fonction Arcane::NumArrayUtils::readFromText() pour remplir une
  instance de Arcane::NumArray à partir d'un fichier au format ASCII (\issue{444}).
- Ajoute support de la lecture des en parallèle des fichiers au format
  MED (\issue{449}).
- Ajoute support pour la lecture des groupes de noeuds
  (Arcane::NodeGroup) dans les maillages au format MSH (\issue{475}).
- Ajoute support pour la renumérotation des maillages AMR en 3D. Cela
  permet d'avoir la même numérotation en quel que soit le découpage (\issue{495}).
- Ajoute accès à Arcane::IMeshMng dans Arcane::ICaseMng et
  Arcane::IPhysicalUnitSystem (\issue{461}).
- Ajoute support des accélérateurs pour les classes gérant le
  maillage cartésien (Arcane::CellDirectionMng,
  Arcane::FaceDirectionMng et Arcane::NodeDirectionMng) (\issue{474})
- Ajoute classe Arcane::impl::MutableItemBase pour remplacer
  l'utilisation de Arcane::ItemInternal (\issue{499}).
- Ajoute possibilité d'indexer les composantes de Arcane::Real2,
  Arcane::Real3, Arcane::Real2x2 et Arcane::Real3x3 par l'opérateur
  `operator()` (\issue{485}).
- Développements préliminaires pour les variables du maillage à
  plusieurs dimensions (\issue{459}, \issue{463}, \issue{464}, \issue{466}, \issue{471}).
- Ajoute interface Arcane::IDoFFamily pour gérer les
  Arcane::DoF. Auparavant il fallait utiliser directement
  l'implémentation Arcane::mesh::DoFFamily (\issue{480})
- Ajoute support dans `Aleph` des variables qui n'ont pas de familles
  par défaut (comme les Arcane::DoF par exemple) (\issue{468}).

### Changements:

- Utilise toujours une classe à la place d'un entier pour spécifier
  les dimensions (extents) dans les classes Arcane::NumArray et
  Arcane::MDSpan. Cela permet de se rapprocher de l'implémentation
  prévue dans le norme C++23 et d'avoir des dimensions statiques
  (connues à la compilation) (\issue{419}, \issue{425}, \issue{428}).
- Supprime les timers utilisant le temps CPU au lieu du temps écoulé. Le
  type Arcane::Timer::TimerVirtual existe toujours mais se comporte
  maintenant comme le type Arcane::Timer::TimerReal (\issue{421}).
- Supprime paramètre template avec le rang du tableau dans les classes
  Arcane::DefaultLayout, Arcane::RightLayout et Arcane::LefLayout
  (\issue{436}).
- Rend obsolète les méthodes de Arcane::ModuleBuildInfo qui utilisent
  Arcane::IMesh. Il faut utiliser les méthodes qui utilisent
  Arcane::MeshHandle. (\issue{460}).
- Change le type de retour de Arcane::IMeshBase::handle() pour ne pas
  retourner de référence mais une valeur (\issue{489}).
- Utilise des classes de base spécifiques par type de service lors de
  la génération des fichiers `axl` (\issue{472}).

### Corrections:

- Corrige inconsistence possible entre les connectivités conservées
  dans Arcane::ItemConnectivityList et
  Arcane::mesh::IncrementalItemConnectivity (\issue{478}).

### Interne:

- Supprime classes internes obsolètes Arcane::IAllocator,
  Arcane::DefaultAllocator, Arcane::DataVector1D,
  Arcane::DataVectorCommond1D et Arcane::Dictionary. Ces classes ne
  sont plus utilisées depuis longtemps (\issue{422}).
- Ajoute classe Arcane::TestLogger pour comparer les résultats des
  tests par rapport à un fichier listing de référence (\issue{418}).
- Ajoute possibilité de conserver les instances de
  Arcane::ItemSharedInfo' dans
  Arcane::ItemInternalConnectivityList'. Cela permettra de supprimer
  une indirection lors des accès aux connectivités. Cette option est
  pour l'instant uniquement utilisée en phase de test (\issue{371})
- Ajoute support pour l'appel Arccore::MessagePassing::mpLegacyProbe()
  pour les différents modes d'échange de message disponibles (\issue{431})
- Refactorisation des classes Arcane::NumArray, Arcane::MDSpan,
  Arcane::ArrayExtents et Arcane::ArrayBounds pour unifier le code
  et supporter des dimensions à la fois statiques et dynamiques. La
  page \ref arcanedoc_core_types_numarray explique l'utilisation de
  ces classes (\issue{426}, \issue{428}, \issue{433}, \issue{437}, \issue{440}).
- Utilise par défaut la Version 2 des synchronisations avec
  MPI. Cette version est la même que la version 1 utilisée auparavant
  mais sans le support des types dérivés (\issue{434}).
- [accelerator] Unifie le lancement des noyaux de calcul créés par les
  macros RUNCOMMAND_LOOP et RUNCOMMAND_ENUMERATE (\issue{438}).
- Unifie l'API de profiling entre les commandes
  (Arcane::Accelerator::RunCommand) et les énumérateurs classiques (via
  Arcane::IItemEnumeratorTracer). En fin de calcul l'affichage est trié
  par ordre décroissant du temps passé dans chaque boucle (\issue{442}, \issue{443}).
- Commence le développement des classes Arcane::NumVector and
  Arcane::NumMatrix pour généraliser les types Arcane::Real2,
  Arcane::Real3, Arcane::Real2x2 et Arcane::Real3x3. Ces classes sont
  pour l'instant à usage interne de %Arcane (\issue{441}).
- Diverses optimisations dans les classes internes gérant la
  connectivités et les itérateurs pour réduire leur taille (\issue{479}, \issue{482},
  \issue{483}, \issue{484})
- Supprime utilisation de Arcane::ItemInternalList dans
  Arcane::ItemVector et Arcane::ItemVectorView (\issue{486}, \issue{487}).
- Supprime utilisation de Arcane::ItemInternalVectorView (\issue{498})
- Supprime utilisation de Arcane::ItemInternal dans de nombreuses
  classes internes (\issue{488}, \issue{492}, \issue{500}, \issue{501})
- Supprime dans Arcane::MDSpan et Arcane::NumArray les indexeurs
  supplémentaires pour les classes Arcane::Real2, Arcane::Real3,
  Arcane::Real2x2 et Arcane::Real3x3. Ces indexeurs avaient été
  ajoutés à des fin de test mais n'étaient pas utilisés (\issue{490}).

### Arccore:

- Corrige bug si la méthode
  Arccore::MessagePassing::Mpi::MpiAdapter::waitSomeRequestsMPI() est
  appelée avec des requêtes déjà terminées (\issue{423}).
- Ajoute dans Arccore::Span et Arccore::Span un paramètre template
  indiquant le nombre d'éléments de la vue. Cela permettra de gérer
  des vues avec un nombre d'éléments connus à la compilation comme
  cela est possible avec `std::span` (\issue{424}).
- Ajoute fonctions Arccore::MessagePassing::mpLegacyProbe() dont la
  sémantique est similaire à `MPI_Iprobe` and `MPI_Probe` (\issue{430}).
- Corrige détection de la requête vide (\issue{427}, \issue{429}).

### Axlstar

- Add support for using a specific mesh in service instance (\issue{451})
- Remove support to build with `mono` (\issue{465}).

___

## Arcane Version 3.7.23 (17 novembre 2022) {#arcanedoc_version370}

### Nouveautés/Améliorations:

- Refonte complète de la documentation afin qu'elle soit plus
  cohérente et visuellement plus agréable (\issue{378}, \issue{380}, \issue{382}, \issue{384},
  \issue{388}, \issue{390}, \issue{393}, \issue{396})
- Ajoute un service de gestion de sorties au format CSV (voir
  \ref arcanedoc_services_modules_simplecsvoutput) (\issue{277}, \issue{362})
- Ajoute possibilité de spécifier le mot clé `Auto` pour la variable
  CMake `ARCANE_DEFAULT_PARTITIONER`. Cela permet de choisir
  automatiquement lors de la configuration le partitionneur utilisé
  en fonction de ceux disponibles (\issue{279}).
- Ajoute implémentation des synchronisations qui utilise la fonction
  `MPI_Neighbor_alltoallv` (\issue{281}).
- Réduction de l'empreinte mémoire utilisée pour la gestion des
  connectivités suite aux différentes modifications internes
- Optimisations lors de l'initialisation (\issue{302}):
  - Utilise `std::unordered_set` à la place de `std::set` pour les
    vérifications de duplication des uniqueId().
  - Lors de la création de maillage, ne vérifie la non-duplication des
    uniqueId() que en mode check.
- Crée une classe Arcane::ItemInfoListView pour remplacer à terme
  Arcane::ItemInternalList et accéder aux informations des entités à
  partir de leur localId() (\issue{305}).
- [accelerator] Ajoute le support des réductions Min/Max/Sum atomiques
  pour les types `Int32`, `Int64` et `double` (\issue{353}).
- [accelerator] Ajoute nouvel algorithme de réduction sans passer par
  des opérations atomiques. Cet algorithme n'est pas utilisé par
  défaut. Il faut l'activer en appelant
  Arcane::Accelerator::Runner::setDeviceReducePolicy() (\issue{365}, \issue{379})
- [accelerator] Ajoute possibilité de changer le nombre de threads par
  bloc lors du lancement d'une commande via
  Arcane::Accelerator::RunCommand::addNbThreadPerBlock() (\issue{374})
- [accelerator] Ajoute support pour le pre-chargement (prefetching) de
  conseils (advice) de zones mémoire (\issue{381})
- [accelerator] Ajoute support pour récupérer les informations sur les
  accélérateurs disponibles et associer un accélérateur à une instance
  de Arcane::Accelerator::Runner (\issue{399}).
- Début des développements pour pouvoir voir une variable tableau sur
  les entités comme une variable multi-dimensionnelle (\issue{335}).
- Ajoute un observable Arcane::MeshHandle::onDestroyObservable() pour
  pouvoir être notifié lors de la destruction d'une instance de
  maillage Arcane::IMesh (\issue{336}).
- Ajoute méthode Arcane::mesh_utils::dumpSynchronizerTopologyJSON()
  pour sauver au format JSON la topologie de communication pour les
  synchronisation (\issue{360}).
- Ajoute méthode Arcane::ICartesianMesh::refinePatch3D() pour raffiner
  un maillage 3D en plusieurs patchs AMR (\issue{386}).
- Ajoute implémentation de lecture de compteurs hardware via l'API
  perf de Linux (\issue{391}).
- Ajoute support pour le profiling automatiquement des commandes
  lancées via RUNCOMMAND_ENUMERATE (\issue{392}, \issue{394}, \issue{395})

### Changements:

- Modifie les classes associées à Arcane::NumArray
  (Arcane::MDSpan, Arcane::ArrayBounds, ...) pour que le paramètre
  template gérant le rang soit une classe et pas un entier. Le but à
  terme est d'avoir les mêmes paramètres templates que les classes
  `std::mdspan` et `std::mdarray` prévues pour les normes 2023 et 2026
  du C++. Il faut donc maintenant remplacer les dimensions en dur par
  les mots clés Arcane::MDDim1, Arcane::MDDim2, Arcane::MDDim3 ou
  Arcane::MDDim4 (\issue{333})
- La méthode Arcane::NumArray::resize() n'appelle plus le constructeur
  par défaut pour les éléments du tableau. C'était déjà le cas pour
  les types simples (Arcane::Real, Arcane::Real3, ...) mais maintenant
  c'est le cas aussi pour les types utilisateurs. Cela permet
  d'appeler cette méthode lorsque la mémoire est allouée sur l'accélérateur.
- Ajoute classe Arcane::ItemTypeId pour gérer le type de l'entité (\issue{294})
- Le type de l'entité est maintenant conservé sur un Arcane::Int16 au
  lieu d'un Arcane::Int32 (\issue{294})
- Supprime méthodes obsolètes de Arcane::ItemVector, `MathUtils.h`,
  Arcane::IApplication, Arcane::Properties, Arcane::IItemFamily
  (\issue{304}).
- Refonte des classes gérant l'énumération des entités (\issue{308}, \issue{364}, \issue{366}).
  - Supprime la classe de base Arcane::ItemEnumerator de
    Arcane::ItemEnumeratorT. L'héritage est remplacé par un opérateur
    de conversion.
  - Simplifie Arcane::ItemVectorViewConstIterator
  - Simplifie la gestion interne de l'opérateur `operator*` pour ne
    pas utiliser Arcane::ItemInternal.
- Refonte de la gestion du fichier de configuration
  `ArcaneConfig.cmake` géneré (\issue{318}):
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
  Arcane::ItemInternal (\issue{311})
- Crée une classe Arcane::ItemFlags pour gérer les flags concernant
  les propriétés des objets qui étaient avant dans
  Arcane::ItemInternal (\issue{312})
- Rend obsolète l'opérateur `operator->` pour la classe Arcane::Item
  et les classes dérivées (\issue{313})
- Change la valeur par défaut de la numérotation des faces dans le
  service de génération cartésien pour utiliser la numérotation
  cartésienne (\issue{315})
- Modification de la signature des méthodes de
  Arcane::IItemFamilyModifier et Arcane::mesh::OneMeshItemAdder
  pour utiliser Arcane::ItemTypeId au lieu de Arcane::ItemTypeInfo
  et Arcane::Item au lieu de Arcane::ItemInternal (\issue{322})
- Supprime méthodes Arcane::Item::activeFaces() et
  Arcane::Item::activeEdges() qui ne sont plus utilisées (\issue{351}).
- [C#] Ajoute la possibilité en fin de calcul de détruire les
  instances des différents gestionnaires comme lorsque le support de
  `.Net` n'est pas activé. Auparavant ces gestionnaires n'étaient
  jamais détruit pour éviter des plantages potentiels lorsque le
  'garbage collector' de l'environnement `.Net` se déclenche. Il est
  possible d'activer cette destruction en positionnant la variable
  d'environnement `ARCANE_DOTNET_USE_LEGACY_DESTROY` à la valeur
  `0`. Cela n'est pas actif par défaut car il peut rester des
  problèmes avec certains services utilisateurs (\issue{337}).
- [configuration] Il est maintenant nécessaire d'utiliser au moins la
  version 3.21 de CMake pour compiler ou utiliser #Arcane (\issue{367}).
- Ajoute constructeur par déplacement (`std::move`) pour
  Arcane::NumArray (\issue{372}).
- [accelerator] Supprime les méthodes obsolètes de création de
  Arcane::Accelerator::RunQueue et Arcane::Accelerator::Runner (\issue{397}).
- Rend obsolète la classe Arcane::AtomicInt32. Il faut utiliser
  la classe std::atomic<Int32> à la place (\issue{408}).

### Corrections:

- Corrige bug lors de la lecture des informations avec le service
  `BasicReaderwriter` lorsqu'une compression est active (\issue{299})
- Corrige bug introduit dans la version 3.6 qui changeait le nom du
  répertoire de sortie pour les comparaisons bit à bit avec le service
  `ArcaneCeaVerifier` (\issue{300}).
- Corrige mauvais recalcul du nombre maximum d'entités connectées à
  une entité dans le cas des particules (\issue{301})

### Interne:

- Simplifie l'implémentation des synchronisations fournissant un
  mécanisme indépendant du type de la donnée (\issue{282}).
- Utilise des variables pour gérer certaines données sur les entités
  telles que Arcane::Item::owner(), Arcane::Item::itemTypeId(). Cela
  permettra à terme de rendre ces informations accessibles sur
  accélérateurs (\issue{284}, \issue{285}, \issue{292}, \issue{295})
- Ajout d'une classe Arcane::ItemBase servant de classe de base pour
  Arcane::Item et Arcane::ItemInternal (\issue{298}, \issue{363}).
- Suppression d'une indirection lorsqu'on accède aux informations des
  connectivités à partir d'une entité (par exemple
  Arcane::Cell::node()) (\issue{298}).
- Simplification de la gestion des informations communes aux entités
  dans une famille pour qu'il n'y ait maintenant plus qu'une seule
  instance commune de Arcane::ItemSharedInfo (\issue{290}, \issue{292}, \issue{297}).
- Supprime certains usages de Arcane::ISubDomain (\issue{327})
  - Ajoute possibilité de créer une instance de Arcane::ServiceBuilder
    à partir d'un Arcane::MeshHandle.
  - Ajoute possibilité de créer une instance de
    Arcane::VariableBuildInfo via un Arcane::IVariableMng.
- Optimise les structures gérant le maillage cartésien pour ne plus
  avoir à conserver les instances de Arcane::ItemInternal*. Cela
  permet de réduire la consommation mémoire et potentiellement
  d'améliorer les performances (\issue{345}).
- Utilise des vues au lieu de Arccore::SharedArray pour les classes
  gérant les directions cartésiennes (Arcane::CellDirectionMng,
  Arcane::FaceDirectionMng et Arcane::NodeDirectionMng) (\issue{347}).
- Utilise un compteur de référence pour gérer
  Arccore::Ref<Arcane::ICaseFunction> (\issue{329}, \issue{356}).
- Ajoute constructeur pour la classe Arcane::Item et ses classes
  dérivées à partir d'un localId() et d'un Arcane::ItemSharedInfo (\issue{357}).
- Mise à jour des références des projets C# pour utiliser les
  dernières version des packages (\issue{359}).
- Nettoyage des classes Arcane::Real2, Arcane::Real3, Arcane::Real2x2
  et Arcane::Real3x3 et ajoute constructeurs à partir d'un
  Arcane::Real (\issue{370}, \issue{373}).
- Refonte partiel de la gestion de la concurrence pour mutualiser
  certaines fonctionnalités (\issue{389}).
- Utilise un Arccore::UniqueArray pour conteneur de
  Arcane::ListImplT. Auparavant le conteneur était un simple tableau C
  (\issue{407}).
- Dans Arcane::ItemGroupImpl, utilise Arcane::AutoRefT pour conserver
  des référence aux sous-groupes à la place d'un simple
  pointeur. Cela permet de garantir que les sous-groupes ne seront pas
  détruits tant que le parent associé existe.
- Corrige divers avertissements signalés par coverity (\issue{402}, \issue{403},
  \issue{405}, \issue{409}, \issue{410} )
- [C#] Indique qu'il faut au moins la version 8.0 du langage.

### Arccon:

Utilise la version 1.5.0:

- Add CMake functions to unify handling of packages  arccon Arccon
  componentbuildBuild configuration (\issue{342}).

### Arccore:

Utilise la version 2.0.12.0:

- Remove some coverity warnings (\issue{400})
- Use a reference counter for IMessagePassingMng (\issue{400})
- Fix method asBytes() with non-const types (\issue{400})
- Add a method in AbstractArray to resize without initializing (\issue{400})
- Make class ThreadPrivateStorage deprecated (\issue{400})

___
## Arcane Version 3.6.13 (06 juillet 2022) {#arcanedoc_news_changelog_version360}

### Nouveautés/Améliorations:

- Ajout d'une interface Arcane::IRandomNumberGenerator pour un service
  de génération de nombre aléatoire (\issue{266})
- Ajoute support des variables matériaux dans les fichiers `axl` pour
  le générateur C# (\issue{273})
- Supprime allocation de la connectivité des noeuds dans les anciennes
  connectivités. Cela permet de réduire l'empreinte mémoire (\issue{231}).
- Ajoute pour les classes Arccore::Span, Arccore::ArrayView,
  Arccore::ConstArrayView ainsi que les vues sur les variable
  l'opérateur 'operator()' qui se comporte comme l'opérateur
  'operator[]'. Cela permet d'uniformiser les écritures entre les
  différents conteneurs et les vues associées (\issue{223}, \issue{222}, \issue{205}).
- Ajoute dans Arcane::ICartesianMeshGenerationInfo les informations
  sur l'origine et la dimension du maillage cartésien (\issue{221}).
- Ajoute en fin d'exécution des statistiques collectives sur les temps
  passés dans les opérations d'échange de message. Ces statistiques
  comprennent le temps minimal, maximal et moyen pour l'ensemble des
  rangs passés dans ces appels (\issue{220})
- Ajoute deux implémentations supplémentaires pour la synchronisation
  des matériaux. La version 7 permet de faire une seule allocation
  lors de cette synchronisation et la version 8 permet de conserver
  cette allocation d'une synchronisation à l'autre (\issue{219}).
- Ajoute implémentation des synchronisations en plusieurs phases
  permettant d'utiliser des tableaux de taille fixe et/ou de faire sur
  une sous-partie des voisins (\issue{214}).
- Ajoute accès pour les accélérateurs à certaines méthodes de
  Arcane::MDSpan (\issue{217}).
- Ajoute accès aux connectivités aux arêtes dans
  Arcane::UnstructuredMeshConnectivityView (\issue{216})
- Ajoute interface accessible via
  'Arcane::IMesh::indexedConnectivityMng()' permet de facilement
  ajouter de nouvelles connectivités (\issue{201}).
- Ajout d'un nouvel algorithme de calcul des uniqueId() des arêtes
  (Edge) pour les maillages cartésiens
- Ajoute support pour les classes de Arccore de l'opérateur
  `operator[]` avec arguments multiples (\issue{241}).
- Possibilité de rendre thread-safe les appels à
  Arcane::Accelerator::makeQueue() en appelant la méthode
  Arcane::Accelerator::Runner::setConcurrentQueueCreation() (\issue{242})

### Changements:

- Scinde en deux composantes les classes gérant les matériaux. Une
  partie est maintenant dans la composante `arcane_core`. Ce
  changement est normalement transparent pour les utilisateurs
  d'%Arcane et il n'y a pas besoin de modifier les sources (\issue{264},\issue{270},\issue{274})
- Compacte les références après un appel à
  Arcane::IItemFamily::compactItems(). Cela permet d'éviter de faire
  grossir inutilement le tableau des contenant les informations
  internes des entités. Comme ce changement peut induire une
  différence sur l'ordre de certaines opérations, il est possible de
  le désactiver en positionnant la variable d'environnement
  `ARCANE_USE_LEGACY_COMPACT_ITEMS` à la valeur `1` (\issue{225}).
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
  gérés par un 'Garbage Collector' (\issue{200}).
- Rend obsolète l'utilisation de Arcane::Timer::TimerVirtual. Les timers qui
  utilisent cette propriété se comportent comme s'ils avaient
  l'attribut Arcane::Timer::TimerReal.

### Corrections:

- Corrige mauvaises valeurs de Arcane::IItemFamily::localConnectivityInfos()
  et Arcane::IItemFamily::globalConnectivityInfos() pour les connectivités
  autres que celles aux noeuds. Ce bug avait été introduit lors du
  passage aux nouvelles connectivités (\issue{230}, \issue{27}).
- Corrige divers bugs dans la version 3 de BasicReaderWriter (\issue{238})

### Interne:

- Utilise des variables pour conserver les champs Arcane::ItemInternal::owner() et
  Arcane::ItemInternal::flags() au lieu de conserver l'information dans
  Arcane::ItemSharedInfo. Cela permettra à terme de supprimer le champ
  correspondant dans Arcane::ItemSharedInfo (\issue{227}).

Passage version 2.0.3.0 de Axlstar:

  - Ajoute support dans les fichiers 'axl' des propriétés Arcane::IVariable::PNoExchange,
    Arcane::IVariable::PNoReplicaSync et Arcane::IVariable::PPersistant.

Passage version 2.0.11.0 de %Arccore:

  - Add function `mpDelete()` to destroy `IMessagePassingMng` instances (\issue{258})
  - Optimizations in class `String`(\issue{256},\issue{247})
      - Add move constructor String(String&&) and move copy operator operator=(String&&)
      - Make `String` destructor inline 
      - Make method `String::utf16()` deprecated (replaced by `StringUtils::asUtf16BE()`)
      - Methods `String::bytes()` and `String::format` no longer throws exceptions
      - Add a namespace `StringUtils` to contains utilitarian functions.
  - Add support for multisubscript `operator[]` from C++23 (\issue{241})
  - Add `operator()` to access values of `ArrayView`, `ArrayView2`,
    `ArrayView3`, `ArrayView4`, `Span`, `Span2` and `const` versions
    of these views (\issue{223}).
  - Add `SmallSpan2` implementation for 2D arrays whose `size_type` is an `Int32` (\issue{223}).
  - Add `SpanImpl::findFirst()` method  (\issue{211})
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
  `ARCANE_MESH_EXCHANGE_USE_COLLECTIVE` (\issue{138},\issue{154}).
- Dans la comparaison bit à bit, ajoute possibilité de ne faire la
  comparaison qu'à la fin de l'exécution au lieu de le faire à chaque
  pas de temps. Cela se fait en spécifiant la variable d'environnement
  STDENV_VERIF_ONLY_AT_EXIT.
- Ajoute générateur de maillages en nid d'abeille en 3D (\issue{149}).
- Ajoute support pour spécifier la disposition des éléments (layout)
  dans la classe Arcane::NumArray. Il existe actuellement deux
  dispositions implémentées: LeftLayout et RightLayout (\issue{151})
- Ajoute méthode Arcane::Accelerator::RunQueue::copyMemory() pour faire des
  copies mémoire asynchrones (\issue{152}).
- Améliore le support ROCM/HIP. Le support des GPU AMD est maintenant
  fonctionnellement équivalent à celui des GPU NVIDIA via Cuda (\issue{158}, \issue{159}).
- Ajoute support pour la mémoire punaisée (Host Pinned Memory) pour
  CUDA et ROCM (\issue{147}).
- Ajoute classe 'Arcane::Accelerator::RunQueueEvent' pour supporter
  les évènements sur les 'Arcane::Accelerator::RunQueue' et permettre
  ainsi de synchroniser entre elle des files différentes (\issue{161}).

### Changements:

- Supprime les macros plus utilisées ARCANE_PROXY et ARCANE_TRACE (\issue{145})

### Corrections:

- Corrige mauvaise détection de la version OneTBB 2021.5 suite à la
  suppression du fichier 'tbb_thread.h' (\issue{146})
- Corrige certaines informations cartésiennes manquantes lorsqu'il n'y
  a qu'une seule couche de maille en Y ou Z (\issue{162}).
- Corrige implémentation manquante de 'Arccore::Span<T>::operator=='
  lorsque le type `T` n'est pas constant (\issue{163}).
- Supprime quelques messages listings trop nombreux.

### Interne:

- Utilise au lieu de Arccore::UniqueArray2 une implémentation
  spécifique pour le conteneur de Arcane::NumArray (\issue{150}).
- Utilise des `Int32` au lieu de `Int64` pour indexer les éléments
  dans Arcane::NumArray (\issue{153})
- Ajoute `constexpr` et `noexcept` à certaines classes de Arccore
  (\issue{156}).
- Passage version 2.0.9.0 de Arccore




___
## Arcane Version 3.4.5 (10 février 2022) {#arcanedoc_news_changelog_version340}

### Nouveautés/Améliorations:

- Dans l'API accélérateur, support dans Arcane::NumArray pour allouer
  directement la mémoire sur l'accélérateur. Auparavant seule la
  mémoire unifiée était disponible. L'énumération
  Arcane::eMemoryRessource et le type Arcane::IMemoryRessourceMng
  permettent de gérer cela (\issue{111}, \issue{113}).
- Amélioration mineures sur la documentation (\issue{117}) :
  - ajout des chemins relatifs pour les fichiers d'en-tête.
  - ajout des classes et type issus de %Arccore
- Ajoute nouvelle méthode de calcul des uniqueId() des faces
  dans le cas cartésien. Cette nouvelle méthode permet une
  numérotation cartésienne des faces qui est cohérente avec celles des
  noeuds et des mailles. Pour l'utiliser, il faut spécifier l'option
  `<face-numbering-version>4</face-numbering-version>` dans le jeu de
  données dans la balise du générateur de maillage (\issue{104}).
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
'Arcane::arcane_core'. Les anciens noms restent valides (\issue{120}).
- Rend obsolète la conversion de Arcane::ItemEnumerator vers
  Arcane::ItemEnumeratorT. Cela permet d'éviter d'indexer par erreur
  une variable du maillage avec un énumérateur du mauvais type (par
  exemple indexer une variable aux mailles avec un énumérateur aux noeuds).

### Corrections:

- Corrige opérateur 'operator=' pour la classe
  'Arcane::CellDirectionMng' (\issue{109})
- Corrige conversion inutile `Int64` vers
  `Int32` dans la construction des maillages cartésiens ce qui
  empêchait de dépasser 2^31 mailles (\issue{98})
- Corrige mauvais calcul des temps passés dans les
  synchronisations. Seul le temps de la dernière attente était utilisé
  au lieu du cumul (commit cf2cade961)
- Corrige nom prise en compte de l'option 'MessagePassingService' en
  ligne de commande (commit 15670db4)

### Interne:

- Nettoyage de l'API Accélérateur

Passage version 2.0.8.1 de %Arccore:

  - Improve doxygen documentation for types et classes in `message_passing` component.
  - Add functions in `message_passing` component to handle non blocking collectives (\issue{116}, \issue{118})
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
- Update Array views (\issue{76})
  - Add `constexpr` and `noexcept` to several methods of `Arccore::ArrayView`, `Arccore::ConstArrayView` and `Arccore::Span`
  - Add converters from `std::array`
- Separate metadata from data in 'Arccore::AbstractArray' (\issue{72})
- Deprecate `Arccore::Array::clone()`, `Arccore::Array2::clone()` and make `Arccore::Array2`
  constructors protected (\issue{71})
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
  imbriquées jusqu'à 4 niveaux (PR #10)

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

