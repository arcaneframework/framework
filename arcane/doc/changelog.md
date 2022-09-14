Nouvelles fonctionnalités {#arcanedoc_changelog}
=====================================

[TOC]

Cette page contient les nouveautés de chaque version de %Arcane.

Arcane Version 3.7.x (... septembre 2022) {#arcanedoc_version370}
======================================

WIP

Nouveautés/Améliorations:

- Ajoute un service de gestion de sorties au format CSV (voir
  \ref arcanedoc_services_csv) (#277)
- Ajoute possibilité de spécifier le mot clé `Auto` pour la variable
  CMake `ARCANE_DEFAULT_PARTITIONER`. Cela permet de choisir
  automatiquement lors de la configuration le partitionneur utilisé
  en fonction de ceux disponibles (#279).
- Ajoute implémentation des synchronisations qui utilise la fonction
  `MPI_Neighbor_alltoallv` (#281).
- Réduction de l'empreinte mémoire utilisée pour la gestion des
  connectivités suite aux différentes modifications internes
- Optimisations lors de l'initialisation (#302):
  - Utilise `std::unordered_set` à la place de `std::set` pour les
    vérifications de duplication des uniqueId().
  - Lors de la création de maillage, ne Vérifie la non-duplication des
    uniqueId() que en mode check.
- Crée une classe Arcane::ItemInfoListView pour remplacer à terme
  Arcane::ItemInternalList et accéder aux informations des entités à
  partir de leur localId() (#305)

Changements:

- Modifie les classes associées à Arcane::NumArray
  (Arcane::MDSpan, Arcane::ArrayBounds, ...) pour que le paramètre
  template gérant le rang soit une classe et pas un entier. Le but à
  terme est d'avoir les mêmes paramètres templates que les classes
  std::mdspan et std::mdarray du C++. Il faut remplacer les dimensions
  en dur par les mots clés Arcane::MDDim1, Arcane::MDDim2,
  Arcane::MDDim3 ou Arcane::MDDim4 (#333)
- Ajoute classe Arcane:ItemTypeId pour gérer le type de l'entité (#294)
- Le type de l'entité est maintenant conservé sur un Arcane::Int16 au
  lieu d'un Arcane::Int32 (#294)
- Supprime méthodes obsolètes de 'Concurrency.h'. Il s'agit des méthodes
  qui se trouvaient dans le namespace 'Arcane::Parallel' et qui sont
  maintenant dans le namespace 'Arcane' mais préfixée par
  `arcaneParallel` (#303)
- Supprime méthodes obsolètes de Arcane::ItemVector, `MathUtils.h`,
  Arcane::IApplication, Arcane::Properties, Arcane::IItemFamily
  (#304).
- Supprime la classe de base Arcane::ItemEnumerator de
  Arcane::ItemEnumeratorT. L'héritage est remplacé par un opérateur de
  conversion (#308).
- Refonte de la gestion du fichier de configuration
  `ArcaneConfig.cmake` géneré (#318):
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
  Arcane::ItemInternal (#311)
- Crée une classe Arcane::ItemFlags pour gérer les flags concernant
  les propriétés des objets qui étaient avant dans
  Arcane::ItemInternal (#312)
- Rend obsolète l'opérateur `operator->` pour la classe Arcane::Item
  et les classes dérivées (#313)
- Change la valeur par défaut de la numérotation des faces dans le
  service de génération cartésien pour utiliser la numérotation
  cartésienne (#315)
- Modification de la signature des méthodes de
  Arcane::IItemFamilyModifierInterface et Arcane::OneMeshItemAdder
  pour utiliser Arcane::ItemTypeId au lieu de Arcane::ItemTypeInfo
  et Arcane::Item au lieu de Arcane::ItemInternal (#322)

Corrections:

- Corrige bug lors de la lecture des informations avec le service
  `BasicReaderwriter` lorsqu'une compression est active (#299)
- Corrige bug introduit dans la version 3.6 qui changeait le nom du
  répertoire de sortie pour les comparaisons bit à bit avec le service
  `ArcaneCeaVerifier` (#300).
- Corrige mauvais recalcul du nombre maximum d'entités connectées à
  une entité dans le cas des particules (#301)

Interne:

- Simplifie l'implémentation des synchronisations fournissant un
  mécanisme indépendant du type de la donnée (#282).
- Utilise des variables pour gérer certaines données sur les entités
  telles que Arcane::Item::owner(), Arcane::Item::typeId(). Cela
  permettra à terme de rendre ces informations accessibles sur
  accélérateurs (#284, #285, #292, #295)
- Ajout d'une classe Arcane::ItemBase servant de classe de base pour
  Arcane::Item et Arcane::ItemInternal (#298).
- Suppression d'une indirection lorsqu'on accède aux informations des
  connectivités à partir d'une entité (par exemple
  Arcane::Cell::node()) (#298).
- Simplification de la gestion des informations communes aux entités
  dans une famille pour qu'il n'y ait maintenant plus qu'une seule
  instance commune de Arcane::ItemSharedInfo (#290, #292, #297).
- Supprime certains usages de Arcane::ISubDomain (#327)
  - Ajoute possibilité de créer une instance de Arcane::ServiceBuilder
    à partir d'un Arcane::MeshHandle.
  - Ajoute possibilité de créer une instance de
    Arcane::VariableBuildInfo via un Arcane::IVariableMng.

Arcane Version 3.6.13 (06 juillet 2022) {#arcanedoc_version360}
======================================

Nouveautés/Améliorations:

- Ajout d'une interface Arcane::IRandomNumberGenerator pour un service
  de génération de nombre aléatoire (#266)
- Ajoute support des variables matériaux dans les fichiers `axl` pour
  le générateur C# (#273)
- Supprime allocation de la connectivité des noeuds dans les anciennes
  connectivités. Cela permet de réduire l'empreinte mémoire (#231).
- Ajoute pour les classes Arccore::Span, Arccore::ArrayView,
  Arccore::ConstArrayView ainsi que les vues sur les variable
  l'opérateur 'operator()' qui se comporte comme l'opérateur
  'operator[]'. Cela permet d'uniformiser les écritures entre les
  différents conteneurs et les vues associées (#223, #222, #205).
- Ajoute dans Arcane::ICartesianMeshGenerationInfo les informations
  sur l'origine et la dimension du maillage cartésien (#221).
- Ajoute en fin d'exécution des statistiques collectives sur les temps
  passés dans les opérations d'échange de message. Ces statistiques
  comprennent le temps minimal, maximal et moyen pour l'ensemble des
  rangs passés dans ces appels (#220)
- Ajoute deux implémentations supplémentaires pour la synchronisation
  des matériaux. La version 7 permet de faire une seule allocation
  lors de cette synchronisation et la version 8 permet de conserver
  cette allocation d'une synchronisation à l'autre (#219).
- Ajoute implémentation des synchronisations en plusieurs phases
  permettant d'utiliser des tableaux de taille fixe et/ou de faire sur
  une sous-partie des voisins (#214).
- Ajoute accès pour les accélérateurs à certaines méthodes de
  Arcane::MDSpan (#217).
- Ajoute accès aux connectivités aux arêtes dans
  Arcane::UnstructuredMeshConnectivityView (#216)
- Ajoute interface accessible via
  'Arcane::IMesh::indexedConnectivityMng()' permet de facilement
  ajouter de nouvelles connectivités (#201).
- Ajout d'un nouvel algorithme de calcul des uniqueId() des arêtes
  (Edge) pour les maillages cartésiens
- Ajoute support pour les classes de Arccore de l'opérateur
  `operator[]` avec arguments multiples (#241).
- Possibilité de rendre thread-safe les appels à
  Arcane::Accelerator::makeQueue() en appelant la méthode
  Arcane::Accelerator::Runner::setConcurrentQueueCreation() (#242)

Changements:

- Scinde en deux composantes les classes gérant les matériaux. Une
  partie est maintenant dans la composante `arcane_core`. Ce
  changement est normalement transparent pour les utilisateurs
  d'%Arcane et il n'y a pas besoin de modifier les sources (#264,#270,#274)
- Compacte les références après un appel à
  Arcane::IItemFamily::compactItems(). Cela permet d'éviter de faire
  grossir inutilement le tableau des contenant les informations
  internes des entités. Comme ce changement peut induire une
  différence sur l'ordre de certaines opérations, il est possible de
  le désactiver en positionnant la variable d'environnement
  `ARCANE_USE_LEGACY_COMPACT_ITEMS` à la valeur `1` (#225).
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
  gérés par un 'Garbage Collector' (#200).
- Rend obsolète l'utilisation de Arcane::Timer::TimerVirtual. Les timers qui
  utilisent cette propriété se comportent comme s'ils avaient
  l'attribut Arcane::Timer::TimerReal.

Corrections:

- Corrige mauvaises valeurs de Arcane::IItemFamily::localConnectivityInfos()
  et Arcane::IItemFamily::globalConnectivityInfos() pour les connectivités
  autres que celles aux noeuds. Ce bug avait été introduit lors du
  passage aux nouvelles connectivités (#230, #27).
- Corrige divers bugs dans la version 3 de BasicReaderWriter (#238)

Interne:

- Utilise des variables pour conserver les champs Arcane::ItemInternal::owner() et
  Arcane::ItemInternal::flags() au lieu de conserver l'information dans
  Arcane::ItemSharedInfo. Cela permettra à terme de supprimer le champ
  correspondant dans Arcane::ItemSharedInfo (#227).

- Passage version 2.0.3.0 de Axlstar:

  - Ajoute support dans les fichiers 'axl' des propriétés Arcane::IVariable::PNoExchange,
    Arcane::IVariable::PNoReplicaSync et Arcane::IVariable::PPersistant.

- Passage version 2.0.11.0 de %Arccore:

  - Add function `mpDelete()` to destroy `IMessagePassingMng` instances (#258)
  - Optimizations in class `String`(#256,#247)
      - Add move constructor String(String&&) and move copy operator operator=(String&&)
      - Make `String` destructor inline 
      - Make method `String::utf16()` deprecated (replaced by `StringUtils::asUtf16BE()`)
      - Methods `String::bytes()` and `String::format` no longer throws exceptions
      - Add a namespace `StringUtils` to contains utilitarian functions.
  - Add support for multisubscript `operator[]` from C++23 (#241)
  - Add `operator()` to access values of `ArrayView`, `ArrayView2`,
    `ArrayView3`, `ArrayView4`, `Span`, `Span2` and `const` versions
    of these views (#223).
  - Add `SmallSpan2` implementation for 2D arrays whose `size_type` is an `Int32` (#223).
  - Add `SpanImpl::findFirst()` method  (#211)
  - Fix build on Ubuntu 22.04

    
Arcane Version 3.5.7 (07 avril 2022) {#arcanedoc_version350}
======================================

Nouveautés/Améliorations:

- Ajoute classe Arcane::SimdReal3x3 et Arcane::SimdReal2x2 qui sont
  l'équivalent vectorielle des Arcane::Real3x3 et Arcane::Real2x2
- Support de la version 4.2 des fichiers de maillage au format VTK
- Ajoute une nouvelle implémentation des synchronisations qui utilise
  l'appel `MPI_Sendrecv`.
- Ajoute possibilité d'utiliser des messages collectifs
  (MPI_AllToAllv) au lieu de messages point à point lors de l´échange
  des entités suite à un équilibrage de charge. Ce mécanisme est
  temporairement accessible en spécifiant la variable d'environnement
  `ARCANE_MESH_EXCHANGE_USE_COLLECTIVE` (#138,#154).
- Dans la comparaison bit à bit, ajoute possibilité de ne faire la
  comparaison qu'à la fin de l'exécution au lieu de le faire à chaque
  pas de temps. Cela se fait en spécifiant la variable d'environnement
  STDENV_VERIF_ONLY_AT_EXIT.
- Ajoute générateur de maillages en nid d'abeille en 3D (#149).
- Ajoute support pour spécifier la disposition des éléments (layout)
  dans la classe Arcane::NumArray. Il existe actuellement deux
  dispositions implémentées: LeftLayout et RightLayout (#151)
- Ajoute méthode Arcane::Accelerator::RunQueue::copyMemory() pour faire des
  copies mémoire asynchrones (#152).
- Améliore le support ROCM/HIP. Le support des GPU AMD est maintenant
  fonctionnellement équivalent à celui des GPU NVIDIA via Cuda (#158, #159).
- Ajoute support pour la mémoire punaisée (Host Pinned Memory) pour
  CUDA et ROCM (#147).
- Ajoute classe 'Arcane::Accelerator::RunQueueEvent' pour supporter
  les évènements sur les 'Arcane::Accelerator::RunQueue' et permettre
  ainsi de synchroniser entre elle des files différentes (#161).

Changements:

- Supprime les macros plus utilisées ARCANE_PROXY et ARCANE_TRACE (#145)

Corrections:

- Corrige mauvaise détection de la version OneTBB 2021.5 suite à la
  suppression du fichier 'tbb_thread.h' (#146)
- Corrige certaines informations cartésiennes manquantes lorsqu'il n'y
  a qu'une seule couche de maille en Y ou Z (#162).
- Corrige implémentation manquante de 'Arccore::Span<T>::operator=='
  lorsque le type `T` n'est pas constant (#163).
- Supprime quelques messages listings trop nombreux.

Interne:

- Utilise au lieu de Arccore::UniqueArray2 une implémentation
  spécifique pour le conteneur de Arcane::NumArray (#150).
- Utilise des `Int32` au lieu de `Int64` pour indexer les éléments
  dans Arcane::NumArray (#153)
- Ajoute `constexpr` et `noexcept` à certaines classes de Arccore
  (#156).
- Passage version 2.0.9.0 de Arccore

Arcane Version 3.4.5 (10 février 2022) {#arcanedoc_version340}
======================================

Nouveautés/Améliorations:

- Dans l'API accélérateur, support dans Arcane::NumArray pour allouer
  directement la mémoire sur l'accélérateur. Auparavant seule la
  mémoire unifiée était disponible. L'énumération
  Arcane::eMemoryRessource et le type Arcane::IMemoryRessourceMng
  permettent de gérer cela (#111, #113).
- Amélioration mineures sur la documentation (#117) :
  - ajout des chemins relatifs pour les fichiers d'en-tête.
  - ajout des classes et type issus de %Arccore
- Ajoute nouvelle méthode de calcul des uniqueId() des faces
  dans le cas cartésien. Cette nouvelle méthode permet une
  numérotation cartésienne des faces qui est cohérente avec celles des
  noeuds et des mailles. Pour l'utiliser, il faut spécifier l'option
  `<face-numbering-version>4</face-numbering-version>` dans le jeu de
  données dans la balise du générateur de maillage (#104).
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

Changements:

- Ajoute namespace '%Arcane::' aux cibles `CMake` fournit par
%Arcane. Par exemple, la cible 'arcane_core' devient
'Arcane::arcane_core'. Les anciens noms restent valides (#120).
- Rend obsolète la conversion de Arcane::ItemEnumerator vers
  Arcane::ItemEnumeratorT. Cela permet d'éviter d'indexer par erreur
  une variable du maillage avec un énumérateur du mauvais type (par
  exemple indexer une variable aux mailles avec un énumérateur aux noeuds).

Corrections:

- Corrige opérateur 'operator=' pour la classe
  'Arcane::CellDirectionMng' (#109)
- Corrige conversion inutile `Int64` vers
  `Int32` dans la construction des maillages cartésiens ce qui
  empêchait de dépasser 2^31 mailles (#98)
- Corrige mauvais calcul des temps passés dans les
  synchronisations. Seul le temps de la dernière attente était utilisé
  au lieu du cumul (commit cf2cade961)
- Corrige nom prise en compte de l'option 'MessagePassingService' en
  ligne de commande (commit 15670db4)

Interne:

- Nettoyage de l'API Accélérateur

Passage version 2.0.8.1 de %Arccore:

- Improve doxygen documentation for types et classes in `message_passing` component.
- Add functions in `message_passing` component to handle non blocking collectives (#116, #118)
- Add some '#defines'  to compile with [hipSYCL](https://github.com/illuhad/hipSYCL).
- Update '_clang-format' file for version 13 of LLVM/Clang.


Arcane Version 3.3.0 (16 décembre 2021) {#arcanedoc_version330}
======================================

Nouveautés/Améliorations:

- Ajoute possibilité de spécifier le nombre maximum de messages en vol
  lors d'un équilibrage de charge. Pour l'instant cela se fait en
  spécifiant la variable d'environnement
  ARCANE_MESH_EXCHANGE_MAX_PENDING_MESSAGE.
- Ajoute possibilité d'utiliser les Arcane::Real2x2 et Arcane::Real3x3 sur accélérateurs
- Ajoute méthode Arcane::mesh_utils::printMeshGroupsMemoryUsage() pour
  afficher la consommation mémoire associée aux groupes et
  Arcane::mesh_utils::shrinkMeshGroups() pour redimensionner au plus
  juste la mémoire utilisée par les groupes
- Support pour punaiser les threads (voir \ref arcanedoc_launcher)

Changements:

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

Corrections:

- Ajoute une version 4 pour le calcul des couches fantômes qui permet
  d'appeler Arcane::IMeshModifier::updateGhostLayers() même s'il y a
  déjà une ou plusieurs couches de mailles fantomes.

Interne:

- Nettoyage de la gestion des messages des synchronisations
- Débute support accélérateurs pour la version ROCM/HIP (AMD)
- Support pour la version 2.34 de la glibc qui ne contient plus les
  'hooks' de gestion mémoire (ce mécanisme était obsolète depuis des
  années).
- Ajoute possibilité de compiler avec le standard C++20.

Passage version 2.0.6.0 de %Arccore:
- Update Array views (#76)
  - Add `constexpr` and `noexcept` to several methods of `Arccore::ArrayView`, `Arccore::ConstArrayView` and `Arccore::Span`
  - Add converters from `std::array`
- Separate metadata from data in 'Arccore::AbstractArray' (#72)
- Deprecate `Arccore::Array::clone()`, `Arccore::Array2::clone()` and make `Arccore::Array2`
  constructors protected (#71)
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

Arcane Version 3.2.0 (15 novembre 2021) {#arcanedoc_version320}
======================================

Nouveautés/Améliorations:

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
  \ref arcanesnippet_imeshgridpartitioner1 montre un exemple d'utilisation.

Changements:

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

Corrections:

- Corrige plantage (SEGV) lorsqu'on utilise les tâches et sous-tâches
  en séquentiel.

Arcane Version 3.1.2 (21 octobre 2021) {#arcanedoc_version310}
======================================

Nouveautés/Améliorations:

- Nouvelle implémentation des graphes de maillage utilisant les
  `Arcane::DoF`.
- Ajoute possibilité de renuméroter (via la méthode
  Arcane::ICartesianMesh::renumberItemsUniqueId()) les entités dans les
  maillages AMR par patch pour avoir la même numérotation quel que
  soit le découpage.
- Mise à jour de la documentation pour les accélérateurs

Changements:

- Le lecteur de maillage au format `GMSH` est maintenant dans la
  bibliothèque `arcane_std` au lieu de `arcane_ios`. Il n'y a donc
  plus besoin de faire l'édition de lien avec cette dernière pour
  pouvoir lire les maillages de ce format.
- Suppression des anciens types d'entités `Link` et `DualNode` et des
  énumérations et classes associées
- Suppression de certaines classes associées aux anciennes connectivités
- Supprime le support pour le système d'exploitation RedHat 6.

Corrections:

- Corrige plantage lors de la mise à jour des matériaux si la variable
  globale associée à un matériau est désallouée
  (Arcane::IVariable::isUsed()==false)
- Corrige exception flottante (FPE) avec les versions 2.9.9+ de
  `libxml2`. Cette bibliothèque fait explicitement des divisions par 0
  lors de l'initialisation.

Arcane Version 3.0.5 (30 septembre 2021) {#arcanedoc_version305}
======================================

Nouveautés/Améliorations:

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

Changements:

- Rend obsolète la classe interne Arcane::IDataFactory et les méthodes
  correspondantes
- Rend obsolète les méthodes
  Arcane::IDataFactoryMng::createEmptySerializedDataRef() et
  Arcane::IDataFactoryMng::createSerializedDataRef()
- Supprime les méthodes obsolète Arcane::IData::clone() et
  Arcane::IData::cloneTrue().

Corrections:

- Corrige l'opérateur de recopie lorsqu'on utilise deux vues du même
  type. Comme l'opérateur de recopie n'était pas surchargé, seule la
  référence était modifiée et pas la valeur. Cela se produisait dans
  le cas suivant:
~~~{.cpp}
using namespace Arcane;
auto v1 = viewInOut(var1);
auto v2 = viewInOut(var2);
ENUMERATE_CELL(icell,allCells()){
  v2[icell] = v1[icell]; // ERREUR: v2 faisait ensuite référence à v1.
}
~~~
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

Arcane Version 3.0.3 (Not released) {#arcanedoc_version303}
======================================

Nouveautés/Améliorations:

- Support de l'AMR par patch en parallèle
- Ajout d'une classe Arcane::SimpleSVGMeshExporter pour exporter au
  format SVG un ensemble de mailles
- Support dans l'AMR par patch dans la classe Arcane::DirNode des
  mailles voisines par direction.
- Lors de la synchronisation des groupes, s'assure que tous les
  sous-domaines ont les mêmes groupes et que la synchronisation se
  fait dans le même ordre.

Changements:

- Rend obsolète les méthodes Arcane::IArrayDataT::value() et
  Arcane::IArray2DataT::value(). On peut à la place utiliser les
  méthodes Arcane::IArrayDataT::view() et
  Arcane::IArray2DataT::view(). Le but de ces changements est de
  pouvoir masquer le conteneur utilisé pour l'implémentation
- Ajoute méthodes Arcane::arcaneParallelFor() et
  Arcane::arcaneParallelForeach() pour remplacer les
  différentes méthodes Arcane::Parallel::For() et Arcane::Parallel::Foreach().

Corrections:

- Dans l'AMR par patch, s'assure que les entités voisines par
  direction sont toujours dans le même niveau de patch.
- Corrige quelques dépendances manquantes lors de la compilation qui
  pouvaient entrainer des erreurs de compilation dans certains cas.
- Corrige erreurs de compilation des exemples en dehors du répertoire
  des sources.

Arcane Version 3.0.1 (27 mai 2021) {#arcanedoc_version301}
======================================

Cette version est la première version 'open source' de %Arcane.

Nouveautés/Améliorations:

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

Changements:

- Ajoute possibilité lors de la compilation de %Arcane de spécifier les
  packages requis et de ne pas chercher de packages par défaut.

Arcane Version 2.22.0 (03 mars 2021) {#arcanedoc_version2220}
======================================

Nouveautés/Améliorations:

- Support des protections/reprises pour les maillages AMR
- Modifications internes pour le support des accélérateurs, notamment
  le support des cartes graphiques (GPGPU).
- Suppression des classes et familles d'entités pour les
  Arcane::DualNode et Arcane::Link. Ces classes peuvent être
  remplacées par l'utilisation de Arcane::DoF.
- Support partiel des fichiers de maillage au format 'msh' version 4.1.

Changements:

- [IMPORTANT] Cette version utilise par défaut les nouvelles
  structures de connectivités. Cela ne devrait normalement pas avoir
  d'impact sur les codes existants à condition qu'ils n'utilisent pas
  les classes internes de %Arcane qui gèrent les entités (comme par
  exemple Arcane::ItemInternal).
- Les statistiques d'exécution sur les temps passés dans les
  différents points d'entrée et modules qui sont affichées à la fin du
  calcul contiennent maintenant le cumul sur l'ensemble des exécutions
  (en cas d'utilisation de protection/reprise) et plus seulement les
  statistiques sur l'exécution courante.
- Simplification des différents mécanismes pour initialiser et
  exécuter un code utilisant %Arcane (voir \ref arcanedoc_launcher)

Corrections:

- Corrige problème potentiel en parallèle lors de la création du
  répertoire pour les protections. Il était possible pour certains
  processus de tenter d'accéder à ce répertoire avant sa création.
- Corrige utilisation du script `arcane_convert_mesh` servant pour la
  conversion de maillages.
- Ré-active la génération de la documentation des services et modules
  fournis par %Arcane

Arcane Version 2.20.0 (07 janvier 2021) {#arcanedoc_version2200}
======================================

Nouveautés/Améliorations:

- Ajout d'une nouvelle interface 'Arcane::IAsyncParticleExchanger'
  permettant l'échange de particules de manière asynchronone. Un
  nouveau service `AsyncParticleExchanger` implémente cette interface.
- Ajoute la possibilité de supprimer pour une variable la mémoire
  additionnelle éventuellement utilisées via la méthode
  Arcane::IVariable::shrinkMemory()
- Début du support pour le raffinement de maillage cartésien en 2D. Il est
  maintenant possible d'appeler Arcane::ICartesianMesh::refinePatch2D()
  pour raffiner une partie du maillage
- Support pour les Arccore::MessagePassing::MessageId dans les messages de
  sérialisation (Arccore::MessagePassing::ISerializeMessage)
- Support de la version 6.0.0 de Papi et des versions récentes de PETSc pour `aleph`

Changements:

- Unification de la documentation entre les parties CEA et la partie commune
- Support des tests unitaires en parallèle (voir \ref arcanedoc_user_unit_tests_parallel)
- Modifications des classes Arcane::IData, Arcane::ISerializedData
  pour utiliser Arccore::Ref pour gérer les références
- Début refonte de la création et gestion de la mémoire des variables
  pour pouvoir à terme permettre aux développeurs d'ajouter leurs
  propres variables.

Corrections:

- Corrige plantage lors de l'utilisation de PTScotch lors d'un
  partitionnement en cas de redistribution mono-noeud.
- Nomme correctement le fichier 'ArcaneConfigVersion.cmake' pour
  pouvoir utiliser un numéro de version pour %Arcane dans la commande
  CMake `find_package()`.

Arcane Version 2.19.0 (30 juin 2020) {#arcanedoc_version2190}
======================================

\note Suite aux modifications dans la sérialisation, il n'est en
général pas possible de reprendre avec cette version de %Arcane un
calcul commencé avec une version antérieure.

Changements:

- Afin de préparer la modularisation d'%Arcane, la récupération du
  sous-domaine (Arcane::ISubDomain) est obsolète via beaucoup de classes (Arcane::IMesh,
  Arcane::IVariable, Arcane::ICaseOptions, ...). Les codes ayant besoin d'une
  instance de Arcane::ISubDomain doivent maintenant le passer explicitement
  aux méthodes ou classes l'utilisant. Les services et les modules continuent
  d'avoir accès aux sous-domaines.
- Utilisation de Arcane::MeshHandle au lieu de Arcane::IMesh. Avec la
  possibilité d'utiliser les services pour créér des maillages, les maillages ne
  sont plus créés avant les services. La classe Arcane::MeshHandle permet de
  gérer une référence sur un maillage avant que ce dernier n'existe. Il faut
  notamment utiliser cette classe dans les constructeurs des modules et
  services. La méthode Arcane::MeshHandle::mesh() permet de récupérer le
  maillage associé s'il a déjà été créé.
- Utilisation de la classe Arccore::Ref pour gérer les références sur les
  Arcane::IParallelMng. Par conséquent, la méthode
  Arcane::IParallelMng::createSubParallelMng() est déclarée obsolète et il faut
  maintenant utiliser Arcane::IParallelMng::createSubParallelMngRef().
- Modification de Arccore::Span2::operator[]() pour retourner des
  Arccore::Span au lieu de Arccore::ArrayView. Cela permet d'être cohérent avec
  les autres méthodes de Arccore::Span2.
- Modification de Arccore::ISerializer pour supporter l'écriture
  directe de tableaux et vérifier la cohérence entre la lecture et
  l'écriture d'une valeur donnée. Cela nécessite que les appels
  `reserve()/put()/get()` soient cohérents. Par exemple, si on utilise
  la méthode Arccore::ISerializer::reserveArray(), il faut utiliser
  ensuite Arccore::ISerializer::putArray() et
  Arccore::ISerializer::getArray(). Suite à cette modification, les méthodes
  Arccore::ISerializer::get() et Arccore::ISerializer::put() qui prenaient un
  Arccore::Span ou Arccore::ArrayView en argument sont obsolètes
- Utilisation d'une macro spécifique ARCANE_CHECK_MATH pour tester la
  validité des opérations mathématiques de Arcane::math. Auparavant
  cela était fait via la macro ARCANE_CHECK mais cela pouvait induire
  des effets de bord lorsqu'on comparait deux calculs avec et sans ce
  mode de vérification.

Nouveautés/Améliorations:

- Nombreuses améliorations sur l'échange de messages:
  - Support de toutes les méthodes de Arcane::IParallelMng en mode hybride et en
    mode mémoire partagée. En particulier, il manquait certaines méthodes de
    sérialisation.
  - Support des tags utilisateurs dans les méthodes point à point (send/receive)
  - Utilisation de types spécifiques pour gérer les tags (Arccore::MessageTag)
    et les rangs (Arccore::MessageRank). Ces types remplacent le type 'Int32'
    utilisé précédemment.
  - Unification de la gestion des méthodes point à point. Les paramètres
    nécessaires sont gérés par la classe Arccore::PointToPointMessageInfo.
  - support des méthodes MPI utilisant un `MPI_Message` (`MPI_Mprobe`,
    `MPI_Improbe`, `MPI_Imrecv`, `MPI_Mrecv`). Ces méthodes permettent de garantir
    qu'un `MPI_Recv` correspond bien au `MPI_Probe` utilisé précédemment. Ces
    mécanismes sont aussi accessibles en mode hybride.
- Support du partitionnement de maillage en mode mémoire partagée et hybride
- Ajoute dans le partitionneur `Metis` un mode permettant d'utiliser un seul
  processus pour le partitionnement. Auparavant il était nécessaire d'utiliser
  au minumum deux processus (à cause d'un bug dans `ParMetis`)
- Correction de l'algorithme permettant de garantir qu'il n'y a pas de
  partitions vides après utilisation de `ParMetis`.

- Préparation de Arccore::ISerializer pour supporter à terme d'autres types de
  données (par exemple des 'Float32' ou 'Float16')
- Enregistre un hash du nom de la variable lors des sérialisations. Cela
  permet de garantir qu'on est bien en train de désérialiser la bonne
  variable.
- Amélioration diverses du lecteur/écrivain au format JSON
- Génère dans les logs après un repartitionnement les informations sur les
  sous-domaines connectés à un sous-domaine.
- Support de la compilation via l'outil 'spack'.
- Dans le jeu de données, ajoute la possibilité de mettre un bloc
  `<comment>` en dessous de la racine. Cela est utile si un module
  n'est pas actif mais qu'on souhaite garder ses options. Il suffit
  alors de mettre ce bloc d'option entre les balises `<comment>` et
  `</comment>`.
- [.Net] Génère des packages `nuget` pour l'utilisation du wrapping. Il
  est donc possible de directement référencer ces pacakges plutôt que
  d'aller chercher directement les `.dll`. L'utilisation de packages
  `nuget` permet aussi de gérer automatiquement les dépendances.
- [.Net] Passage à la version 3.1 de '.Net Core'.

Les points suivants sont en cours de développement mais non finalisés:

- Gestion de la création des maillages sous forme de service. A terme seul ce
  mode sera disponible. Il permettra notamment de pouvoir créér des structures
  de maillages spécifiques (par exemple cartésien)
- Refonte de l’initialisation pour permettre d’utiliser %Arcane sans passer par
  une boucle en temps

Arcane Version 2.18.0 (09 décembre 2019) {#arcanedoc_version2180}
======================================

Cette version comporte les développements suivants:

- Amélioration de l'implémentation de la classe Arcane::HPReal qui
  corrige quelques cas limites et ajoute le support de la
  multiplication et de la division
- Création d'une classe universelle Arccore::Ref pour gérer la durée
  de vie des objets tels que les services. Cette classe permet de
  détruire automatiquement un objet lorsqu'il n'y a plus de
  référence dessus. Les services alloués dynamiquement utilisent
  maintenant cette classe. Par exemple:
  ~~~~~~~~~~~~~~~~~~~~~{.cpp}
  using namespace Arcane;
  ISubDomain* sd = ...;
  ServiceBuilder<IDataReader> sb(sd);
  IDataReader* old_reader = sb.createInstance(...); // Obsolète
  Arccore::Ref<IDataReader> new_reader = sb.createReference(...); // Nouveau mécanisme.
  ~~~~~~~~~~~~~~~~~~~~~
  Via cette classe Arccore::Ref, il ne faut plus détruire
  explicitement un objet. La méthode Arccore::makeRef() permet de
  créer une référence à partir d'un pointeur alloué par 'operator
  new'.
- Refonte partielle du mécanisme de lecture des options du jeu de
  données. Le but à terme est de pouvoir utiliser autre chose que
  XML pour l'entrée des données. Les modifications concernent
  essentiellement les classes de bases gérant les options et ne
  devraient pas avoir d'incidence sur le code utilisateur.
- Détecte les options du jeu de données sous la racine qui ne sont
  pas lues et retourne une erreur. La variable d'environnement
  **ARCANE_ALLOW_UNKNOWN_ROOT_ELEMENT** permet si elle positionnée
  d'afficher un avertissement plutôt qu'une erreur.
- [.Net] Passage à la version 3.0 de '.Net Core'.
- [.Net] Support en C# des classes gérants les matériaux et les milieux et
  des variables scalaires sur les constituants.
- [.Net] Support pour lancer les extensions utilisateur C# via l'outil
  `dotnet` à la place de `mono` ce qui permet de débugger le code C#
  via par exemple *Visual Studio Code*.
- Passage à la version 6.38.0 de Lima, qui permet de lire les
  fichiers au format `mli2`. Ce format utilise une version récente
  de *HDF5* et optimise notamment la taille des fichiers de maillage
  lorsqu'il y a un grand nombre de groupes vides.

Arcane Version 2.17.0 (09 octobre 2019) {#arcanedoc_version2170}
======================================

Cette version comporte les développements suivants:
- [.Net] Refonte de l'environnement `.Net` pour permettre de compiler
  soit avec `mono`, soit avec l'implémentation Microsoft `dotnet`.
- [.Net] Uniformisation de l'initialisation avec ou sans l'utilisation du
  runtime `.Net`.
- Ajout dans le répertoire 'samples' d'un exemple 'EOS' montrant comment
  rendre accessible en C# des services ou des classes C++.
- Application stricte des règles de conversion XML pour la lecture de
  l'attribut `active` dans l'élément `<module>` indiquant si un module
  est actif ou non. Maintenant, seules les valeurs `true`, `false`,
  `0` ou `1` sont autorisées. Auparanvant, toute valeur autre que
  `false` était considérée comme `true`.
- Support dans le profiling MPI via la bibliothèque OTF2 des cas avec
  équilibrage de charge et retour-arrière.
- Possibilité de détruire les maillages additionnels via un la classe
  Arcane::IMeshMng:
  ~~~~~~~~~~~~~~~~~~~~{.cpp}
  Arcane::ISubDomain* sd = ...;
  Arcane::IMesh* mesh_to_destroy = ...;
  sd->meshMng()->destroyMesh(mesh_to_destroy);
  ~~~~~~~~~~~~~~~~~~~~
  Le pointeur `mesh_to_destroy` ne doit plus être utilisé après appel
  à la méthode de destruction.
- Support dans Arcane::mesh::BasicParticleExchanger des particules qui
  ne sont pas dans des mailles (celles pour lequelles
  Arcane::Particle::cell() retournent une maille nulle)
- Ajoute options dans le service 'BasicParticleExchanger' pour
  permettre de choisir le nombre de messages maximumum à effectuer
  avant de faire les réductions
- Ajoute pour test une variable d'environnement
  `ARCANE_VARIABLE_SHRINK_MEMORY` qui si elle est positionnée à `1`
  redimensionne au plus juste la mémoire alloué par les variables
  après modification du nombre d'entités
- Dans `Aleph`, détruit les communicateurs MPI dans le destructeur de
  `AlephKernel`. Cela avait désactivé pour des raisons de
  compatibilité avec PETSc et d'anciennes versions de MPI.

Arcane Version 2.16.0 (18 juillet 2019) {#arcanedoc_version2160}
======================================

Cette version comporte les développements suivants:

- Ajout du profiling interne pour les points d'entrée et les appels
  MPI. Le format des traces dépend de la valeur de la variable
  d'environnement qui active cette fonctionnalité :
  **ARCANE_MESSAGE_PASSING_PROFILING=OTF2** ou **JSON** (cf doc
  variable environnement et analyse de performances).
- Refonte du wrapper C#. Le wrapper est maintenant composé de
  plusieurs modules (Arcane.Core, Arcane.Hdf5 et Arcane.Services)
  et les méthodes wrappées utilisent maintenant les règles de
  codage du C#: elles commencent par une majuscule.
- ajout d'une classe Arcane::MeshHandle pour gérer les maillages avant
  leur création effective. A terme cela permettra de supprimer des
  maillages et de les créer via des services.

Arcane Version 2.15.0 (13 juin 2019) {#arcanedoc_version2150}
======================================

Cette version comporte les développements suivants:

- Support pour les messages MPI de plus de 2Go. Cela ne concerne que
  les message de type MPI_Send/MPI_Recv et uniquement si on passe
  par Arcane::ISerializer.
- Complète l'implémentation de Arcane::IParallelMng en mode mémoire
  partagée. Toutes les méthodes à part
  Arcane::IParallelMng::createSubParallelMng() sont implémentées. Les classes gérant ce
  mode de parallélisme sont renommées et commencent par *SharedMemory* au lieu de
  *Thread*.
- Renommage des classes gérant les échanges de message en mode
  MPI+Mémoire partagée. Ces classes ont un nom qui commence par
  *Hybrid* au lieu de *MpiThread*.
- Support de `.NetCore` en plus de `mono` pour la gestion des fichiers
  *AXL*. Les projets gérant les fichiers *AXL* peuvent donc utiliser
  soit le .NetFramework 4.5 (avec mono), soit .NetCoreApp 2.2 (avec
  .NetCore). Il est maintenant indispensable d'avoir l'outil `msbuild`
  pour compiler avec `mono`. Par défaut, c'est l'implémentation
  .NetCore qui est utilisée.
- la classe Arccore::String considère maintenant que les arguments
  de type `const char*` sont encodés en UTF-8 et plus en ISO-8859-1.
- la classe Arccore::String gère maintenant en interne les chaînes de caractères
  de plus de 2Go. Par conséquent, la méthode Arccore::String::len()
  devient obsolète et doit être remplacée par la méthode
  Arccore::String::length() qui retourne un *Int64*.
- Il est maintenant possible de construire des instances de
  Arccore::String à partir de la classe std::string_view du C++17. Par
  conséquent, les méthodes de Arccore::String qui prenaient un `const
  char*` et une longueur en argument devient obsolètes et sont
  remplacées par des méthodes prenant des std::string_view en argument.
- un mode reproductible pour ParMetis a été développé qui permet de
  garantir le même partitionnement entre deux exécutions en regroupant
  le graphe sur un seul processeur. Ce mode ne doit pas être utilisé lorsque
  le nombre de mailles est supérieur à quelques dizaines de millions.

Arcane Version 2.14.0 (04 mars 2019) {#arcanedoc_version2140}
======================================

Cette version comporte les développements suivants:

- intégration des développements IFPEN concernant les entités de type
  *degré de liberté* (classe Arcane::DoF).
- possibilité de rediriger les sorties listing à la fois sur le flot
  standard (`stdout`) et dans un fichier avec la possibilité de ne pas
  avoir des niveaux de verbosité différents. Suite à ce changement, si
  la variable d'environnement `ARCANE_PARALLEL_OUTPUT` est positionnée
  alors le sous-domaine 0 écrit le listing dans un fichier comme les
  autres sous-domaine (les fichiers s'appellent `output...`). Pour
  spécifier qu'on souhaite écrire un fichier listing il faut
  positionner la variable d'environnement
  `ARCANE_MASTER_HAS_OUTPUT_FILE` ou appeler la méthode
  Arcane::ITraceMngPolicy::setIsMasterHasOutputFile() avant
  l'initilisation (par exemple en surchargeant Arcane::MainFactory).
- passage en UTF-8 de toutes les sources (%Arcane,%Arccore,...)
- déplacement des sources de 'arccore' dans son propre dépôt GIT et
  déplacement des sources de `arcane/dof` dans `arcane/mesh`

Arcane Version 2.13.0 (21 janvier 2019) {#arcanedoc_version2130}
======================================

Cette version comporte les développements suivants:

- **[INCOMPATIBILITÉ]** Modification des méthodes *begin()* et *end()* pour les
  classes tableaux et vues sur les tableaux (Arccore::ArrayView,
  Arccore::ConstArrayView, Arccore::Span, Arccore::Array) afin de
  retourner un itérateur et plus un pointeur. Si on souhaite récupérer
  un pointeur, il faut utiliser la méhode *data()* à la place.
- **[INCOMPATIBILITÉ]** Interdit la création d'instances de la classe
  Arccore::Array. Il faut utiliser soit Arccore::UniqueArray, soit
  Arccore::SharedArray. La classe Arccore::Array doit être utilisée
  par référence en argument ou en retour de fonction.
- Passage partiel des sources de %Arcane UTF-8
- Passage à la version 1.10+ de `HDF5` (au lieu de 1.8)
- Support des maillages au format `MED` (format utilisé par CEA/DEN
  notamment par la plateforme Salomé)
- Modification du générateur 'axldoc' pour génèrer des fichiers
  Doxygen au format Markdown avec l'extension '.md' au lieu de
  fichiers .dox

Arcane Version 2.12.0 (11 décembre 2018) {#arcanedoc_version2120}
======================================

Cette version comporte les développements suivants:

- amélioration de la visualisation des variables avec les dernières
  versions de totalview et ajout de la visualisation pour les
  variables scalaires
- création d'une classe Arccore::Span2 pour gérer les vues sur les
  tableaux 2D avec les tailles sur 64bits.
- refonte interne des protections/reprises pour permettre de changer
  le nombre de sous-domaines en reprise.

Arcane Version 2.11.0 (18 octobre 2018) {#arcanedoc_version2110}
======================================

Cette version comporte les développements suivants:

- Support des tableaux (Arccore::Array) et chaînes de caractères
  (Arccore::String) dépassant 2Go. En interne, le nombre d'éléments
  est maintenant stocké sur 64bits au lieu de 32bits.
- Ajout des classes 'LargeArrayView' et 'ConstLargeArrayView' qui
  sont identiques à 'ArrayView' et 'ConstArrayView' mais utilisent
  une taille sur 64 bits.

Arcane Version 2.10.1 (04 octobre 2018) {#arcanedoc_version2101}
======================================

Cette version comporte les développements suivants:

- Possibilité de spécifier plusieurs interfaces pour les services via
  la macro ARCANE_REGISTER_SERVICE(). Auparavant cela n'était possible
  que via les fichiers 'axl'.
- Les services singletons qui implémentent plusieurs interfaces
  ne sont créés qu'une seule fois et les interfaces font donc
  référence à la même instance. Auparavant, il y a avait autant
  d'instances que d'interfaces.
- Il est possible de spécifier dans le fichier 'axl' qu'un service est
  singleton (voir \ref arcanedoc_service_desc).
- Possibilité de spécifier des valeurs par défaut par catégorie via un
  nouvel élément **defaultvalue** dans les fichiers 'axl'. La
  catégorie utilisée lors de l'exécution peut être positionnée pour
  chaque code via la méthode
  Arcane::ICaseDocument::setDefaultCategory(). La page
  \ref arcanedoc_caseoptions_struct_common indique comment ajouter ces
  valeurs par défaut.
- Possibilité de charger des services singleton dans le jeu de
  données à la manière des modules via un nouvel élément **services**
  dans l'élément <arcane> (voir \ref arcanedoc_casefile_arcaneelement).

Arcane Version 2.10.0 (septembre 2018) {#arcanedoc_version2100}
======================================

Cette version comporte les développements suivants:

- Création d'une nouvelle composante '%Arccore' regroupant la partie de
  %Arcane commune avec Alien. Cette composante contient une partie de
  'arcane_utils' et 'arcane_mpi'. Par conséquent, les classes de base
  telles que Arccore::String, Arccore::ArrayView, Arccore::Array sont maintenant dans le
  namespace 'Arccore' au lieu de 'Arcane'. Via le mécanisme 'using' du
  C++ ces classes sont aussi disponibles dans le namespace Arcane et
  il ne devrait donc pas y avoir d'incompatibilité avec le code source
  existant. La seule condition est de ne pas déclarer explicitement
  les types %Arcane mais d'utiliser le fichier d'en-tête
  'arcane/utils/UtilsTypes.h'.
- Utilisation de la bibliothèque 'libxml2' pour gérer le XML au lieu
  de la bibliothèque 'XercesC'. Cette modification a été effectuée
  pour deux fonctionnalités qui ne sont pas disponibles dans
  'XercesC': le support de XInclude avec XmlSchema et l'encodage UTF-8
  des chaînes de caractères.
- Rend obsolète la construction de 'Arccore::Array'. Il faut explicitement
  utiliser soit 'Arccore::UniqueArray', soit 'Arccore::SharedArray'.
- Sépare l'implémentation de Arccore::UniqueArray de celle de Arccore::SharedArray pour
  éviter de conserver des informations qui ne sont utiles que pour
  l'une ou l'autre de ces classes.
- Possibilité de spécifier un nombre de threads compris en 1 et le
  nombre maximum de threads alloué lors d'une boucle parallèle
  Arcane::Parallel::ForEach. Cela se fait via la classe Arcane::ParallelLoopOptions.
- Le fichier 'arcane.pc' généré pour pkg-config est obsolète et ne
  contient plus toutes les bibliothèques utilisées par %Arcane. Pour
  avoir une fonctionnalité équivalente, il faut utiliser 'cmake' avec
  l'option '--find-package'.
- Refonte de la gestion des itérateurs sur les classes tableaux
  (Arccore::Array, Arccore::ArrayView, Arccore::ConstArrayView). Les
  itérateurs actuels qui retournaient un pointeur sont déclarés
  obsolète et remplacés par un objet du type Arccore::ArrayIterator. Le fait de
  retourner un pointeur pouvait poser problème en cas d'héritage. Par
  exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
class A { ... };
class B : public A { ... };

Arcane::Array<B*> array_of_b;
for( A* a : array_of_b){
  // Plantage si sizeof(A)!=sizeof(B)
}
~~~~~~~~~~~~~~~~~~~~~
  Le nouvel itérateur est partagé entre les trois classes tableaux et
  est du type std::random_iterator.
  Suivant les cas d'utilisation, le code actuel peut-être changé comme
  suit:
  
  - Utilisation de begin() pour récupérer un pointeur sur le début du
    tableau: utiliser data() à la place:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
Arcane::ArrayView<Int32> a;
Int32* v = a.begin(); // Obsolète
Int32* v = a.data();  // OK.
~~~~~~~~~~~~~~~~~~~~~
  - Utilisation de begin()/end() dans les algorithmes de la STL. Dans
    ce cas il faut remplacer ces méthodes par std::begin() et std::end():
~~~~~~~~~~~~~~~~~~~~~{.cpp}
Arcane::ArrayView<Int32> a;
std::sort(a.begin(),a.end()); // Obsolète
std::sort(std::begin(a),std::end(a)); // OK
~~~~~~~~~~~~~~~~~~~~~
  - Utilisation dans le cas d'une boucle for-range du C++11. Dans ce
    cas il faut utiliser la méthode range du tableau. Par exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
Arcane::ArrayView<Int32> a;
for( Int32 x : a) {} // Obsolète
for( Int32 x : a.range()) {} // OK
~~~~~~~~~~~~~~~~~~~~~

Arcane Version 2.9.1 (juin 2018) {#arcanedoc_version291}
======================================

Cette version comporte les développements suivants:
- Possiblité de changer les valeurs par défaut des options du jeu de
  données. Pour plus d'informations, se reporter à la section
  \ref arcanedoc_caseoptions_defaultvalues.
- Ajoute méthode valueIfPresentOrArgument() pour les options
  simples, énumérées ou étendues. Cela permet de remplacer le code
  suivant:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
Real x = 3.2;
if (options()->myOption.isPresent())
  x = options()->myOptions();
~~~~~~~~~~~~~~~~~~~~~
  par le code suivant:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
Real x = options()->myOption.valueIfPresentOrArgument(3.2);
~~~~~~~~~~~~~~~~~~~~~
- Support 64 bits pour l'échangeur de particule non bloquant
  (service **NonBlockingParticleExchanger** implémentant
  Arcane::IParticleExchanger). Cela permet de dépasser 2^31 particules
  lors d'un échange.
- Ajoute méthode Arcane::ITimeLoopMng::stopReason() permettant de connaitre la
  raison de l'arrêt du code. En particulier, il est maintenant
  possible de savoir si on exécute la dernière itération lorsqu'un
  nombre maximum d'itération est spécifié.
- Dans le jeu de données, dans la liste des modules à activer, il est
  maintenant possible de spécifier le nom (traduit ou nom) de l'élément XML du
  Module au lieu du nom du module.
  Par exemple, pour le module de protection/reprise de %Arcane, dont le
  nom est 'ArcaneCheckoint', on a actuellement:
~~~~~~~~~~~~~~~~~~~~~{.xml}
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <modules>
      <module name="ArcaneCheckpoint" actif="true" />
    </modules>
  </arcane>
  ...
  <arcane-protections-reprises>
    <periode>3</periode>
  </arcane-protections-reprises>
</cas>
~~~~~~~~~~~~~~~~~~~~~
  et maintenant il est possible de le remplacer par:
~~~~~~~~~~~~~~~~~~~~~{.xml}
<cas codename="ArcaneTest" xml:lang="fr" codeversion="1.0">
  <arcane>
    <modules>
      <module name="arcane-protections-reprises" actif="true" />
    </modules>
  </arcane>
  ...
  <arcane-protections-reprises>
    <periode>3</periode>
  </arcane-protections-reprises>
</cas>
~~~~~~~~~~~~~~~~~~~~~

Arcane Version 2.8.0 (31 janvier 2018) {#arcanedoc_version280}
======================================

Cette version comporte les développements suivants:

- Ajout pour les tests unitaires de deux macros #ASSERT_NEARLY_ZERO et
  #ASSERT_NEARLY_ZERO_EPSILON qui permettent de comparer avec zéro
  une valeur.
- Détermination lors de la compilation des types 'Arcane::Int16', 'Arcane::Int32' et
  'Arcane::Int64'. A cause de cela, le fichier 'ArcaneGlobal.h' n'inclut plus
  le fichier 'limits.h'.
- Par défaut, les unités de longueur spécifiées dans les fichiers
  au format Lima (.unf, .mli, ...) sont maintenant toujours prises en
  compte. Auparavant, il fallait mettre la valeur 1 à l'attribut 'use-unit' (en
  anglais) ou 'utilise-unite' (en francais) dans l'élément <mesh> (en
  anglais) ou <maillage> (en francais). Pour retrouver le comportement
  d'avant, il faut mettre la valeur '0' à cet attribut.
  Par exemple en francais:
  ~~~~~~~~~~~~~~~~~~~~~{.xml}
  <maillage utilise-unite="0">
  ...
  </maillage>
  ~~~~~~~~~~~~~~~~~~~~~

  ou en anglais:

  ~~~~~~~~~~~~~~~~~~~~~{.xml}
  <mesh use-unit="0">
  ...
  </mesh>
  ~~~~~~~~~~~~~~~~~~~~~

Arcane Version 2.7.0 (20 octobre 2017) {#arcanedoc_version270}
======================================

Cette version comporte les développements suivants sur les matériaux
et les milieux:
- possibilité d'itérer uniquement sur les mailles pures ou impures
  d'un matériau ou d'un milieu.
- nouveaux mécanismes pour boucler sur les matériaux et les milieux
- début du support de la vectorisation sur les matériaux et les
  milieux. Pour l'instant ce supporte concerne seulement les variables
  scalaires et uniquement sur les milieux.

Pour plus d'informations sur ces notions, se référer à la page \ref arcanedoc_materialloop.

Arcane Version 2.6.1 (18 septembre 2017) {#arcanedoc_version261}
======================================

Cette version comporte les développements visibles suivants:
- dans les options du jeu de données, ne tolère plus les caractères
  invalides en fin de chaîne de caractères de l'option. Par exemple, la chaîne '12a3'
  était considérée comme valide pour une option de type entier et sa
  valeur était '12'. Maintenant une erreur est retournée dans ce cas.

Cette version comporte des développements internes visant à terme à
supprimer les anciens mécanismes d'accès aux connectivités.

Arcane Version 2.6.0 (22 aout 2017) {#arcanedoc_version260}
======================================

Cette version met en service le nouveau mécanisme d'accès aux
connectivités. L'ancien mécanisme reste accessible. Les deux
mécanismes utilisants des gestions de la mémoire différents, les
connectivités sont donc allouées à la fois avec le nouveau et
l'ancien mécanisme ce qui se traduit par une augmentation de la
mémoire d'environ 1ko par maille.
Pour plus d'informations, se reporter à la page \ref arcanedoc_connectivity_internal.

Arcane Version 2.5.2 (25 juillet 2017) {#arcanedoc_version252}
======================================

Cette version contient les développements suivants:

- ajout d'un mécanisme de gestion des évènements (voir fichier
  "arcane/utils/Event.h" et la classe EventObservable pour un exemple d'utilisation)
- ajout d'évènements lors de l'ajout/suppression de variables
  (Arcane::IVariableMng::onVariableAdded() et
  Arcane::IVariableMng::onVariableRemoved()) et lors des synchronisations
  (Arcane::IVariableSynchronizer::onSynchronized()).
- possibilité de spécifier un partitionneur statique pour les boucles
  utilisant le multi-threading. Cela permet d'avoir un comportement
  déterministe (i.e répétable) entre plusieurs exécutions à condition
  d'utiliser le même nombre de threads.
~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <arcane/Concurrency.h>
using namespace Arcane;
ParallelLoopOptions opt;
opt.setPartitioner(ParallelLoopOptions::Partitioner::Static);
TaskFactory::setDefaultParallelLoopOptions(opt);
~~~~~~~~~~~~~~~~~~~~~

- ajout d'un mécanisme de barrière nommée pour garantir que tous les
  processus arrivent à la même barrière:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
#include <arcane/Parallel.h>
using namespace Arcane;
IParallelMng* pm = ...;
MessagePassing::namedBarrier(pm,"MyBarrier");
~~~~~~~~~~~~~~~~~~~~~
- possibilité de mixer déclarations de variables matériaux et milieux
  (pour des variables scalaires uniquement). Dans ce cas toutes les
  variables sont allouées sur les matériaux et les milieux. Par défaut
  cela n'est pas actif car cela consomme de la mémoire inutilement.
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane::Materials;
IMeshMaterialMng* mm = ...;
mm->setAllocateScalarEnvironmentVariableAsMaterial(true);
MaterialVariableCellReal mat_var(MaterialVariableBuildInfo(mm,"Var1"));
EnvironmentVariableCellReal env_var(MaterialVariableBuildInfo(mm,"Var1"));
// 'mat_var' et 'env var' sont les mêmes variables.
~~~~~~~~~~~~~~~~~~~~~


Arcane Version 2.5.1 (04 mai 2017) {#arcanedoc_version251}
======================================

 Cette version contient uniquement des développements internes et
 n'ajoute pas de fonctionnalités directement accessibles aux utilisateurs.

Arcane Version 2.5.0 (15 mars 2017) {#arcanedoc_version250}
======================================

Cette version intègre les développements pour découpler la gestion
des connectivités de celle des entités. Pour des raisons d'économie
mémoire, le mécanisme actuel permet d'avoir toutes les connectivités
disponibles pour chaque type entité. Potentiellement, cela signifie par exemple qu'une
particule et une maille peuvent avoir les mêmes connectivités. Pour
plus d'informations sur les nouveaux mécanismes, se reporter à la page suivante \ref
arcanedoc_connectivity_internal. Dans cette version 2.5.0, les
nouvelles connectivités ne sont pas actives (ce qui correspond au
mode de configuration '--with-legacy-connectivity' décrit dans la
page du lien précédent) donc le comportement des
codes utilisant %Arcane ne devraient pas changer.

Les autres nouvelles fonctionnalités sont:
- ajout d'une méthode Item::isShared() pour indiquer si une entité est
  partagée par plusieurs sous-domaines. Par exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
Cell cell = ...;
if (cell.isShared())
  info() << "Cell is shared";
~~~~~~~~~~~~~~~~~~~~~

- la synchronisation de plusieurs variables en un seul message MPI est
  maintenant active par défaut. Par exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
VariableCellReal temperature = ...;
VariableCellReal3 cell_center = ...;
VariableCollection vars;
vars.add(temperature);
vars.add(cell_center);
mesh()->cellFamily()->synchronize(vars);
~~~~~~~~~~~~~~~~~~~~~
A noter que si la collection spécifiée en paramètre contient des
variables partielles, la synchronisation se fait comme avant soit
variable par variable.

Arcane Version 2.4.2 (13 Janvier 2017) {#arcanedoc_version242}
==============================

- Ajoute la possibilité de configurer les traces lors de l'exécution en
  spécifiant une chaîne de caractères au format de configuration des
  traces (tel que dans la documentation \ref arcanedoc_traces). Par
  exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
String x = "<?xml version=\"1.0\"?>\n"
 "<arcane-config>\n"
 "<traces>\n"
 "<trace-class name='MyTest2' info='true' debug='med' print-elapsed-time='true' print-class-name='false'/>\n"
 "</traces>\n"
 "</arcane-config>\n";
ISubDomain* sd = ...;
ITraceMng* tm = sd->traceMng();
sd->application()->getTraceMngPolicy()->setClassConfigFromXmlBuffer(tm,x.utf8());
~~~~~~~~~~~~~~~~~~~~~

- Ajoute la possibilité d'afficher ou non pour chaque message de
  trace le nom de la classe de message dans les traces ainsi que le
  temps écoulé. Par exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
ITraceMng* tm = ...;
TraceClassConfig tcc = tm->classConfig("MyTest");
tcc.setFlags(Trace::PF_ElapsedTime|Trace::PF_NoClassName);
tm->setClassConfig("MyTest",tcc);
~~~~~~~~~~~~~~~~~~~~~
- Ajoute une méthode Arcane::IMeshUtilities::mergeNodes() permettant la fusion de noeuds deux à deux.
- Le mécanisme interne de compactage des entités a été réécrit afin
  de pouvoir être plus facilement paramétrable par famille d'entité et plus
  facilement extensible si on ajoute un nouveau type de famille d'entité

Arcane Version 2.4.1 (01 Décembre 2016) {#arcanedoc_version241}
======================================

- Support de la méthode Arcane::MeshMaterialVariableRef::synchronize() pour les
  variables uniquement sur les milieux.
- Ajoute une méthode sur les variables matériaux pour remplir les
  valeurs partielles avec la valeur de la maille du composant
  supérieur. Cela permet de remplir les valeurs matériaux avec les
  valeurs milieux ou les valeurs milieux avec les valeurs globales. La
  méthode s'appelle
  Arcane::Materials::MeshMaterialVariableRef::fillPartialValuesWithSuperValues().
- Ajoute les itérateurs STL constants pour la classe
  Arcane::ItemVectorView. Cela permet d'utiliser les algorithmes de la STL avec
  cette classe. Par exemple:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
ItemVector v1;
ItemGroup group1;
ItemVectorView group_view = group1.view();
for( Item item : v1.view() ){
  // Regarde si 'item' est dans le groupe 'group1'.
  auto iter = std::find(group_view.begin(),group_view().end,item);
  if (iter!=group_view.end()){
    // Trouvé.
  }
}
for( Item item : group1.view() ){
  info() << "Item=" << ItemPrinter(item);
}
~~~~~~~~~~~~~~~~~~~~~
- Ajoute la possibilité de spécifier sa propre fonction de calcul des
  connectivités pour les Arcane::ItemPairGroup. La documentation de la classe
  ItemPairGroup contient un exemple d'un tel calcul
- Ajoute dans les services de partitionnement plusieurs options:
  - pour Parmetis la possibilité d'écrire le graphe sur un fichier
    et de spécifier la tolérance,
  - pour PTScotch la possibilité d'écrire le graphe sur un fichier
    et de vérifier la cohérence du graphe.

Arcane Version 2.4.0 (Novembre 2016) {#arcanedoc_version240}
======================================

- Support des variables uniquement sur les milieux. Ces variables
  s'utilisent comme les variables matériau mais n'ont de valeurs
  que sur les milieux et les mailles globales. Il ne faut donc pas
  les indexer avec des mailles matériaux (MatCell) sous peine de
  provoquer un accès mémoire illégal. La déclaration de ces variables
  se fait comme suit:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
EnvironmentVariableCellReal pressure(VariableBuildInfo(mesh,"Pressure"));
IMeshEnvironment* env = ...;
ENUMERATE_ENVCELL(ienvcell,env){
  pressure[ienvcell] = 2.0;
}
~~~~~~~~~~~~~~~~~~~~~
- Support du repartitionnement et de l'équilibrage de charge en
  conservant les informations des matériaux. Les valeurs partielles
  des mailles matériaux et milieux sont aussi conservées.
- Refonte interne de la gestion des échanges des entités lors d'un
  repartitionnement. Le but est de pouvoir ajouter plus facilement de
  nouvelles familles d'entités où de changer le traitement de l'un
  d'elle. Cette modification n'a aucun impact sur le code existant.

Arcane Version 2.3.9 (Septembre 2016) {#arcanedoc_version239}
======================================

Amélioration de la vectorisation {#arcanedoc_version239_simd}
--------------------------------------

- Possibilité de choisir un allocateur spécifique (implémentant
  l'interface IMemoryAllocator) pour les classes Array et
  UniqueArray. Une implémentation gérant l'alignement
  (Arcane::AlignedMemoryAllocator) est disponible et permet de garantir que
  l'adresse de la mémoire allouée est un multiple d'une certaine
  valeur. Pour l'instant la seule valeur valide pour l'alignement est 64 et il
  faut utiliser l'allocateur Arcane::AlignedMemoryAllocator::Simd() pour
  récupérer cet l'allocateur avec cet alignement. En plus d'aligner la
  mémoire, cet allocateur garantit que la taille dela mémoire allouée est un
  multiple de l'alignement.
  Par exemple, le code suivant garantit un alignement:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
UniqueArray<Real> x(AlignedMemoryAllocator::Simd());
x.resize(25);
// &x[0] est un multiple de 64.
// Un réel faisant 8 octets, la capacité de x est un
// multiple de (64 / 8), soit x.capacity()>=32 (car 32 est le
// premier multiple de 8 supérieur à 25).
~~~~~~~~~~~~~~~~~~~~~
  L'alignement de 64 octets permet d'utiliser tous les mécanismes de
  vectorisation disponibles à ce jour dans %Arcane, soit SSE, AVX et
  AVX512.
- Les variables tableaux 1D et 2D ainsi que les variables scalaires
  et tableau sur les entités du maillage sont maintenant toujours
  allouées avec l'allocateur Arcane::AlignedMemoryAllocator::Simd().
- Les indices des entités d'un ItemGroup sont maintenant toujours
  alloués avec l'allocateur Arcane::AlignedMemoryAllocator::Simd(). A noter que
  ce n'est pas le cas de Arcane::ItemVector.
- Les différentes macros utilisées pour gérer l'enumération avec la
  vectorisation (#ENUMERATE_SIMD_CELL, #ENUMERATE_SIMD_NODE, ...)
  nécessitent maintenant des tableaux
  alignés. Si ce n'est pas le cas, cela provoque une exception lors de
  l'exécution.
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CellGroup cells = ...;
CellVector vec_cells = ...;
ENUMERATE_SIMD_CELL(ivcell,cells){ // OK car un CellGroup est toujours aligné
}
ENUMERATE_SIMD_CELL(ivcell,vec_cells){ // ERREUR car un CellVector n'est pas toujours aligné
}
~~~~~~~~~~~~~~~~~~~~~
- L'utilisation des méthodes Parallel::Foreach() sur les groupes
  d'entités garantit que les itérations se feront sur des multiples de
  8 valeurs. Il est ainsi valide d'écrire le code suivant:
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
CellGroup cells = ...;
Parallel::Foreach(cells,[this](CellVectorView cvv){
  ENUMERATE_SIMD_CELL(ivcell,cvv){
    SimdCell cell = *ivcell;
    ...
  }
}
~~~~~~~~~~~~~~~~~~~~~
- Ajoute le support pour la vectorisation de type AVX512 et supprime
  la vectorisation spécifique aux processeurs Intel Knight Corner
  (KNC). L'AVX512 est supportée à partir des Xeon Skylake et sur les
  processeurs de XeonPhi à partir de l'Intel Knight Landing (KNL).
- Autorise plusieurs types de vectorisation à la fois via les
  classes de vectorisation de %Arcane. Par exemple, sur les machines haswell qui supportent
  l'AVX et le SSE les classes SSESimdReal et AVXSimdReal sont
  toujours disponibles. Dans les versions précédentes de %Arcane, seul le type de vectorisation
  le plus performant était disponible (AVX512 > AVX > SSE).

Améliorations diverses {#arcanedoc_version239_misc}
--------------------------------------

- Support de la sémantique std::move() du C++11 pour la classe
  UniqueArray. Cela permet entre autre de retourner des UniqueArray
  sans faire de recopie mémoire, d'implémenter std::swap() de manière
  optimisée.
- Ajoute de versions optimisées pour échanger les valeurs de deux
  variables du maillage (scalaire ou tableau) via une méthode swapValues(). Cette méthode
  permet d'échanger juste les pointeurs contenant les zones mémoires
  des deux variables sans recopie. Par exemple
~~~~~~~~~~~~~~~~~~~~~{.cpp}
using namespace Arcane;
VariableCellReal temperature = ...;
VariableCellReal old_temperature = ...;
VariableCellArrayReal energy = ...; // Variable 1D sur les mailles
VariableCellArrayReal old_energy = ...;
old_temperature.swapValues(temperature);
old_energy.swapValues(energy);
~~~~~~~~~~~~~~~~~~~~~
- Passage à la version 1.11.4 de hwloc.
- [C#] Dans Arcane.Curves, support des courbes avec plusieurs
  valeurs par itération (courbes 2D).
- Utilise le temps réel (elapsed) au lieu du temps CPU pour les
  statistiques de fin de calcul. Le temps réel est plus précis que le
  temps CPU et est indépendant du nombre de threads utilisés. Il est
  possible de remettre l'utilisation du temps CPU en positionnant la variable
  d'environnement ARCANE_USE_VIRTUAL_TIMER.

