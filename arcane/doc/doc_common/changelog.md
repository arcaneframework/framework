# Nouvelles fonctionnalités {#arcanedoc_news_changelog}

[TOC]

Cette page contient les nouveautés de chaque version de %Arcane v3.X.X.

Les nouveautés successives apportées par %Arcane v2.X.X sont listées ici : \ref arcanedoc_news_changelog20



___
## Arcane Version 3.6.13 (06 juillet 2022) {#arcanedoc_news_changelog_version360}

### Nouveautés/Améliorations:

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

### Changements:

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

### Corrections:

- Corrige mauvaises valeurs de Arcane::IItemFamily::localConnectivityInfos()
  et Arcane::IItemFamily::globalConnectivityInfos() pour les connectivités
  autres que celles aux noeuds. Ce bug avait été introduit lors du
  passage aux nouvelles connectivités (#230, #27).
- Corrige divers bugs dans la version 3 de BasicReaderWriter (#238)

### Interne:

- Utilise des variables pour conserver les champs Arcane::ItemInternal::owner() et
  Arcane::ItemInternal::flags() au lieu de conserver l'information dans
  Arcane::ItemSharedInfo. Cela permettra à terme de supprimer le champ
  correspondant dans Arcane::ItemSharedInfo (#227).

Passage version 2.0.3.0 de Axlstar:

  - Ajoute support dans les fichiers 'axl' des propriétés Arcane::IVariable::PNoExchange,
    Arcane::IVariable::PNoReplicaSync et Arcane::IVariable::PPersistant.

Passage version 2.0.11.0 de %Arccore:

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

### Changements:

- Supprime les macros plus utilisées ARCANE_PROXY et ARCANE_TRACE (#145)

### Corrections:

- Corrige mauvaise détection de la version OneTBB 2021.5 suite à la
  suppression du fichier 'tbb_thread.h' (#146)
- Corrige certaines informations cartésiennes manquantes lorsqu'il n'y
  a qu'une seule couche de maille en Y ou Z (#162).
- Corrige implémentation manquante de 'Arccore::Span<T>::operator=='
  lorsque le type `T` n'est pas constant (#163).
- Supprime quelques messages listings trop nombreux.

### Interne:

- Utilise au lieu de Arccore::UniqueArray2 une implémentation
  spécifique pour le conteneur de Arcane::NumArray (#150).
- Utilise des `Int32` au lieu de `Int64` pour indexer les éléments
  dans Arcane::NumArray (#153)
- Ajoute `constexpr` et `noexcept` à certaines classes de Arccore
  (#156).
- Passage version 2.0.9.0 de Arccore




___
## Arcane Version 3.4.5 (10 février 2022) {#arcanedoc_news_changelog_version340}

### Nouveautés/Améliorations:

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

### Changements:

- Ajoute namespace '%Arcane::' aux cibles `CMake` fournit par
%Arcane. Par exemple, la cible 'arcane_core' devient
'Arcane::arcane_core'. Les anciens noms restent valides (#120).
- Rend obsolète la conversion de Arcane::ItemEnumerator vers
  Arcane::ItemEnumeratorT. Cela permet d'éviter d'indexer par erreur
  une variable du maillage avec un énumérateur du mauvais type (par
  exemple indexer une variable aux mailles avec un énumérateur aux noeuds).

### Corrections:

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

### Interne:

- Nettoyage de l'API Accélérateur

Passage version 2.0.8.1 de %Arccore:

  - Improve doxygen documentation for types et classes in `message_passing` component.
  - Add functions in `message_passing` component to handle non blocking collectives (#116, #118)
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
- Support pour punaiser les threads (voir \ref arcanedoc_general_launcher)

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

