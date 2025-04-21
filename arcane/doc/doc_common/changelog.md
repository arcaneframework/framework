# Nouvelles fonctionnalités {#arcanedoc_news_changelog}

[TOC]

Cette page contient les nouveautés de chaque version de %Arcane v3.X.X.

Les nouveautés successives apportées par les versions de %Arcane
antérieures à la version 3 sont listées ici : \ref arcanedoc_news_changelog20

___

## Arcane Version 3.15.3 (04 février 2025) {#arcanedoc_version3150}

### Nouveautés/Améliorations

- Ajoute service pour sub-diviser un maillage initial (\pr{1937}, \pr{1938})
- Commence support pour les maillages avec des mailles de
  plusieurs dimensions et les maillages non manifold (\pr{1922},
  \pr{1923}, \pr{1931}, \pr{1932}, \pr{1934}, \pr{1935}, \pr{1936},
  \pr{1943}, \pr{1944}, \pr{1945}, \pr{1948})
- Ajoute support expérimental pour générer les uniqueId() des faces et
  des arêtes à partir des uniqueId() des noeuds qui les composent
  (\pr{1851}, \pr{1920})
- Améliore support pour les maillages polyédriques (\pr{1846},
  \pr{1847}, \pr{1925})
- Ajoute propriété `compact-after-allocate` pour désactiver le
  compactage en d'allocation de maillage (\pr{1857})

### API Accélérateur

- Ajoute surchages avec \arcane{SmallSpan} des méthodes de \arcane{NumArray} et
  \arcane{MDSpan} pour la dimension 1.
- Autorise la création concurrente de \arcaneacc{RunQueue} et
  \arcaneacc{RunCommand} (\pr{1842})
- N'utilise pas l'API accélérateur pour la création de
  \arcanemat{ComponentItemVector} si le multi-threading est actif
  (\pr{1840})
- Utilise l'API accélérateur pour initialiser
  \arcanemat{AllCellToAllEnvCell} (\pr{1839})
- Appelle automatiquement `cudaMemAdvise()` sur les zones allouées par
  `cudaMallocManaged()` si la variable d'environnement
  `ARCANE_CUDA_MEMORY_HINT_ON_DEVICE` est positionnée (TODO: faire
  doc) (\pr{1838})
- Ajoute support SYCL pour le partitionnement de liste en deux parties
  (\pr{1858})
- Ajoute version multi-thread des algorithmes de scan
  (\arcaneacc{GenericScan}) et de filtrage
  \arcaneacc{GenericFiltering} (\pr{1878}, \pr{1880}).
- Ajoute support pour \arcane{ParallelLoopOptions::grainSize()} pour
  les boucles multi-dimensionnelles (\pr{1890})
- Ajoute implémentation spécifique de \arcaneacc{GenericFiltering}
  quand l'entrée et la sortie utilisent le même tableau (\pr{1882})
- Rend thread-safe la gestion des observers de \arcane{TaskFactory}
  (\pr{1883})
- Ajoute support pour copier les instances de
  \arcaneacc{RunQueueEvent} avec une sémantique par référence
  (\pr{1895})
- Rend toujours thread-safe les appels à \arcaneacc{makeQueue()}
  (\pr{1898})
- Supprime l'implémentation des réductions utilisant les opérations
  atomiques. Elle ne sont pas disponibles pour tous les types et sont
  moins performances que l'implémentation utilisant un arbre
  (\pr{1908})
- Rend `const` les méthodes de \arcane{StandaloneAcceleratorMng} (\pr{1912})
- Optimisations et refactoring divers (\pr{1843}, \pr{1845},
  \pr{1854}, \pr{1855}, \pr{1860}, \pr{1875}, \pr{1881}, \pr{1891})

### Changements

- Refactoring divers dans \arcane{builtInGetValue()} pour éviter
  certaines copies et utiliser `std::from_chars()` pour le type
  `double`(\pr{1903}, \pr{1909}, \pr{1910}, \pr{1911}, \pr{1918},
  \pr{1919}, \pr{1949}, \pr{1956})
- Déplace les méthodes `isNearlyZero()`, `normL2()` et
  `squareNormL2()` des classes numériques \arcane{Real2},
  \arcane{Real3}, \arcane{Real2x2} et \arcane{Real3x3} dans le
  namespace Arcane::math (\pr{1871}).
- Renomme \arcanemat{ComponentItemLocalId} en
  \arcanemat{ConstituentItemLocalId} et ajoute deux classes
  \arcanemat{MatItemLocalId} et \arcanemat{EnvItemLocalId} pour typer
  fortement l'accès aux variables matériaux (\pr{1862})
- Utilise par défaut l'extension `mli2` au lieu de `mli` pour les fichiers généré
  par Lima. Le format `mli` est obsolète et ne doit plus être utilisé
  (\pr{1849}, \pr{1850})

### Corrections

- Corrige compilation avec ROCM version 5 (\pr{1852})
- Fixe à `0` au lieu de `1` le numéro du premier uniqueId() généré
  après une reprise lorsque le maillage est vide. Cela permet d'avoir
  la même numérotation avec ou sans reprise pour ce cas spécifique
  (\pr{1924})
- Corrige fuite mémoire pour les \arcaneacc{RunCommand} créées par
  des \arcaneacc{RunQueue} avec une priorité autre que la priorité par
  défaut (\pr{1927})
- Corrige mauvaises valeurs de retour dans \arcaneacc{GenericFilterer}
  et \arcaneacc{GenericPartitioner} lorque les tableaux d'entrées ont
  une taille de zéro (\pr{1929})
- Corrige appel manquant à `MPI_Group_free` lors de la création d'un
  \arcane{IParallelMng} issu d'un \arcane{IParallelMng} (\pr{1933})
- Recalcule correctement les informations de synchronisation après la
  création d'un patch AMR (\pr{1946})
- Corrige fuite mérmoire dans Aleph lors de la création de matrices
  Hypre, PETSc, ou Epetra (\pr{1947})
- Corrige débordement de tableau si on utilise le multi-threading avec
  un seul thread (\pr{1954})
- Corrige fuites mémoires dans certaines opérations sur le DOM
  (\pr{1965})

### Interne

- Ajoute classe \arcane{mesh::ItemsOnwerBuilder} pour calculer les
  propriétaires des faces à partir de celui des mailles (\pr{1861}).
- Utilise \arcane{IMesh} au lieu de \arcane{mesh::DynamicMesh} dans
  \arcane{mesh::MeshExchangeMng} (\pr{1841})
- Ajoute type spécifique `float` pour les utilitaires HDF5 (\pr{1837})
- Sépare l'instantiation explicite de certaines classes en plusieurs
  fichiers pour accélérer la compilation (\pr{1836}, \pr{1873})
- Commence support pour les types \arcane{Float128}, \arcane{Int128},
  \arcane{Float16}, \arcane{Float32}, \arcane{BFloat16} et
  \arcane{Int8} (\pr{1835}, \pr{1863}, \pr{1864}, \pr{1866},
  \pr{1867}, \pr{1868}, \pr{1869}, \pr{1870}, \pr{1914})
- Uniformise la gestion du profiling en multi-thread (\pr{1877})
- Implémente \arccore{IMemoryAllocator3} à la place de
  \arccore{IMemoryAllocator} pour \arcane{SmallArray} (\pr{1889})
- Utilise l'injection de dépendence pour créér les services gérant les
  threads dans \arcane{Application} (\pr{1900}, \pr{1902})
- Supprime méthode obsolète \arcane{mesh::DynamicMesh::addFace()} (\pr{1930})
- Ajoute observables pour garantir que les vues sur les variables
  internes à \arcane{mesh::ItemFamily} sont toujours valides suite à
  un changement externe (\pr{1953})

### Compilation et Intégration Continue (CI)

- Supprime la possiblité d'installer %Arccore séparément de
  %Arcane. Cela permet de garantir que %Arccore et %Arcane seront
  toujours cohérents (\pr{1865}).
- Simplifie la gestion des dépendances des composantes pour accélérer
  la compilation (\pr{1874}).
- Mise à jour du workflow `compile-all-vcpkg` avec la version 2024.12
  (\pr{1894}).
- Ajoute workflow `compile-all-vcpkg` avec ubuntu 24.04 (\pr{1896})
- Utilise `.Net 8` pour certains workflows de `compile-all-vcpkg` et
  ajoute tests du wrapper C# (\pr{1897})
- Ajoute workflow pour ROCM version 6.3.1 et 5.7.1 (\pr{1899})
- Utlise `.mli2` au lieu de `.mli` pour fichiers de maillage Lima
  (\pr{1904})
- Mise à jour des images pour le CI (\pr{1905}, \pr{1915})
- Ajoute workflow avec `-fsanitize=address` (\pr{1940}, \pr{1955})

### Arccore

- Déplace les classes de %Arccore du namespace Arccore vers le
  namespace Arcane (\pr{1963}, \pr{1966})
- Supprime méthodes obsolètes de \arccore{IMemoryAllocator}
  (\pr{1959})

### Alien

- Supprime utilisation de `boost::remove_reference` et le remplace par
  `std::remove_reference` (\pr{1844})
- Corrige compilation quand PETSc est compilé avec le support de MUMPS
  (\pr{1917})

## Arcane Version 3.14.15 (11 décembre 2024) {#arcanedoc_version3140}

### Nouveautés/Améliorations

- Ajoute possibilité de choisir dans le jeu de données la version
  utilisée pour le calcul des identifiants des faces (\pr{1826})
- Ajoute support expérimental pour la lecture des fichiers MSH version
  4.1 au format binaire (\pr{1824})
- Ajoute mécanisme pour supprimer les mailles fantômes des mailles
  raffinées (\pr{1716}, \pr{1785}, \pr{1818})
- Ajoute méthodes \arcane{ICartesianMesh::coarseZone2D()} et
  \arcane{ICartesianMesh::coarseZone3D()} pour déraffiner un bloc d'un
  patch AMR (\pr{1697})
- Ajoute support dans l'AMR par patch pour déraffiner le maillage
  initial (\pr{1678}, \pr{1774})
- Ajoute possibilité de choisir la version de numérotation des faces
  dans le jeu de données (\pr{1674})
- Ajoute une nouvelle implémentation de tableau associatif
  \arcane{impl::HashTableMap2}. Cette implémentation est pour
  l'instant interne à Arcane. La nouvelle implémentation est plus
  rapide, consomme moins de mémoire que l'ancienne
  (\arcane{HashTableMapT}). Elle a aussi une API compatible avec
  `std::unordered_map` (\pr{1638}, \pr{1639}, \pr{1640}, \pr{1650})
- Ajoute support de l'écriture MPI/IO par bloc dans l'écrivain
  `VtkHdfV2PostProcessor` (\pr{1648}, \pr{1649})
- Ajoute support des maillage polyédriques (\pr{1619}, \pr{1620},
  \pr{1496}, \pr{1746}, \pr{1747}, \pr{1748}, \pr{1761}, \pr{1762},
  \pr{1795}, \pr{1816}, \pr{1829})
- Ajoute méthode utilitaire
  \arcane{MeshUtils::computeNodeNodeViaEdgeConnectivity()} pour créer
  les connectivités noeud-noeud via les arêtes (\pr{1614})
- Ajoute mécanisme permettant de vérifier que tous les rangs
  synchronisent bien la même variable. Cela se fait en positionnant la
  variable d'environnement `ARCANE_CHECK_SYNCHRONIZE_COHERENCE`
  (\pr{1604})
- Ajoute support pour positionner le nom de débug pour
  \arcane{NumArray} (\pr{1590})
- Ajoute possibilité d'afficher la pile d'appel via le debugger
  lorsqu'un signal (SIGSEGV, SIGBUS, ...) est recu (\pr{1573})
- Ajoute nouvelle version du déraffinement initial qui conserve la
  numérotation initiale et garanti la même numérotation quel que soit
  le découpage (\pr{1557})
- Ajoute implémetation de \arcane{IParallelMng::scan()} pour le mode
  mémoire partagé et hybride (\pr{1548})

### API Accélérateur

- Ajoute méthode \arcaneacc{Runner::deviceMemoryInfo()} pour récupérer
  la mémoire libre et la mémoire totale d'un accélérateur (\pr{1821})
- Affiche plus de propriétés lors de la description de l'accélérateur
  utilisé (\pr{1819})
- Ajoute possibilité de changer la ressource mémoire associée à
  \arcane{MemoryUtils::getDefaultDataAllocator()} (\pr{1808})
- Utilise `const RunQueue&` ou `const RunQueue*` au lieu de
  `RunQueue&` et `RunQueue*` pour certains arguments des méthodes qui
  utilisent des \arcaneacc{RunQueue} (\pr{1798})
- Interdit d'utiliser deux fois une même instance de
  \arcaneacc{RunCommand}. Il est temporairement possible d'autoriser
  cela en positionnant la variable d'environnement
  `ARCANE_ACCELERATOR_ALLOW_REUSE_COMMAND` à `1` (\pr{1790})
- Ajoute classe \arcaneacc{RegisterRuntimeInfo} pour passer des arguments
  pour l'initialisation du runtime accélérateur (\pr{1766})
- Ajoute méthodes pour récupérer de manière propre l'implémentation
  native correspondante à \arcaneacc{RunQueue} et rend obsolète
  \arcaneacc{RunQueue::platformStream()} (\pr{1763})
- Ajoute fichiers d'en-ête pour les algorithmes avancés dont le nom
  est identique à celui de la classe (\pr{1757})
- Ajoute implémetation de RUNCOMMAND_MAT_ENUMERATE() pour
  \arcanemat{AllEnvCell} (\pr{1754})
- Ajoute méthodes \arcaneacc{IAcceleratorMng::runner()} et
  \arcaneacc{IAcceleratorMng::queue()} qui retournent des instances au
  lieu de pointeurs sur \arcaneacc{Runner} et \arcaneacc{RunQueue}
  (\pr{1752})
- Rend privées les méthodes de construction de \arcaneacc{RunQueue} et
  \arcaneacc{RunCommand}. Il faut passer par \arcaneacc{makeQueue} ou
  \arcaneacc{makeCommand} pour créer des instances de ces classes
  (\pr{1752})
- Optimisations diverses dans la mise à jour des constituants
  (\arcanemat{MeshMaterialModifier}) (\pr{1559}, \pr{1562}, \pr{1679},
  \pr{1681}, \pr{1682}, \pr{1683}, \pr{1687}, \pr{1689}, \pr{1690},
  \pr{1691}, \pr{1704}, \pr{1720}, \pr{1729}, \pr{1731}, \pr{1733},
  \pr{1738}, \pr{1739}, \pr{1741}, \pr{1742}, \pr{1831})
- Ajoute classe \arcaneacc{ProfileRegion} pour spécifier une région
  pour le profilage sur accélérateur (\pr{1695}, \pr{1734}, \pr{1768})
- Ajoute une classe interne \arcane{impl::MemoryPool} pour conserver
  une liste de blocs alloués. Ce mécanisme ne fonctionne actuellement
  qu'avec l'implémentation CUDA. Il n'est pas actif par défaut (voir
  \ref arcanedoc_acceleratorapi_memorypool) (\pr{1684}, \pr{1685},
  \pr{1686}, \pr{1699}, \pr{1703}, \pr{1724}, \pr{1725}, \pr{1726},
  \pr{1776})
- Ajoute algorithme de partitionnement \arcaneacc{GenericPartitioner}
  (\pr{1713}, \pr{1717}, \pr{1718}, \pr{1721}, \pr{1722})
- Uniformise les constructeurs des algorithmes accélérateurs pour
  prendre une \arcaneacc{RunQueue} en argument (\pr{1714}
- Utilise l'API accélérateur pour la création des
  \arcanemat{EnvCellVector} et \arcanemat{MatCellVector} (\pr{1710},
  \pr{1711})
- Alloue la mémoire des \arcane{ItemVector} via l'UVM. Cela permet de
  rendre accessible sur accélérateurs les éléments de cette classe
  (\pr{1709})
- Ajoute algorithme de tri \arcaneacc{GenericSorter} (\pr{1705},
  \pr{1706})
- Ajoute possibilité d'utiliser la mémoire hôte pour la valeur de
  retour de \arcaneacc{GenericFilterer} (\pr{1701})
- Ajoute synchronisation explicite pour \arcane{DualUniqueArray}
  (\pr{1688})
- Ajoute possibilité de récupérer sur accélérateur la `backCell()` et
  `frontCell()` d'une face (\pr{1607})
- Ajoute implémentation sur accélérateur de
  \arcanemat{IMeshMaterialMng::synchronizeMaterialsInCells()}. Cette
  implémentation est activée si la variable d'environnement
  `ARCANE_ACC_MAT_SYNCHRONIZER` est positionnée à `1` (\pr{1584})
- Ajoute possibilité d'utiliser le mécanisme ATS avec CUDA pour
  l'allocation (\pr{1576}) 
- S'assure que les messages affichés par l'intégration avec CUPTI ne
  sont pas découpés en multi-thread (\pr{1571}) 
- Ajoute possiblité d'afficher et d'interompre le profilage
  (\pr{1561}, \pr{1569})
- Ajoute macro RUNCOMMAND_SINGLE() pour effectuer une commande
  accélérateur sur une seule itération (\pr{1565})
- Alloue par défaut la mémoire managée (UVM) sur un multiple de la
  taille d'une page système (\pr{1564})
- Ajoute détection et affichage des 'Page Faults' dans l'intégration
  CUPTI (\pr{1563})
- Choisit automatiquement le device associé à un rang MPI sur un noeud
  selon un mécanisme round-robin (\pr{1558})
- Ajoute support pour compiler pour CUDA avec le compilateur
  Clang (\pr{1552})

### Changements

- Renomme \arcane{NumArray::resize()} en
  \arcane{NumArray::resizeDestructive()} et \arcane{NumArray::fill()}
  en \arcane{NumArray::fillHost()} (\pr{1809})
- Déplace les méthodes de \arcane{platform} gérant les allocateurs dans
  \arcane{MemoryUtils} et rend obsolètes certaines d'entre
  elles (\pr{1806}, \pr{1817})
- Dans le service \arcane{ArcaneCaseMeshService}, Initialise les variables
  spécifiées dans le jeu de données après application du
  partitionnement au lieu de le faire avant (\pr{1751})
- Utilise tous les rangs pour la lecture en parallèle des fichiers
  GMSH. Cela permet d'éviter d'avoir des partitions vides par la suite
  ce qui n'est pas supporté par ParMetis (\pr{1735})
- Rend obsolète \arcane{NumArray::span()} and
  \arcane{NumArray::constSpan()} \pr{1723}
- Supprime la classe obsolète `Filterer` (\pr{1708})
- Rend explicite les constructeurs des variables qui prennent un
  \arcane{VariableBuildInfo} en argument (\pr{1693})
- Change le comportement par défaut de \arcane{ILoadBalanceMng} pour
  ne pas prendre en compte le nombre de variables allouées et ainsi
  avoir un partitionnement qui dépend uniquement du maillage (\pr{1673})
- Supprime le post-traitement au format `EnsighHdf`. Ce format était
  expérimental et n'est plus disponible dans les versions récentes de
  Ensight (\pr{1668})
- Écrit les arêtes et les connectivités associées dans
  \arcane{MeshUtils::writeMeshConnectivity()} (\pr{1651})
- Rend obsolète \arcane{mesh::ItemFamily::infos()} (\pr{1647})
- Lève une exception si on tente de lire un maillage déjà alloué
  (\pr{1644})
- Rend obsolete \arcane{Item::internal()} (\pr{1627}, \pr{1642})
- Rend obsolète \arcane{IItemFamily::findOneItem()} (\pr{1623})
- Lève une exception (au lieu d'un avertissement) si le partitionneur
  spécifiée par l'utilisateur n'est pas disponible (\pr{1635})
- Supprime possibilité de compiler avec `mono` (\pr{1583})
- Rend obsolète la version 1 de \arcane{CartesianMeshCoarsening} (\pr{1580})

### Corrections

- Utilise les mêmes valeurs pour \arccore{ISerializer::eDataType} et
  \arcane{eDataType} (\pr{1827}, \pr{1828})
- Corrige erreur de compilation avec CRAY MPICH (\pr{1778})
- Corrige compilation avec les version 2.13+ de libxml2 (\pr{1715})
- Positionne correctement le communicateur de
  \arccore{MessagePassing::MessagePassingMng} associé à l'implémentation séquentielle
  \arcane{SequentialParallelMng}. Auparavant cela n'était fait que
  pour l'implémentation MPI (\pr{1661})
- Dans le lecteur VTK, conserve après la destruction du lecteur les
  variables lues dans le fichier (\pr{1655})
- Ne détruit en fin d'exécution pas l'instance utilisée dans
  \arcane{ArcaneLauncher::setDefaultMainFactory()}. Cette instance
  n'est pas forcément allouée via `new` (\pr{1643})
- Corrige mauvais comportement dans la gestion des sous-maillages avec
  les nouvelles connectivités (\pr{1636})
- Corrige non prise en compte de la variable d'environnemnt
  `ARCANE_DOTNET_USE_LEGACY_DESTROY` (\pr{1570})
- Ajoute appel manquant à \arcane{ArcaneMain::arcaneFinalize()}
  lorsqu'on utilise `.Net` (\pr{1567})
- Corrige mauvaise prise en compte des options pour l'outil
  `arcane_partition_mesh` (\pr{1555})

### Interne

- Ajoute variable d'environnement `ARCANE_CUDA_MEMORY_HINT_ON_DEVICE`
  pour appeler automatiquement `cudaMemAdvise()` sur la mémoire
  unifiée pour la forcer à aller sur un allocateur spécifique (\pr{1833})
- Appelle \arccore{Array::resizeNoInit()} au lieu de
  \arccore{Array::resize()} lors du redimensionnement des variables
  matériaux. L'initialisation est faite par la suite et cela évite
  d'initialiser deux fois (\pr{1832})
- Renomme `Adjency` en `Adjacency` dans certaines classes et méthodes
  (\pr{1823})
- Ajoute option de la ligne de commande `-A,UseAccelerator=1` pour
  spécifier qu'on souhaite une exécution sur accélérateur. Le backend
  utilisé sera automatiquement choisi en fonctions de la configuration
  lors de la compilation (\pr{1815})
- Ajoute typedef `AlephInt` qui sert pour spécifier les index
  des lignes et colonnes des matrices et vecteurs. Pour l'instant ce
  type est `int` mais lorsque le support 64 bit sera actif il sera
  `long int` ou `long long int` (\pr{1770})
- Libère les buffers de sérialisation dès que possible lors de
  l'équilibrage de charge (\pr{1744}, \pr{1756})
- Ajout d'un service expérimental permettant de subdiviser un maillage
  lors de l'initialisation (\pr{1606}, \pr{1728})
- Rend publique les classes \arcane{ItemBase} et
  \arcane{MutableItemBase} \pr{1740}
- Ajoute méthode interne de finalisation de l'API accélérateur. Cela
  permet d'afficher des statistiques et de libérer les ressoures
  associées (\pr{1727}
- Ajoute tests pour l'utilisation de plusieurs critères avec le
  partitionnement avec plusieurs maillages (\pr{1719}, \pr{1772})
- Dans \arcane{BitonicSort}, n'alloue pas les tableaux des rangs et
  des index s'ils ne sont pas utilisés (\pr{1680})
- Utilise une nouvelle implémentation de table de hashage pour `ItemInternalMap`.
  Cette implémentation est active par défaut mais il est possible
  d'utiliser l'ancienne en positionant l'option
  `ARCANE_USE_HASHTABLEMAP2_FOR_ITEMINTERNALMAP` à `OFF` lors de la
  configuration (\pr{1611}, \pr{1617}, \pr{1622}, \pr{1624},
  \pr{1625}, \pr{1628}, \pr{1629}, \pr{1631}, \pr{1677}, \pr{1745})
- Nettoyage et refonte du partitionnement avec ParMetis pour utiliser
  \arcane{IParallelMng} au lieu d'appeler MPI directement (\pr{1662},
  \pr{1665}, \pr{1667}, \pr{1671})
- Ajoute support pour créér un sous-communicateur à la manière de
  `MPI_Comm_Split` (\pr{1669}, \pr{1672})
- Nettoyage et refactoring des classes gérant la numérotation des
  `uniqueId()` des arêtes (\pr{1658})
- Utilise un pointeur sur \arcane{mesh::DynamicMeshKindInfos} à la
  place d'une instance dans \arcane{mesh::ItemFamily} (\pr{1646})
- Ajoute dans \arcane{ICaseMeshService} la possibilité d'effectuer des
  opérations après le partitionnement (\pr{1637})
- Ajoute API interne pour \arcane{IIncrementalItemConnectivity}
  (\pr{1615}, \pr{1618}, \pr{1626})
- Ajoute point d'entrée de type `build` pour le service de test
  unitaire (\pr{1613})
- Ajoute possibilité dans \arcane{IIncrementalItemConnectivity} de
  notifier de l'ajout de plusieurs entités à la fois (\pr{1610})
- Ajoute méthode pour récupérer la pile d'appel via LLVM ou GDB. Cela
  permet notamment d'avoir les numéros de ligne dans la pile d'appel
  (\pr{1572}, \pr{1597}, \pr{1616}) TODO INDIQUER METHOD
- Ajoute support expérimental pour KDI (\pr{1594}, \pr{1595},
  \pr{1599})
- Optimisations diverses dans le service `VtkHdfV2PostProcessor`
  (\pr{1575}, \pr{1600})
- [EXPÉRIMENTAL] Ajoute nouvelle version de calcul des `uniqueId()` des
  faces basée sur les `uniqueId()` des noeuds qui la compose. Cela ne
  fonctionne qu'en séquentiel pour l'instant (\pr{1550})
- Ajoute fonction pour générer un `uniqueId()` à partir d'une liste de
  `uniqueId()` (\pr{1549})
- Optimise le calcul de la version 3 des mailles fantômes (\pr{1547})

### Compilation et Intégration Continue (CI)

- Utilise un wrapper de `dotnet` pour la compilation. Ce wrapper
  s'assure qu'on ne va pas modifier le HOME de l'utilisateur ce qui
  pouvait poser des problèmes de verrou lorsque plusieurs instances de
  `dotnet` se lancent en même temps (\pr{1789}, \pr{1791}, \pr{1792},
  \pr{1830})
- Ajoute possibilité d'ajouter des bibliothèques lors de l'édition de
  lien de `arccore_message_passing_mpi`. Cela permet de garantir que
  certaines bibliothèques seront bien ajoutées à l´édition de lien et
  est notamment utilisé pour le support du MPI 'GPU-Aware' avec CRAY
  MPICH (\pr{1786})
- Ajoute workflow 'ubuntu 22.04' pour les dockers IFPEN (\pr{1781})
- Ajoute variable CMake `ARCCON_NO_TBB_CONFIG` pour forcer à ne pas
  utiliser le fichier de configuration CMake pour les TBB (\pr{1779})
- Ajoute tests de protection/reprise pour le déraffinement (\pr{1707})
- Corrige erreur de compilation lorsque PETSc n'est pas compilé avec
  MUMPS (\pr{1694})
- Ajoute test du lecteur VTK avec des propriétés (\pr{1656},
  \pr{1659})
- Ajoute support pour une somme de contrôle de la connectivité dans
  les tests de maillage (\pr{1654})
- Écrit les fichiers de sortie des tests dans le répertoire des tests
  pour permettre de les lancer en parallèle (\pr{1541}, \pr{1653})
- Mise à jour des images IFPEN 2021 (\pr{1542}, \pr{1579}, \pr{1587},
  \pr{1588}, \pr{1592}, \pr{1593}, \pr{1598})
- Supprime version interne de `hostfxr.h` et
  `coreclr_delegates.h`. Ces fichiers sont maintenant dans le SDK
  dotnet (\pr{1591})
- Corrige détection et configuration de FlexLM (\pr{1602}, \pr{1630})
- Supprime les répertoires de sortie des tests après exécution pour
  réduire l'empreinte sur le stockage (\pr{1581})
- Lance les tests de CI en parallèle pour plusieurs workflow (\pr{1553})
- Refonte du système d'action du CI et des images pour le rendre plus
  souple (\pr{1545})

### Arccore

- Ajoute conversions de \arccore{SmallSpan<std::byte>} vers et depuis
  \arccore{SmallSpan<DataType>} {\pr{1731})
- Corrige mauvaise valeur pour la taille passée en arguments de
  \arccore{IMemoryAllocator::deallocate()} (\pr{1702})
- Ajoute argument \arcaneacc{RunQueue} pour
  \arccore{MemoryAllocationArgs} and
  \arccore{MemoryAllocationOptions}. Cela n'est pas utilisé pour
  l'instant mais cela permettra par la suite de faire des allocation
  spécifiques à une file d'exécution (\pr{1696})
- Ajoute support pour les allocateurs spécifiques pour
  \arccore{SharedArray} (\pr{1692})
- Rend `inline` les constructeurs et destructeurs de
  \arccore{MemoryAllocationOptions} (\pr{1664})
- Ajoute méthode pour positionner le communicateur associé à
  \arccore{MessagePassing::MessagePassingMng} (\pr{1660})

### Axlstar

- Remplace `std::move()` par `std::forward()` dans la génération
  de certaines méthodes (\pr{1773})
- Génère via l'interface spécifiée les méthodes pour récupérer les
  fonctions associées aux options des jeux de données (\pr{1601})
- Supprime la date dans les fichiers générés afin de ne pas les
  modifier s'ils sont re-générés à l'identique (\pr{1797}

### Alien

- Ajoute pluggin basé sur la bibliothèque
  [composyx](https://gitlab.inria.fr/composyx/composyx) (\pr{1801})
- Utilise un répertoire de sortie différent pour chaque test afin
  qu'on puisse les lancer en parallèle (\pr{1775})
- Corrige sorties listings pour certains tests (\pr{1765})
- Corrige sorties listings pour le backend IFPSolver (\pr{1730})
- Corrige erreur d'exécution lorsqu'on utilise l'implémentation
  séquentielle de \arcane{IParallelMng} (\pr{1666})
- Utilise un nom de fichier unique pour les fichiers de sortie des
  tests. Cela permet de les lancer en parallèle (\pr{1663})
- Récupère le communicateur MPI via
  \arccore{MessagePassing::MessagePassingMng::communicator()} (\pr{1657})
- Ajoute support accélérateur pour certaines parties (\pr{1632},
  \pr{1634})
- Corrige initialisation avec les versions récentes (2.27+) de Hypre
  (\pr{1603})
- Ajoute support pour le solver SPAI via PETSc (\pr{1578})
- Ajoute support pour les sorties au format 'Matrix Market' avec PETSc
  (\pr{1577})

## Arcane Version 3.13.08 (19 juillet 2024) {#arcanedoc_version3130}

### Nouveautés/Améliorations

- Ajoute support expérimental pour créer une variable du maillage sans
  référence. Il faut pour cela l'initialiser avec une instance de
  \arcane{NullVariableBuildInfo} (\pr{1510}).
- Ajoute support pour l'équilibrage de charge dynamique avec plusieurs
  maillages (\pr{1505}, \pr{1515}).
- Ajoute support pour les synchronisations sur un sous-ensemble des
  entités fantômes (\pr{1468}, \pr{1484}, \pr{1486}).
- Amélioration et passage en anglais du README (\pr{1466}).
- Ajoute support expérimental de l'AMR par patch (\pr{1413}).
- Débute le support pour intégrer du code python lors de
  l'exécution. Ce support est considéré comme expérimental
  (\pr{1447}, \pr{1449}, \pr{1454}, \pr{1456},\pr{1461}, \pr{1462},
  \pr{1471}, \pr{1479}, \pr{1493}, \pr{1494}, \pr{1499}, \pr{1501},
  \pr{1502}, \pr{1513}, \pr{1522}, \pr{1525}).

#### API Accélérateur

- Ajoute accès accélérateur à certaines méthodes (\pr{1539})
- Optimise les noyaux de calcul pour la gestion des mise à jour des
  matériaux (\pr{1421}, \pr{1422}, \pr{1424}, \pr{1426}, \pr{1431},
  \pr{1432}, \pr{1434}, \pr{1437}, \pr{1440}, \pr{1441}, \pr{1443},
  \pr{1458}, \pr{1472}, \pr{1473}, \pr{1474}, \pr{1488}, \pr{1489})
- Mise à jour de la documentation (\pr{1483}, \pr{1492}, \pr{1508})
- Ajoute arguments templates pour spécifier la taille des entiers
  utilisés pour l'indexation (`Int32` ou `Int64`) (\pr{1398}).
- Ajoute support pour les réductions version 2 dans
  RUNCOMMAND_MAT_ENUMERATE() (\pr{1390}).
- Continue travail de portage SYCL (\pr{1389}, \pr{1391}, \pr{1393},
  \pr{1396}).
- Améliore la gestion du padding SIMD pour \arcane{ItemGroup} pour ne
  le faire qu'à la demande et le faire sur accélérateur si possible
  (\pr{1405}, \pr{1523}, \pr{1524}).

### Changements

- Supprime la composante gérant les expressions vectorielles sur les
  variables car elle n'est plus utilisée depuis longtemps (\pr{1537})
- Utilise `clock_gettime()` au lieu de `clock()` pour mesurer le temps
  CPU utilisé (\pr{1532})
- Utilise par défaut la version parallèle du lecteur *MSH*. Il est
  toujours possible d'utiliser la version séquentielle en positionnant
  la variable d'environnement `ARCANE_USE_PARALLEL_MSH_READER` à `0`
  (\pr{1528}).
- Supprime le support de la compilation du C# avec `mono`. Il faut
  maintenant obligatoire utiliser `dotnet` (version 6 au minimum)
  (\pr{1470}).
- Renomme \arcane{ArrayIndex} en \arcane{MDIndex} (\pr{1397}).
- Ajoute possibilité de récupérer une \arcanemat{ComponentCell} à
  partir d'un \arcanemat{ComponentItemVectorView} (\pr{1478})

### Corrections

- Corrige divers bugs dans la gestion des graphes du maillage (\pr{1536},
  \pr{1535})
- Corrige *situations de concurrence* potentielles en mode multi-thread
  (\pr{1467}, \pr{1534}, \pr{1533}, \pr{1529})
- Ajoute support des maillages 1D dans le format VtkHdfV2 (\pr{1519})
- Ré-active l'envoi de signaux dans un handler de signal. Cela avait
  été implicitement désactivé lors de l'utilisation de `sigaction` au
  lieu de `sigset` pour positionner les signaux (\pr{1518})
- Corrige lecture des groupes dans le format `MSH` lorsqu'il y a
  plusieurs groupes par entité physique (\pr{1507}, \pr{1509}).
- Ajoute tests de vérification dans l'utilisation des variables
  partielles (\pr{1485})
- Indique explicitement le type sous-jacent pour \arcane{eDataType}
  (\pr{1418})
- Corrige non-réutilisation de la mémoire allouée pour les réductions
  introduite lors de la création de la version 2 des réductions
  (\pr{1439}).
- Corrige mauvais calcul potentiel du nombre d'entités à sérialiser ce
  qui pouvait se traduire par un débordement de tableau (\pr{1423}).
- Ne conserve pas les propriétés de changement de comportement dans
  \arcanemat{MeshMaterialModifier} pour éviter des incohérences
  (\pr{1453}).

### Interne

- Utilise une seule instance de \arcane{DofManager} (\pr{1538})
- Supprime plusieurs utilisation de la Glib (\pr{1531})
- Supprime certaines utilisations obsolètes de \arcane{ItemInternal}
  dans le wrapper C# (\pr{1517}, \pr{1520})
- Améliore la gestion de la compilation C# et des dépendances SWIG en
  utilisant un seul projet et corrige divers problèmes de dépendance
  dans la gestion CMake (\pr{1433}, \pr{1407}, \pr{1410}, \pr{1411},
  \pr{1412}, \pr{1414}, \pr{1425}, \pr{1427}, \pr{1428}, \pr{1429},
  \pr{1455}, \pr{1480}, \pr{1487}, \pr{1495}, \pr{1497}, \pr{1498})
- Améliorations diverses dans la gestion des maillages polyédriques
  (\pr{1435}, \pr{1436}, \pr{1438}, \pr{1463})

### Compilation et Intégration Continue (CI)

- Ajoute tests python dans certains workflows (\pr{1448}, \pr{1526}).
- Mise à jour des images ubuntu (\pr{1417}, \pr{1442}, \pr{1503}).
- Active `ccache` pour le workflow `codecov` (\pr{1464}).
- Passage à la version `vcpkg` 2024.04 pour le workflow
  `compile-all-vcpkg` (\pr{1450}).
- Utilise une variable CMake spécifique pour activer la couverture de
  test (\pr{1444}, \pr{1445}).
- Mise à jour des CI pour ne plus utiliser les actions github
  obsolètes (\pr{1416}).
- Mise à jour des CI pour les dockers IFPEN (\pr{1402}, \pr{1409})
- Ajoute possibilité d'utiliser des tests accélérateurs via googletest
  (\pr{1401}, \pr{1403}).
- Utilise explicitement le service d'échange de message séquentiel
  pour les tests séquentiels (\pr{1430}).

### Arccore

- Ajoute dans \arccore{AbstractArray} la possibilité de spécifier la
  localisation mémoire de la zone allouée (\pr{1803}, \pr{1804})
- Utilise \arccore{MessagePassing::MessageRank::anySourceRank()} au
  lieu du constructeur par défaut pour spécifier
  `MPI_ANY_SOURCE`. L'ancien mécanisme reste temporairement valide
  (\pr{1511}, \pr{1512}).
- Supprime `const` pour le type de retour de certaines méthodes de
  \arccore{ArrayIterator} pour qu'elles soient conformes à la norme
  C++ (\pr{1392})

### Axlstar

- Amélioration de la documentation générée à partir des fichiers AXL
  (\pr{1452})

### Alien

- Corrections diverses (\pr{1395}, \pr{1415}, \pr{1465}, \pr{1491},
  \pr{1500}, \pr{1506}, \pr{1514})
- Ajoute support de la norme infinie (\pr{1504})

## Arcane Version 3.12.18 (02 mai 2024) {#arcanedoc_version3120}

\note Avec cette version, il est nécessaire d'activer le C++20 pour pouvoir
utiliser l'API accélérateur. Pour cela, il faut positionner la
variable CMake `ARCCORE_CXX_STANDARD=20` lors de la configuration.  Il
faut alors au moins les versions GCC 11, Clang 16 ou Visual Studio
2022 ainsi que la version 3.26 de CMake.

### Nouveautés/Améliorations

- Ajoute possibilité de trier par uniqueId() croissant les faces et
  arêtes connectées aux noeuds (\pr{990}). Cela n'est pas actif par
  défaut pour compatibilité avec l'existant. Le tri permet d'avoir
  toujours le même ordre de parcours pour les faces et les arêtes des
  noeuds ce qui peut aider pour avoir des calculs répétables. Pour
  l'activer, il faut positionner la variable d'environnement
  `ARCANE_SORT_FACE_AND_EDGE_OF_NODE` à `1` ou utiliser le code
  suivant:
  ~~~{.cpp}
  Arcane::IMesh* mesh = ...;
  mesh->nodeFamily()->properties()->setBool("sort-connected-faces-edges",true);
  mesh->modifier()->endUpdate();
  ~~~
- Permet de modifier le numéro du patch parent lors de la
  re-numérotation via
  \arcane{CartesianMeshRenumberingInfo::setParentPatch()} (\pr{986}).
- Ajoute support pour le calcul incrémental des fonctions de hash
  (\pr{983}, \pr{984}).
- Ajoute implémentation de l'algorithme de hash `SHA1` (\pr{982}).
- Ajoute polique d'initialisation des données pour avoir le même
  comportement que les versions de %Arcane antérieures à la version 3
  (\pr{1017}).
- Ajoute classes \arcane{Vector2} et \arcane{Vector3} pour gérer des
  couples et triplets de types quelconques (\pr{1066}, \pr{1075})
- Ajoute classe \arcane{FixedArray} pour les tableaux de taille
  fixe. Cette classe est similaire à `std::array` mais initialise
  par défaut ses valeurs et permet la détection de débordement de
  tableau (\pr{1063})
- Ajoute support pour le dé-raffinement 3D et la renumérotation des
  entités du maillage lors du raffinement ultérieur (\pr{1061}, \pr{1062})
- Ajoute conversion de \arcanemat{ComponentCell} vers
  \arcanemat{MatCell}, \arcanemat{EnvCell} et \arcanemat{AllEnvCell}
  (\pr{1051}).
- Ajoute support pour sauvegarder des 'hash' des variables pour comparer
  rapidement si les valeurs sont différentes (\pr{1142}, \pr{1143}, \pr{1145},
  \pr{1147}, \pr{1150}, \pr{1152}, \pr{1155}, \pr{1156}, \pr{1158},
  \pr{1159}, \pr{1160})
- Ajoute lecture parallèle des fichiers au format MSH (\pr{1126},
  \pr{1136}, \pr{1137}, \pr{1138}, \pr{1139})
- Refonte de la gestion de \arcane{ITimeHistoryMng} pour pouvoir
  facilement gérer plusieurs maillages (\pr{1203}, \pr{1249},
  \pr{1256}, \pr{1260}, \pr{1369})
- Ajoute support expérimental pour l'AMR cartésien par patch en
  dupliquant les noeuds et les faces (\pr{1167}, \pr{1337}, \pr{1350},
  \pr{1351}, \pr{1353})
- Ajoute nouvelle méthode (méthode 4) de renumérotation des patchs AMR
  pour avoir la même numérotation entre un maillage raffiné et un
  maillage dé-raffiné puis à nouveau raffiné (\pr{1108}, \pr{1109})
- Active la fusion des noeuds en 3D (\pr{1361})
- Ajoute support pour l'indentation des fichiers XML via LibXml2 lors
  des sorties (\pr{1379})

#### API Accélérateur

- Rend nécessaire le C++20 pour l'API accélérateur (\pr{1020},
  \pr{1026}, \pr{1030}, \pr{1050}).
- **INCOMPATIBILITÉ**: Ajoute argument template dans \arcane{ExtentsV},
  \arcane{ArrayIndex} permettant de spécifier le type de l'index. Pour
  l'instant seul `Int32` est supporté mais par la suite `Int64` et
  `Int16` seront disponibles. Cette modification empêche de compiler
  du code utilisant l'ancienne valeur de la classe
  \arcane{ExtentsV}. Cette classe étant normalement interne à %Arcane
  cela ne devrait pas concerner beaucoup d'usages (\pr{1383})
- Ajoute méthode `fill()` asynchrone pour les variables du maillage
  (\arcane{ItemVariableArrayRefT} et \arcane{ItemVariableScalarRefT}) (\pr{991})
- Ajoute classe \arcaneacc{RunQueue::ScopedAsync} pour
  rendre temporairement asynchrone une file d'exécution (\pr{978}).
- Optimise la gestion mémoire de la classe
  \arcaneacc{Filterer} et ajoute des surchages (\pr{1022}, \pr{1023},
  \pr{1034}, \pr{1043}, \pr{1210}, \pr{1225}, \pr{1271})
- Ajoute support des opérations atomiques (\pr{1028}, \pr{1032}, \pr{1033})
- Ajoute support dans Aleph des versions de Hypre compilées avec le
  support des accélérateurs (\pr{1125})
- Ajoute support expérimental pour le pre-fetching automatique des
  valeurs des vues (\pr{1179}, \pr{1180}, \pr{1304})
- Ajoute algorithme de réduction directe via la classe
  \arcaneacc{GenericReducer} (\pr{1306}, \pr{1307})
- Ajoute algorithme de partitionnement de liste via la classe
  \arcaneacc{GenericPartitioner} (\pr{1217})
- Rend accessible sur accélérateurs certaines opérations
  mathématiques (\pr{1294})
- Ajoute surcharges pour l'algorithme `PrefixSum` ( \pr{1253}, \pr{1254})
- Ajoute vues sur les variables partielles (\pr{1299}, \pr{1308}, \pr{1311}, \pr{1313}) 
- Utilise l'API accélérateur lors de la mise à jour des matériaux via
  \arcanemat{MeshMaterialModifier} (\pr{1182}, \pr{1183}, \pr{1185}, \pr{1186},
  \pr{1196}, \pr{1204}, \pr{1205},\pr{1211}, \pr{1219}, \pr{1223},
  \pr{1224}, \pr{1226}, \pr{1227}, \pr{1230}, \pr{1233}, \pr{1235},
  \pr{1238}, \pr{1239}, \pr{1241}, \pr{1243}, \pr{1247}, \pr{1257},
  \pr{1258}, \pr{1263}, \pr{1268}, \pr{1283}, \pr{1284}, \pr{1285}, \pr{1287},
  \pr{1292}, \pr{1295}, \pr{1312}, \pr{1347})
- Utilise l'API accélérateur pour l'ajout et la suppression d'entités
  dans un \arcane{ItemGroup} (\pr{1288},\pr{1289}, \pr{1293}).
- Ajoute constructeur par défaut pour \arcaneacc{RunQueue}. Dans ce
  cas l'instance n'est pas utilisable tant qu'elle n'a pas été
  assignée (\pr{1282})
- Ajoute possibilité de copier les \arcaneacc{RunQueue} avec une
  sémantique par référence (\pr{1221})
- Ajoute support pour choisir une ressource mémoire par défaut pour
  une \arcaneacc{RunQueue} (\pr{1278})
- Ajoute mécanismes pour faciliter la gestion mémoire (\pr{1273}, \pr{1274})
- Ajoute possibilité de créér une vue sur des données en spécifiant
  une instance de \arcaneacc{RunQueue} (\pr{1269}).
- Ajoute méthodes de conversion de \arcane{NumArray} et
  \arcane{MDSpan} vers \arccore{SmallSpan} (\pr{1262})
- Ajoute support pour récupérer l'index de l'itérateur pour
  RUNCOMMAND_ENUMERATE() et RUNCOMMAND_MAT_ENUMERATE() (\pr{1270},
  \pr{1272})
- Ajoute support dans RUNCOMMAND_MAT_ENUMERATE() pour
  \arcanemat{MatCell} (\pr{1184})
- Ajoute fonction
  \arcane{VariableUtils::markVariableAsMostlyReadOnly()} pour marquer
  les varibles comme étant principalement en lecture (\pr{1206})
- Autorise l'utilisation de la mémoire
  \arcane{eMemoryRessource::HostPinned} lorsque le runtime
  accélérateur n'est pas défini. Dans ce cas on utilise
  \arcane{eMemoryRessource::Host} (\pr{1315})
- Ajoute méthode \arcane{MemoryUtils::getDeviceOrHostAllocator()} pour
  récupérer un allocateur sur l'accélérateur si un runtime
  accélérateur est disponible et sur l'hôte sinon (\pr{1364})
- Débute support pour un backend avec l'API SYCL. Ce backend n'est pas
  encore fonctionnel et n'est disponible que pour des tests internes
  (\pr{1318}, \pr{1319}, \pr{1320}, \pr{1323}, \pr{1324},
  \pr{1330},\pr{1334}, \pr{1345}, \pr{1355}, \pr{1363}, \pr{1365},
  \pr{1373}, \pr{1374}, \pr{1380}, \pr{1381})
- Débute nouvelle implémentation des réductions qui permettent de
  supporter l'API SYCL (\pr{1366}, \pr{1368}, \pr{1371}, \pr{1372},
  \pr{1377})

### Changements

- **INCOMPATIBILITÉ**: Supprime dans le fichier `StdHeader.h` la
  plupart des `using` sur les fonctions mathématiques (\pr{1370},
  \pr{1384}).
- Supprime classe interne `ComponentItemInternal` qui est
  remplacée par `ComponentItemBase` (\pr{1039}, \pr{1053}, \pr{1059},
  \pr{1172}, \pr{1181}, \pr{1187},\pr{1190}, \pr{1191}, \pr{1192},
  \pr{1193}, \pr{1194}, \pr{1195},\pr{1197}, \pr{1199}, \pr{1200},
  \pr{1335})
- Active toujours les connectivités incrémentales (\pr{1166})
- Rend obsolète \arcane{NumArray::s()}. Il faut utiliser à la place
  `operator()` ou `operator[]` (\pr{1035})
- N'initialise pas le runtime MPI si le service d'échange de message
  (`MessagePassingService`) est `Sequential` (\pr{1029}).
- Ne positionne pas par défaut la variable CMake `ARCANE_BUILD_TYPE`
  lors de la configuration (\pr{1004}).
- Ne filtre plus les assembly signées lors du chargement des plug-ins
  en C# (\pr{1114})
- Utilise \arccore{String} au lieu de `const String&` pour les valeurs
  de retour de \arcane{IVariable} et \arcane{VariableRef} (\pr{1134})
- Utilise par défaut la nouvelle version de la gestion des listes de
  message de sérialisation (\pr{1113})
- Utilise \arccore{Int16} au lieu de \arcane{Int8} pour l'identifiant
  des accélérateurs (\arcaneacc{DeviceId}) (\pr{1276})
- Utilise par défaut la version 2 de la gestion des listes d'entités
  par type dans les groupes (\arcane{ItemGroup}. Cela est utilisé par
  exemple dans \arcane{ItemGroup::applyOperation()}. La nouvelle
  version n'utilise plus des sous-groupes mais uniquement des listes
  et ne duplique par la mémoire si toutes les entités du groupe sont
  de même type (\pr{1174})
- Dans les vues, remplace \arcane{ViewSetter} par
  \arcane{DataViewSetter} et \arcane{ViewGetterSetter} par
  \arcane{DataViewGetterSetter} (\pr{1040})

### Corrections

- Corrige fuite mémoire avec les commandes asynchrones lorsqu'on
  n'appelle pas \arcaneacc{RunQueue::barrier()}. Cela ne
  devrait normalement pas arriver car il faut toujours appeler
  `barrier()` après une commande asynchrone. Maintenant, si des
  commandes sont actives lorsque la \arcanacc{RunQueue} est détruite une
  barrière implicite est effectuée (\pr{995})
- Corrige mauvais type de retour de
  \arcacc{MatItemVariableScalarOutViewT::operator[]()}
  qui ne permettait pas de modifier la valeur(\pr{981}).
- Corrige mauvaise prise en compte des propriétés lors de la création
  de variables matériaux dans certains cas (\pr{1012}).
- Corrige mauvaise valeur pour la dimension Z pour le dé-raffinement
  3D (\pr{1074})
- Corrige numérotation des nouvelles entités des maillages cartésiens après
  dé-raffinement 2D (\pr{1055}).
- Corrige divers problèmes dans la gestion de la classe
  \arcanemat{AllCellToAllEnvCellAccessor} (\pr{1071}, \pr{1133}, \pr{1188})
- S'assure lors de la construction d'une variable matériau que la
  variable globale associée est bien allouée (\pr{1056})
- Corrige mauvaise valeur de retour de `ItemInternal.Cell()` dans le
  wrapper C# (\pr{1131})
- Corrige la version 2 de la sérialisation lorsqu'on utilise
  `MPI_ANY_SOURCE` (\pr{1116})
- Corrige mise à jour des synchronisations manquante lorsqu'on
  dé-raffine mais qu'on ne raffine pas ensuite (\pr{1266})
- Corrige divers problèmes de compilation (\pr{1044}, \pr{1047},
  \pr{1280}, \pr{1309})
- Corrige potentielle mauvaise mise à jour du padding SIMD dans la
  mise à jour des \arcane{ItemGroup} (\pr{1165})
- Utilise la variable CMake `CMAKE_CUDA_HOST_COMPILER` pour spécifier
  le compilateur hôte associé à CUDA au lieu d'imposer l'option
  `-ccbin` du compilateur NVIDIA (\pr{1386})
- Corrige compilation avec ROCM 6.0 (\pr{1170})
- Corrige compilation avec Swig 4.2 (\pr{1098})
- Corrige compilation avec les versions `2.12.0` et ultérieure de LibXml2
  (\pr{1019}).

### Interne

- Utilise un compteur de référérence pour `RunQueueImpl` (\pr{996})
- Ajoute support expérimental pour changer des valeurs dans le jeu de donnée à
  partir de la ligne de commande (\pr{1038}).
- Ajoute nouveau mécanisme de calcul de hash pour les implémentation
  de \arcane{IData} (\pr{1036}).
- Utilise nouveau mécanisme de Hash pour la détection de la validité
  des valeurs des variables lors d'une reprise (\pr{1037}).
- Simplifie l'implémentation de \arcane{MDSpan} via l'utilisation du
  C++20 (\pr{1027})
- Déplace les fichiers d'implémentation de Aleph dans le code source
  au lieu de l'avoir dans les fichiers d'en-tête et corrige le système de build
  si on utilise Trilinos (\pr{1002}, \pr{1003})
- Ajoute classes pour spécifier des arguments supplémentaires pour les
  méthodes \arcane{IMeshModifier::addCells()} et
  \arcane{IMeshModifier::addFaces()} (\pr{1077})
- Supprime ancienne implémentation du générateur de maillage cartésien
  (\pr{1069})
- Utilise les classes \arcane{Int64x2}, \arcane{Int64x3},
  \arcane{Int32x2} et \arcane{Int32x3} au lieu de `std::array` pour
  gérer les informations de construction des maillages cartésiens
  (\pr{1067})
- Supprime avertissements coverity (\pr{1060}, \pr{1065}, \pr{1333},
  \pr{1336}, \pr{1378})
- Refonte du mécanisme interne de gestion des informations des entités
  matériaux pour rendre les propriétés accessibles sur accélérateur
  (\pr{1057}, \pr{1058}, \pr{1086})
- Ajoute support pour le format JSON pour \arcane{VariableDataInfo}
  (\pr{1148})
- Utilise \arccore{ReferenceCounterImpl} pour l'implémentation de
  \arccore{IThreadImplementation} et \arcane{IParallelDispatchT}
  (\pr{1132}, \pr{1127})
- Corrige divers avertissements de compilation (\pr{1130}, \pr{1163},
  \pr{1164}, \pr{1042}, \pr{1118}, \pr{1346})
- Améliore le portage Windows (\pr{1154}, \pr{1157}, \pr{1246})
- Ajoute méthode \arcane{IMesh::computeSynchronizeInfos()} pour
  forcer le recalcul des informations de synchronisation (\pr{1124})
- Désactive les exceptions flottantes en fin de calcul. Cela permet
  d'éviter des exceptions flottantes (FPE) dues aux exécutions
  spéculatives (\pr{1232})
- Ajoute classe expérimentatle \arcane{DualUniqueArray} pour gérer une
  vue double sur CPU et accélérateur (\pr{1215})
- Utilise la bonne valeur de la taille de page pour l'allocateur
  interne lié à CUDA (\pr{1198})
- Ajoute détection du mode GPU Aware de MPI pour ROCM (\pr{1209})
- Ajoute méthode \arcane{ITimeStats::resetStats()} pour remettre à
  zéro les statistiques temporelles de l'instance (\pr{1122}, \pr{1128})
- Ajoute implémentation de \arccore{IThreadImplementation} et de
  \arcane{AtomicInt32} en utilisant la bibliothèque standard du C++
  (\pr{1339}, \pr{1340}, \pr{1342}, \pr{1343})
- Corrige possible duplication des uniqueId() dans le test de fusion
  des frontières (\pr{1362})

### Compilation et Intégration Continue (CI)

- Mise à jour de l'image `U22_G12_C15_CU122` (\pr{1031})
- Supprime utilisation des cibles utilisant CUDA 11.8 ( \pr{1078})
- Mise à jour du worflow `vcpkg` avec la version `2023-12-12`
  (\pr{1076})
- Utilise le C++20 pour le workflow `compile-all-vcpkg` (\pr{1048})
- Positionne par défaut `CMAKE_BUILD_TYPE` à `Release` si cette
  variable n'est pas spécifiée lors de la configuration (\pr{1149})
- Ajoute macro `ARCANE_HAS_ACCELERATOR` si %Arcane est compilé avec le
  support d'un accélérateur (\pr{1123})
- Ajoute support pour utiliser 'googletest' avec des cas MPI (\pr{1092})
- Supprime utilisation de Clang13 dans le CI (\pr{1083})
- Modifie le workflow IFPEN pour compiler directement tous les
  composants de `framework` (\pr{1214}, \pr{1281})
- Désactive les tests en cours si on publie une modification dans la
  branche liée au 'pull-request' (\pr{1250})
- Ajoute workflow pour IFPEN 2021 avec les versions 7, 8 et 9 de RHEL
  (\pr{1255})
- Met à jour certaines actions obsolètes (\pr{1218},\pr{1222})
- Utilise le C++20 pour certains workflow (\pr{1189})

### Arccore

- Ajoute support pour des destructeurs particuliers dans
  \arccore{ReferenceCounterImpl} (\pr{1068})
- Ajoute nouvelle implémentation de \arccore{ReferenceCounterImpl} qui
  n'appelle pas directement l'opérateur `operator delete`. Cela permet
  de détruire l'instance de manière externe (\pr{989}, \pr{1080},
  \pr{1081}, \pr{1120}, \pr{1161}, \pr{1162}).
- Autorise l'utilisation de \arccore{ReferenceCounterImpl} sans avoir
  besoin d'une classe interface (\pr{1121})
- Utilise `std::atomic_flag` au lieu de la `glib` pour la classe
  \arccore{SpinLock} (\pr{1110})
- Début support pour les types \arccore{Float16}, \arccore{Float32} et \arccore{BFloat16}
  (\pr{1087}, \pr{1088}, \pr{1089}, \pr{1095}, \pr{1099}, \pr{1101},
  \pr{1102}, \pr{1106})
- Rend privé la classe `StringImpl` (\pr{1096}, \pr{1097})
- Améliorations diverses dans \arccore{ISerializer} (\pr{1090},
  \pr{1093}, \pr{1112})
- Rend publique la méthode \arccore{Array::resizeNoInit()} (\pr{1220},
  \pr{1297})
- Ajoute argument template dans \arccore{Span}, \arccore{SmallSpan} et
  \arccore{SpanImpl} pour indiquer la valeur minimale de l'index (`0`
  par défaut) (\pr{1296}).
- Ajoute conversions implicites et explicites entre \arccore{Array} et
  \arccore{SmallSpan} (\pr{1277}, \pr{1279})
- Corrige pour \arccore{String} des conversions multiples entre UTF-8
  et UTF-16 (\pr{1251}).
- Ajoute fonction utilitaire pour remplir un \arccore{Span} avec des
  valeurs aléatoires (\pr{1103})
- Ajoute méthode \arccore{arccoreCheckRange()} pour vérifier si une
  valeur est dans un intervalle (\pr{1091}).
- Ajoute classe \arccore{BuiltInDataTypeContainer} pour généraliser
  les opérations sur les types de données de base \pr{1105}
- Ajoute possibilité de récupérer le communicateur MPI dans
  \arccore{MessagePassing::IMessagePassingMng} (\pr{1248})
- Début ré-organisation interne pour rendre l'utilisation de la 'Glib'
  optionnelle (\pr{1328}).

### Axlstar

- Améliore la gestion de la documentation (\pr{1052}, \pr{1070},
  \pr{1072}, \pr{1073}, \pr{1084}, \pr{1104}, \pr{1107})
- Ajoute option pour changer le répertoire d'installation (\pr{1171})
- Ajoute option en ligne de commande pour changer le nom du namespace
  généré (\pr{1321}).
- Ajoute nom du namespace dans la protection de l'en-tête du fichier
  généré (\pr{1322}).

### Alien

- Début support du remplissage des matrices sur accélérateur
  (\pr{1177}, \pr{1216}, \pr{1242}, \pr{1245}, \pr{1252}, \pr{1267})
- Améliore la documentation Alien et la gère en même temps que la
  documentation Arcane (\pr{1265})
- Ajoute gestion des blocs pour le backend PETSc (\pr{1286}, \pr{1302})
- Ajoute possibilité de passer des paramètres supplémentaires dans
  l'API C de Alien (\pr{1259})

##########################################################################################

## Arcane Version 3.11.15 (23 novembre 2023) {#arcanedoc_version3110}

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
- Utilise `std::atomic_ref` au lieu de `std::atomic` pour la gestion
  des réductions sur accélérateur (\pr{1352})

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
  \arcanemat{AllEnvCell} (\pr{742}).
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

