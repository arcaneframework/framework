# Les fenêtres mémoires en mémoire partagée en multi-processus {#arcanedoc_parallel_shmem}

[TOC]

## Introduction {#arcanedoc_parallel_shmem_intro}

Cette page va décrire comment utiliser la mémoire partagée entre les processus d'un même noeud de calcul à l'aide de
fenêtres mémoires.

Une fenêtre mémoire est un espace mémoire alloué dans une partie de la mémoire accessible par tous les processus.
Cette fenêtre sera découpée en plusieurs segments, un par processus.

Deux implémentations différentes sont disponibles : une implémentation avec tous les segments contigus et avec une
taille constante, définie lors de la construction de l'objet et une autre implémentation avec des segments non-contigus
et avec une taille pouvant évoluer.

## Implémentation avec segments contigus {#arcanedoc_parallel_shmem_const}

Cette implémentation va permettre de créer une fenêtre mémoire ayant tous ses segments contigus.
Il est ainsi assez simple de redécouper les segments pendant l'utilisation (par exemple, pour équilibrer un calcul).

### Utilisation {#arcanedoc_parallel_shmem_const_usage}

Cette partie est gérée par la classe Arcane::ContigMachineShMemWin.

Cette classe peut utiliser trois implémentations de IContigMachineShMemWinBase, une par type de
Arcane::IParallelMng.
Il est donc possible d'utiliser cette classe de la même façon, que l'on ait un MpiParallelMng, un SequentialParallelMng,
un SharedMemoryParallelMng ou un HybridParallelMng (\ref arcanedoc_execution_launcher_exchange).

La création d'un objet de ce type est collectif. Une instance de cette classe va créer une fenêtre mémoire composée de
plusieurs segments (un par sous-domaine).

L'accès aux éléments des segments n'est pas collectif. L'accès concurrent à un élément est possible en utilisant
des sémaphores, des mutex ou des std::atomic.
Pour les std::atomic, il faut que les opérations soient `address-free` :

```c
bool is_lock_free = std::atomic<Real>{}.is_lock_free();
```

Lors de la construction de cet objet, chaque sous-domaine va fournir une taille de segment. La taille de la fenêtre va
être égale à la somme des tailles de segments.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_1

Pour accéder à son segment, il est possible d'utiliser la méthode Arcane::ContigMachineShMemWin::segmentView().

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_3

Une fois le segment modifié, on peut faire une barrier pour s'assurer que tout le monde a écrit dans son segment avant
de s'en servir.

Pour savoir quels sous-domaines se partagent une fenêtre sur le noeud, il est possible de récupérer un tableau de rang.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_2

La position des rangs dans ce tableau correspond à la position de leur segment dans la fenêtre.

Pour la lecture des segments des autres sous-domaines du noeud, on peut utiliser la méthode
Arcane::ContigMachineShMemWin::segmentConstView().

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_4

La taille de la fenêtre ne peut pas être modifiée. En revanche, l'implémentation dans %Arcane permet de redimensionner
les segments de manière collectif (à la condition que la nouvelle taille de la fenêtre soit inférieure ou égale à la
taille d'origine).

\note Une implémentation avec des fenêtres mémoires ayant des segments non-contigües pourrait être plus performante
mais rendrait cette fonctionnalité impossible. 

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_5

\remark Les éléments de la fenêtre ne sont pas modifiés pendant le redimensionnement.

La fenêtre étant contigüe, l'accès à toute la fenêtre est possible pour tous les sous-domaines.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_6

## Implémentation avec segments non-contigus {#arcanedoc_parallel_shmem_var}

Cette implémentation est assez différente de la précédente.
Ici, les segments des fenêtres mémoires ne sont plus contigus. De plus, avec cette implémentation, il est possible de
redimensionner les segments comme un tableau dynamique classique.

Néanmoins, cette opération est collective, ce qui contamine la plupart des méthodes de l'implémentation.

### Utilisation {#arcanedoc_parallel_shmem_var_usage}

Cette partie est gérée par la classe Arcane::MachineShMemWin.

Comme pour la précédente implémentation, celle-ci est compatible avec tous les modes de parallélisme de %Arcane.

La création d'un objet de ce type est collectif. Une instance de cette classe va créer une fenêtre mémoire composée de
plusieurs segments (un par sous-domaine).

Comme un UniqueArray, il est possible de spécifier une taille initiale (ici `5`) :
\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_7

Et il est possible de ne pas spécifier de taille initiale.
\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_8

La méthode Arcane::MachineShMemWin::machineRanks() est disponible et renvoie le même tableau que
l'implémentation Arcane::ContigMachineShMemWin.

Pour explorer notre segment ou le segment d'un autre sous-domaine, il est possible d'utiliser les mêmes méthodes que
précédemment :

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_9

En revanche, comme les segments ne sont pas contigus, les méthodes `windowView()` ne sont pas disponibles.

Les segments ont une taille qui peut être augmentée ou diminuée au cours du temps.

Il est possible d'ajouter des éléments avec la méthode Arcane::MachineShMemWin::add(Arcane::Span<const Type>
elem) :

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_10

Cette méthode est collective, tous les sous-domaines d'un noeud doivent l'appeler. Si un sous-domaine ne souhaite pas
ajouter d'éléments dans son segment, il peut appeler la méthode `add()` avec un tableau vide ou sans argument
(Arcane::MachineShMemWin::add()).

Cette opération peut être couteuse à cause de la réallocation mémoire. Il est donc conseillé d'ajouter une grande
quantité d'éléments en une fois plutôt qu'élément par élément.

Si l'ajout élément par élément est indispensable, la méthode
Arcane::MachineShMemWin::reserve(Arcane::Int64 new_capacity) est disponible afin d'éviter de réallouer
plusieurs fois un segment :

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_11

Dans ce bout de code, on va réserver un espace de `20` `Integer` pour tous les sous-domaines. Cette valeur peut être
différente pour chaque sous-domaine (si un sous-domaine ne veut pas réserver plus d'espace, il peut appeler
Arcane::MachineShMemWin::reserve()).

\note Avec cette méthode, on ne peut pas réserver moins d'espace que déjà réservé (appeler `reserve(0)` n'a aucun
effet). Pour réduire l'espace réservé, la méthode Arcane::MachineShMemWin::shrink() est disponible.

\warning Comme pour les UniqueArray, la méthode Arcane::MachineShMemWin::reserve(Arcane::Int64
new_capacity) n'a pas la même fonction que la méthode Arcane::MachineShMemWin::resize(Arcane::Int64
new_nb_elem). La première réserve uniquement l'espace mémoire mais cet espace reste inaccessible sans `add()` ou sans
`resize()`. La seconde change le nombre d'éléments du segment et appelle `reserve()` si nécessaire.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_12

Dans notre exemple, `resize()` va augmenter le nombre d'éléments de tous les segments sauf du sous-domaine qui avait
fait les `add()` précédemment (qui possède 15 éléments, contre 5 pour les autres).
Ce sous-domaine va passer de 15 éléments à 12.

Comme pour la méthode `reserve()`, chaque sous-domaine peut mettre la valeur qu'il veut.

Il est aussi possible d'ajouter des éléments dans le segment d'un autre sous-domaine avec la méthode collective
Arcane::MachineShMemWin::addToAnotherSegment(Arcane::Int32 rank, Arcane::Span<const Type> elem).

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_13


\warning Il est impossible de mélanger les appels à `add()` et à `addToAnotherSegment()`. Si un sous-domaine appelle
la méthode `addToAnotherSegment()`, tous les sous-domaines devront appeler collectivement `addToAnotherSegment()` (avec
ou sans paramètres) et non `add()`.

Le fonctionnement est presque identique à la méthode `add()` mais avec un paramètre en plus pour désigner le rang du
sous-domaine possédant le segment à modifier.

\warning Deux sous-domaines ne peuvent pas ajouter d'éléments dans un même segment (en une fois) (ce qui permet d'éviter
les problèmes de concurrences).
```c
// Pas possible :
if (my_rank == 0){
  window.addToAnotherSegment();
}
else if (my_rank == 1){
  window.addToAnotherSegment(0, mon_tableau);
}
else if (my_rank == 2){
  window.addToAnotherSegment(0, mon_tableau);
}
```
```c
// Possible :
if (my_rank == 0){
  window.addToAnotherSegment();
  window.addToAnotherSegment();
}
else if (my_rank == 1){
  window.addToAnotherSegment();
  window.addToAnotherSegment(0, mon_tableau);
}
else if (my_rank == 2){
  window.addToAnotherSegment(0, mon_tableau);
  window.addToAnotherSegment();
}
```

## Mémoire partagée entre processus {#arcanedoc_parallel_shmem_shmem}

La mémoire partagée entre processus ne doit pas être vu comme la mémoire partagée en multithread.
Ce partage n'est fait que sur une partie de la mémoire, pas sur toute la mémoire.

Prenons cette structure :

```c
struct MaStruct
{
    MaStruct()
    : array_integer(10)
    {}
    
    UniqueArray<Integer> array_integer;
};
```

On peut l'utiliser comme ceci :

```c
MaStruct ma_struct;
ma_struct.array_integer[0] = 123 * (my_rank+1);
```

Si on utilise cette structure dans une fenêtre, ça donnerait :

```c
ContigMachineShMemWin<MaStruct> win_struct(pm, 1);
Span<MaStruct> my_span = win_struct.segmentView();
new (my_span.data()) MaStruct();

my_span[0].array_integer[0] = 123 * (my_rank+1);
```

On peut afficher la valeur que l'on a attribuée, ça fonctionne correctement :

```c
debug() << "Elem : " << my_span[0].array_integer[0];
```

Mais si on veut afficher la valeur d'un autre processus :

```c
window.barrier();

Span<MaStruct> other_span = win_struct.segmentView(machine_ranks[(my_rank + 1) % machine_nb_proc]);
debug() << "Elem : " << other_span[0].array_integer[0];

my_span[0].~MaStruct();
```

En multi-processus (lancement avec `mpirun -n 2 ...`), le programme va planter (segfault), alors qu'il ne plantera pas
en multithreading (lancement avec `-A,S=2`).

En multi-processus, les attributs du tableau `UniqueArray<Integer> array_integer;` de la structure ne sont pas alloués
en mémoire partagée (les `new` ou `malloc` sont faits sur la mémoire locale), les autres processus n'y ont donc pas
accès.

Il est aussi important de noter qu'un même emplacement mémoire en mémoire partagée est adressé différemment entre les
processus. Donc, si l'on donne un allocateur en mémoire partagée à l'`UniqueArray`, les adresses utilisées seront
valables uniquement localement.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_loadbalance
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
