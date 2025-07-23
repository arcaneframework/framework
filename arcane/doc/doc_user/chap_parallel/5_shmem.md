# Les fenêtres mémoires en mémoire partagée en multi-processus {#arcanedoc_parallel_shmem}

[TOC]

## Introduction {#arcanedoc_parallel_shmem_intro}

Cette page va décrire comment utiliser la mémoire partagée entre les processus d'un même noeud de calcul à l'aide de
fenêtres mémoires.


## Utilisation {#arcanedoc_parallel_shmem_usage}

Cette partie est gérée par la classe Arcane::MachineMemoryWindow.

Cette classe peut utiliser trois implémentations de IMachineMemoryWindowBase, une par type de 
Arcane::IParallelMng.
Il est donc possible d'utiliser cette classe de la même façon, que l'on ait un MpiParallelMng, un SequentialParallelMng,
un SharedMemoryParallelMng ou un HybridParallelMng (\ref arcanedoc_execution_launcher_exchange).

La création d'un objet de ce type est collectif. Une instance de cette classe va créer une fenêtre mémoire composée de
plusieurs segments (un par sous-domaine).

L'accès aux éléments des segments n'est pas collectif. L'accès concurrent à un élément est possible (TODO mais les
protections n'ont pas encore été ajoutées).

Lors de la construction de cet objet, chaque sous-domaine va fournir une taille de segment. La taille de la fenêtre va
être égale à la somme des tailles de segments.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_1

Cette fenêtre sera contigüe en mémoire.

Pour accéder à son segment, il est possible d'utiliser la méthode Arcane::MachineMemoryWindow::segmentView().

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_3

Une fois le segment modifié, on peut faire une barrier pour s'assurer que tout le monde a écrit dans son segment avant
de s'en servir.

Pour savoir quels sous-domaines se partagent une fenêtre sur le noeud, il est possible de récupérer un tableau de rang.

\snippet ParallelMngTest.cc snippet_arcanedoc_parallel_shmem_usage_2

La position des rangs dans ce tableau correspond à la position de leur segment dans la fenêtre.

Pour la lecture des segments des autres sous-domaines du noeud, on peut utiliser la méthode
Arcane::MachineMemoryWindow::segmentConstView().

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
MachineMemoryWindow<MaStruct> win_struct(pm, 1);
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
