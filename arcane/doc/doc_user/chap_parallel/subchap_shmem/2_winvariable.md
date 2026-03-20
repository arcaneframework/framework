# Variables en mémoire partagée {#arcanedoc_parallel_shmem_winvariable}

[TOC]

## Introduction {#arcanedoc_parallel_shmem_winvariable_intro}

Les variables %Arcane utilisent habituellement l'allocateur par défaut pour allouer de la mémoire. Sans GPU, on utilise
la mémoire locale de la machine et avec GPU, on utilise la mémoire unifiée.

Un nouvel allocateur mémoire (interne à %Arcane) est disponible et permet d'allouer de la mémoire en mémoire partagée
machine.
Pour cela, en interne, on utilise la classe présentée précédemment : Arcane::MachineShMemWin. On aura donc accès à des
segments non-contigus.

Ce mode est compatible avec l'ensemble des types de variables %Arcane (sauf les variables scalaires sans support
(exemple : `VariableScalarReal`) et les variables partielles).

La principale difficulté pour utiliser ce mode mémoire partagée est de s'assurer que tous les appels qui réallouent la
mémoire soit collectifs.

Pour les variables redimensionnées par %Arcane, l'utilisateur n'a pas besoin de se préoccuper de ces appels collectifs,
%Arcane s'en occupe. Exemple des variables au maillage, %Arcane s'occupe de les redimensionner si le maillage évolue.

En revanche, pour les variables avec lesquelles une méthode `resize()` est disponible (ou `reshape()` pour les variables
multi-dimensionnelles), il est nécessaire de s'assurer que tous les sous-domaines de la machine fassent un appel à cette
méthode (quitte à faire `var.resize(var.size())` pour les sous-domaines ne nécessitant pas de redimensionnement).

Ces appels de redimensionnement mirent de côté, l'utilisation des variables en mémoire partagée est identique à
l'utilisation des variables en mémoire locale.

Pour déclarer une variable en mémoire partagée, il suffit d'ajouter la propriété `IVariable::PInShMem` lors de sa
création (pour l'instant, il n'est pas possible de le faire via l'AXL).

## Accès aux segments mémoire des autres sous-domaines {#arcanedoc_parallel_shmem_winvariable_shared}

L'utilité de mettre des variables en mémoire partagée est de pouvoir accéder aux données d'autres sous-domaines sans
échanges de messages.

Pour accéder aux données de tous les sous-domaines, on peut utiliser les classes `MachineShMemWinVariable`.
Une classe par type de variable %Arcane :

<table>
  <tr>
    <th>Type de variable<br>(`exemple`)</th>
    <th>Classe à utiliser</th>
  </tr>

  <tr>
    <td>Variable tableau 1D sans support<br>(`Arcane::VariableArrayInt32`)</td>
    <td>Arcane::MachineShMemWinVariableArrayT</td>
  </tr>

  <tr>
    <td>Variable scalaire au maillage<br>(`Arcane::VariableCellInt32`)</td>
    <td>Arcane::MachineShMemWinMeshVariableScalarT</td>
  </tr>

  <tr>
    <td>Variable tableau 2D sans support<br>(`Arcane::VariableArray2Int32`)</td>
    <td>Arcane::MachineShMemWinVariableArray2T</td>
  </tr>

  <tr>
    <td>Variable tableau 1D au maillage<br>(`Arcane::VariableCellArrayInt32`)</td>
    <td>Arcane::MachineShMemWinMeshVariableArrayT</td>
  </tr>

  <tr>
    <td>Variable multi-dimensionnelle scalaire<br>(`Arcane::MeshMDVariableRefT<Cell, Real, MDDim2>`)</td>
    <td>Arcane::MachineShMemWinMeshMDVariableT</td>
  </tr>

  <tr>
    <td>Variable multi-dimensionnelle vectorielle<br>(`Arcane::MeshVectorMDVariableRefT<Cell, Real, 7, MDDim2>`)</td>
    <td>Arcane::MachineShMemWinMeshVectorMDVariableT</td>
  </tr>

  <tr>
    <td>Variable multi-dimensionnelle matricielle<br>(`Arcane::MeshMatrixMDVariableRefT<Cell, Real, 2, 5, MDDim1>`)</td>
    <td>Arcane::MachineShMemWinMeshMatrixMDVariableT</td>
  </tr>
</table>

Trois méthodes sont communes pour ces classes :

- `machineRanks()`,
- `barrier()`,
- `updateVariable()`.

Les deux premières ont déjà été brievement décrites dans la partie précedente
(\ref arcanedoc_parallel_shmem_winarray_var_usage).

`Arcane::MachineShMemWinVariableCommon::machineRanks()` permet de récupérer les rangs des sous-domaines du noeud de
calcul.

Par exemple, si la vue renvoyée contient `[0, 2, 4, 6]`, on sait que le noeud de calcul possède ces sous-domaines et
que l'on a accès à leurs données via `MachineShMemWin`.<br>
En utilisant la méthode `Arcane::IParallelMng::commSize()`, sachant que les rangs sont contigus, on peut aussi
déterminer quels sous-domaines ne sont pas dans notre noeud de calcul.
Par exemple, si `commSize() = 8`, alors les sous-domaines pour lesquels on devra faire des communications inter-noeuds
sont les sous-domaines `[1, 3, 5, 7]`.

<br>

`Arcane::MachineShMemWinVariableCommon::barrier()` permet de faire une barrière pour tous les sous-domaines du noeud de
calcul (donc, si on reprend l'exemple précedent, une barrière pour les sous-domaines `[0, 2, 4, 6]`).

C'est utile dans le cas où les sous-domaines utilisent une fenêtre mémoire pour partager des infos, pour attendre que
chaque sous-domaine ait écrit dans sa fenêtre avant que d'autres sous-domaines du noeud lisent ces données. Le grain est
plus petit que `Arcane::IParallelMng::barrier()`.

<br>

La véritable différence avec la partie précédente est la méthode
`Arcane::MachineShMemWinMeshVariableArrayT::updateVariable()`.

En interne, comme expliqué en introduction, on utilise un allocateur qui alloue de la mémoire en mémoire partagée et on
utilise la classe `Arcane::MachineShMemWin` pour y accéder.<br>
`Arcane::MachineShMemWinVariable` utilise à son tour `Arcane::MachineShMemWin` pour accéder à la mémoire partagée des
variables.

Le problème est que la taille d'un tableau en %Arcane n'est pas forcément de la même taille que la mémoire allouée par
celui-ci. Par conséquent, toujours en interne, on ne peut pas se baser sur la taille renvoyée par
`Arcane::MachineShMemWin` pour construire les vues sur les variables.

On doit donc récupérer les tailles des variables de chaque sous-domaine d'une autre façon. Pour cela, on utilise une
fenêtre mémoire afin de les partager.

Lorsque l'on change la taille d'une variable (via un changement dans le maillage ou via un resize pour les variables
tableaux), on doit mettre à jour les tailles des variables.

Aujourd'hui, **c'est à l'utilisateur de le faire via un appel à `updateVariable()`**.

Il est aussi possible de détruire l'objet `Arcane::MachineShMemWinVariable` et de le recréer après mise à jour de la
variable.

### Exemples {#arcanedoc_parallel_shmem_winvariable_shared_examples}

Quelques exemples afin d'illustrer l'utilisation de ces classes :

<div style="text-align: center;">**Exemple 1**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_1

Dans cet exemple, chaque sous-domaine possède un tableau de deux `Int32`.

\note Le tableau pourrait être de taille différente pour chaque sous-domaine.

Chaque sous-domaine met son rang dans les deux cases du tableau puis chaque sous-domaine affiche la vue
de chaque tableau (`var_sh.view(rank)` renvoie une vue de deux `Int32` du tableau de `rank`).

L'appel à la méthode `updateVariable()` pourrait facilement être retiré en mettant `var.resize(2);` entre la création
de la variable et la création du `MachineShMemWinVariable` :

<div style="text-align: center;">**Exemple 1.1**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_1_1

Une alternative à l'appel à `updateVariable()` est la destruction/recréation de `MachineShMemWinVariable` :

<div style="text-align: center;">**Exemple 1.2**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_1_2

----

<div style="text-align: center;">**Exemple 2**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_2

Pour les grandeurs au maillage, on a accès à l'opérateur `Arcane::MachineShMemWinMeshVariableScalarT::operator()()` qui
permet d'accéder à la valeur d'un `Item` grâce à son `local_id`.

\warning Le `local_id` est local au sous-domaine ciblé. Il est donc nécessaire de le partager d'une manière ou d'une
autre. Il ne faut pas utiliser les `local_id` d'un sous-domaine pour accéder aux `Items` d'un autre sous-domaine !

Si plusieurs valeurs doivent être lues d'un autre sous-domaine, il est vivement conseillé de le faire en récupérant une
vue via la méthode `Arcane::MachineShMemWinMeshVariableScalarT::view()`. Exemple :

<div style="text-align: center;">**Exemple 2.1**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_2_1

\warning S'il y a eu suppressions d'`Items` sans compactage, le code de l'**Exemple 2.1** affichera des valeurs
d'`Items` supprimés.

Dans l'**Exemple 2**, la barrière est importante, étant donné que chaque sous-domaine accédera aux données des autres
sous-domaines.<br>
Néanmoins, il est aussi possible de faire ceci, afin d'éviter la barrière :

<div style="text-align: center;">**Exemple 2.2**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_2_2

----

<div style="text-align: center;">**Exemple 3**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_3

Ici, on a un tableau 2D sans support.

\note Comme pour le tableau 1D, le tableau 2D pourrait être de taille différente pour chaque sous-domaine.

La méthode `Arcane::MachineShMemWinVariableArray2T::view()` permet de récupérer une vue (de type Arcane::Span2) sur le
tableau 2D d'un autre sous-domaine du noeud de calcul.

----

<div style="text-align: center;">**Exemple 4**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_4

Une variable tableau 1D au maillage est un tableau 2D mais avec la première dimension qui correspond au nombre
d'`Items`.

\warning Par rapport à la variable tableau 2D sans support, la taille de la seconde dimension doit être identique pour
chaque sous-domaine.

On retrouve la méthode `Arcane::MachineShMemWinMeshVariableArrayT::view()` qui retourne une vue du tableau 2D de la
variable d'un autre sous-domaine.
La première dimension prend un `local_id` d'un `Item` de l'autre sous-domaine et la seconde dimension est la position
dans le tableau de l'`Item`.

----

<div style="text-align: center;">**Exemple 5**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_5

Avec les variables multi-dimensionnelles, la méthode `Arcane::MachineShMemWinMeshMDVariableT::view()` renvoie un
`Arcane::MDSpan` avec une dimension en plus par rapport à la dimension de la variable, la première dimension
correspondant au support.

L'opérateur `Arcane::MachineShMemWinMeshMDVariableT::operator()()` est aussi disponible et
permet de récupérer une vue multi-dimensionnelle de la dimension de la variable (vu que l'on donne aussi le `local_id`).

Comme dit précédemment, si accès à plusieurs tableaux d'`Items` pour un sous-domaine donné, il vaut mieux récupérer une
vue complète via `Arcane::MachineShMemWinMeshMDVariableT::view()`.

----

<div style="text-align: center;">**Exemple 6**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_shared_examples_6

Pour les variables MD vectorielles et matricielles, on retrouve les mêmes méthodes que dans l'exemple précédent.

## Protection/reprise {#arcanedoc_parallel_shmem_winvariable_checkpoints}

Les variables en mémoire partagée sont compatibles avec le mécanisme de protection/reprise.

Une propriété de variable a été ajouté afin de pouvoir ne pas sauvegarder les tableaux de sous-domaines spécifiés.

Il s'agit de la propriété `IVariable::PDumpNull`. Ce n'est pas une propriété réservée aux variables en mémoire partagée.

Cette propriété, lorsqu'elle est spécifiée sur une variable, pour un sous-domaine donné, va permettre de sauvegarder un
tableau vide. C'est particulièrement utile en reprise pour les variables en mémoire partagée étant donné l'obligation de
faire des opérations collectives.

### Exemples {#arcanedoc_parallel_shmem_winvariable_checkpoints_examples}

<div style="text-align: center;">**Exemple 7**</div>
\snippet{trimleft} VariableInShMemUnitTest.cc snippet_arcanedoc_parallel_shmem_winvariable_checkpoints_examples_7

On va appeler un sous-domaine maitre du noeud de calcul, le sous-domaine ayant le plus petit rang du noeud (comme
`machine_ranks` est trié par ordre croissant, il s'agit du premier rang du tableau).

Dans cet exemple, à la première itération de la boucle en temps, on redimensionne la variable pour tous les
sous-domaines.

Ensuite, on attribue la propriété `IVariable::PDumpNull` à tous les sous-domaines non-maitres.

Enfin, à la reprise, on vérifie que les tableaux des sous-domaines maitres ont bien été restaurés et que les autres
tableaux sont vides.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_shmem_winarray
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
