# Intégration avec CUPTI (Cuda Profiling Tools Interface) {#arcanedoc_debug_perf_cupti}

[TOC]

## Description

[CUPTI](https://docs.nvidia.com/cupti/index.html) est une bibliothèque
fournie par NVIDIA. Elle permet entre autre de récupérer des
évènements concernant la gestion de la mémoire unifiée. C'est dans ce
contexte qu'%Arcane utilise CUPTI.

L'utilisation de CUPTI se fait via des variables d'environnement

<table>
<tr><th>Variable d'environnement</th><th>Description</th></tr>

<tr>
<td>ARCANE_CUPTI_LEVEL</td>
<td>
Indique les évènements qu'on veut tracer. A noter que pour le niveau 2
il faut un accès exclusif au GPU et donc ce mode ne fonctionne pas en
parallèle. Les valeurs possibles sont:
- 0 non actif
- 1 transferts mémoire unifiée
- 2 idem 1 + noyaux de calcul
</td>
</tr>

<tr>
<td>ARCANE_CUPTI_FLUSH</td>
<td>
Indique à quel moment on affiche les informations sur les
évènements. Pour avoir un suivi précis il est nécessaire d'afficher
les informations après chaque exécution de noyau GPU mais ce mode
peut augmenter d'une facteur important le temps d'exécution. Les
valeurs possibles sont:
- 0 pas de flush explicite
- 1 flush après chaque noyau
</td>
</tr>

<tr>
<td>ARCANE_CUPTI_PRINT</td>
<td>
Indique si on souhaite effectuer un affichage pour chaque
évènement. Cela peut ralentir considérablement le temps
d'exécution. Les valeurs possibles sont:
- 0 pas d’affichage
- 1 affichage listing (sur std::cout)
</td>
</tr>

<tr>
<td>ARCANE_CUDA_MALLOC_TRACE</td>
<td>
Indique si on souhaite tracer tous les appels à
`cudaMallocManaged()`. Les valeurs possibles sont:
- 0 pas de trace
- 1 trace le nom du tableau
- 2 idem 1 + affichage des malloc et des free
- 3 idem 2 + pile d’appel
</td>
</tr>

<tr>
<td>ARCANE_CUDA_UM_PAGE_ALLOC</td>
<td>
Indique la manière d'allouer via `cudaMallocManaged()`. Il est
possible pour chaque allocation d'allouer un multiple de la taille
d'une page mémoire. Comme les transferts de la mémoire unifiée se font
page par page, cela permet de mieux distinguer quel accès mémoire a
provoqué le transfert. La contrepartie est que chaque allocation
nécessite d'allouer au moins une page (soit 4Ko en général)
Les valeurs possibles sont:
- 0 allocation normale
- 1 allocation par multiple de la taille d'une page.
</td>
</tr>

</table>

## Exemple

L'exemple suivant permet de tracer les transferts en mémoire unifiée
et d'afficher le nom du tableau associé.

```
ARCANE_CUDA_MALLOC_TRACE=1 ARCANE_CUPTI_FLUSH=1 ARCANE_CUPTI_LEVEL=1 ./my_test -A,AcceleratorRuntime=cuda toto.arc
```

Avec le résultat suivant

```
*I-ArcaneMasterInternal *** ITERATION       17  TIME 3.594972986357219e-03  LOOP       17  DELTAT 4.177248169415655e-04 ***
*I-ArcaneMasterInternal Date: 2023-10-26T09:40:35 Conso=(R=2.168,I=0.131,C=0.166) Mem=(222,m=222:0,M=222:0,avg=222)
UNIFIED_MEMORY_COUNTER [ 4179074172 4179078748 ] address=0x7f1cc01f9000 kind=BYTES_TRANSFER_HTOD value=24576 flags=3 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4179078748 4179081788 ] address=0x7f1cc01ff000 kind=BYTES_TRANSFER_HTOD value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4179241052 4179244924 ] address=0x7f1cc01f9000 kind=BYTES_TRANSFER_DTOH value=24576 flags=3 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4179244924 4179246172 ] address=0x7f1cc01ff000 kind=BYTES_TRANSFER_DTOH value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistoryGlobalTime stack=
UNIFIED_MEMORY_COUNTER [ 4223957312 4223960384 ] address=0x7f1cc0bff000 kind=BYTES_TRANSFER_HTOD value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistory_Iterations_1 stack=
*I-ArcaneMasterInternal  
*I-ArcaneMasterInternal *** ITERATION       18  TIME 4.054470284992941e-03  LOOP       18  DELTAT 4.594972986357221e-04 ***
*I-ArcaneMasterInternal Date: 2023-10-26T09:40:35 Conso=(R=2.299,I=0.131,C=0.176) Mem=(222,m=222:0,M=222:0,avg=222)
UNIFIED_MEMORY_COUNTER [ 4353892054 4353895094 ] address=0x7f1cc0bff000 kind=BYTES_TRANSFER_HTOD value=4096 flags=2 source=0 destination=0 name=Mesh0_TimeHistory_Iterations_1 stack=
```

Les deux valeurs après `UNIFIED_MEMORY_COUNTER` correspondent au temps
de début et de fin du tranfert. Les autres champs sont:
- `address` : adresse mémoire du tableau
- `kind` : type de tranfert (`Host to device` ou `Device to host`)
- `value`: quantité (en octet) de mémoire tranférée
- `flags` : si `2` alors le tranfert est explicitement demandé par le
  code. Si `3`, il s'agit d'un tranfert spéculatif initié par le
  driver NVIDIA.
- `source` et `destination` : numéro du device
- `name` : nom du tableau Arcane. Cela n'est actif que si la variable
  d'environnement `ARCANE_CUDA_MALLOC_TRACE` vaut au moins 1. Si le
  transfert n'est pas lié à un tableau %Arcane (\arccore{UniqueArray} ou
  \arcane{NumArray}), il n'y aura pas de nom associé. A noter
  que comme les transferts se font page par page, il est possible que
  le tableau indiqué ne soit pas celui qui a provoqué le
  transfert. Pour éviter cet effet, il est possible d'allouer une page
  pour chaque allocation en positionnant la variable d'environnement
  `ARCANE_CUDA_UM_PAGE_ALLOC` à `1`.

En fin de calcul est affiché la quantité totale de mémoire transférée
et le nombre de transferts. Par exemple:

```
MemoryTransferSTATS: HTOD = 17895424 (680) DTOH = 7102464 (301)
```

Dans cet exemple, on a fait 680 transferts du CPU vers le GPU pour
17Mo de données transférées. On a fait 301 transferts du GPU vers le
CPU pour 7Mo de données transférées.


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_unit_tests
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling
</span>
</div>
