# Pool de mémoire {#arcanedoc_acceleratorapi_memorypool}

[TOC]

%Arcane dispose depuis la version 3.14.10 (novembre 2024) d'un
mécanisme de pool mémoire permettant de conserver une partie de la
mémoire allouée pour les accélérateurs et ainsi éviter des appels
couteux aux fonctions d'allocations ou de désallocation.

\note Ce mécanisme est uniquement fonctionnel pour CUDA et ROCM/HIP.
A partir de la version 4.1 de %Arcane, le gestionnaire de pool mémoire
est activé par défaut.

\warning L'utilisation du pool mémoire peut changer le comportement du
code en supprimant des synchronisations implicites effectuées sur les
streams associées aux allocations et au désallocation. Notamment, les
appels tels que `cudaMalloc()` ou `cudaFree()`. La page
[CUDA implicit synchronization behavior and conditions in detail]
(https://forums.developer.nvidia.com/t/cuda-implicit-synchronization-behavior-and-conditions-in-detail/251729)
explique ce comportement pour CUDA.

Il est possible d'activer et de modifier le comportement du pool
mémoire en positionnant des variables d'environnements.

<table>
<tr><th>Variable d'environnement</th><th>Description</th></tr>

<tr>
<td>ARCANE_ACCELERATOR_MEMORY_POOL</td>
<td>
Indique le type de mémoire pour lesquelles on souhaite activer le
pool. Les valeeurs sont spécifiées par une combinaison de bit:
- 1 pour la mémoire managée (\arcane{eMemoryResource::UnifiedMemory})
- 2 pour la mémoire sur l'accélérateur (\arcane{eMemoryResource::Device})
- 4 pour la mémoire punaisée sur l'hôte
  (\arcane{eMemoryResource::HostPinned})

Si la valeur de la variable d'environnement vaut `7` par exemple,
alors le pool mémoire est actif pour ces 3 types de ressource mémoire.
Si la valeur est `0`, alors le pool mémoire est désactivé pour toutes
les mémoires.
</td>
</tr>

<tr>
<td>ARCANE_ACCELERATOR_MEMORY_POOL_MAX_BLOCK_SIZE</td>
<td>
Indique la taille maximale (en octet) des blocs qui sont conservés
dans le pool mémoire. Une valeur élevée permet de faire moins
d'allocations et de désallocation mais en contre-partie conserve plus
de mémorie ce qui réduit la quantité disponible pour les allocations
qui ne passent pas par le pool de mémoire. La valeur par défaut est de
1Mo (1024*1024).
</td>
</tr>

<tr>
<td>ARCANE_ACCELERATOR_MEMORY_PRINT_LEVEL</td>
<td>
Indique si on affiche des informations sur l'utilisation de la
mémoire. Ces informations sont utiles pour le débug uniquement. Les
valeurs possibles sont:
- 0 n'affiche aucune information
- 1 affiche des statistiques d'utilisation en fin de calcul
- 2 idem 1 et affiche des informations lors des réallocations
- 3 idem 2 et affiche la pile d'appel pour une réallocation pour les
  tableaux sans nom.
- 4 idem 3 mais affiche la pile d'appel lors de la réallocation pour
  tous les tableaux.
</td>
</tr>

</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_acceleratorapi_reduction
</span>
</div>
