# Prise en compte du parallélisme dans %Arcane {#arcanedoc_parallel_intro}

[TOC]

<!-- présente la manière dont %Arcane prend en charge le parallèlisme par partionnement de domaine. -->

## Généralités {#arcanedoc_parallel_intro_general}

Dans ARCANE, le parallélisme est géré par échange de messages. Le maillage 
est partitionné en plusieurs sous-domaines. Chaque sous-domaine est 
éventuellement complété d'une ou plusieurs couches de mailles fantômes qui 
représentent une duplication d'entités pour lesquelles il faudra effectuer
une synchronisation. Chaque processeur effectue les calculs sur un sous-domaine
et synchroniseses variables régilièrement avec les autres processeurs.

En règle générale, il n'y a pas de solution unique au
problème des synchronisations :
- pour certaines variables, il est possible soit de faire le
  calcul sur tout le sous-domaine, c'est à dire à la fois sur les
  entités fantômes et les entités propres, soit de faire le calcul
  uniquement sur les entités propres et synchroniser ensuite les valeurs
  calculées. Le choix de l'une ou l'autre des solutions dépend
  globalement de deux paramètres : le temps nécessaire pour effectuer le
  calcul et le temps nécessaire pour effectuer la synchronisation. Chacun de
  ses paramètres dépend lui même d'autres paramètre comme la puissance
  du processeur, la bande passante du réseau d'interconnexion, ...;
- lorsqu'on augmente le nombre de couches d'entités fantômes,
  certaines variables qui ne sont utilisées que temporairement lors d'une
  itération n'ont plus besoin d'être synchronisées. Cela permet donc de
  réduire le nombre de synchronisations mais en contre-partie chaque
  synchronisation concernera plus de processeurs et d'entités fantômes.

Afin de ne pas compliquer inutilement la prise en compte du
parallélisme, ARCANE fonctionne aujourd'hui en utilisant une seule
couche de maille fantômes et privilégie le calcul au détriment du
nombre de synchronisations. Lors de la réalisation de vos modules
numériques, il est conceillé de priviligier le même critère.

ARCANE possède son propre service de partionnement parallèle. Ce service 
utilise l'algorithme Métis. Il est utilisé pour l'équilibrage de la charge des
processeurs.

## Calculs séquentiels vs calculs parallèles {#arcanedoc_parallel_intro_seqvspar}

ARCANE utilise le même ordre de numérotation dans chaque 
sous-domaine. Cela permet de limiter au maximum le nombre de
synchronisations. Les mailles, les noeuds et les faces
sont toujours décrits dans le même ordre quel que soit le
sous-domaine et quel que soit le découpage. 

Si toutes les opérations se font en itérant sur un groupe de noeuds ou de
mailles, le résultat est identique en séquentiel et en parallèle.

## Opérations disponibles {#arcanedoc_parallel_intro_operation}

Les opérations proposées par le service de parallélisme sont les suivantes :
- barrières,
- envois / réceptions,
- regroupements,
- réductions,
- diffusion,
- sérialisation (pack / unpack).

Le détail de ces opérations est disponible sur la documentation en ligne de l'interface `Arcane::IParallelMng`.

Notons que les variables du code peuvent appeler directement les opérations de parallélisme qui
sont cohérentes pour leur type. Par exemple une variable de type `Arcane::VariableScalarReal` nommée
`m_density_ratio_maximum` peut appeler l'opération de réduction : 

```cpp
m_density_ratio_maximum.reduce(Parallel::ReduceMax);
``` 

## Implémentation {#arcanedoc_parallel_intro_impl}

Le parallélisme dans ARCANE est conçu comme un service (cf \ref arcanedoc_core_types_service).
Toutes les opérations disponibles (synchronisation, réduction...) sont interfacées
par le service. La seule implémentation de ce service développée à ce jour 
est MPI. 

Pour exécuter un code en parallèle avec MPI, la procédure à suivre dépend de l'implémentation Mpi cible.
Par exemple, si on utilise une version de mpich2, il faut :
- lancer le démon `mpd` dans une fenêtre xterm,
- positionner la variable d'environnement `ARCANE_PARALLEL_SERVICE` à la valeur `Mpi`
- exécuter votre application en tapant la commande : `mpiexec -n nb_proc nom_executable`


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_concurrency
</span>
</div>