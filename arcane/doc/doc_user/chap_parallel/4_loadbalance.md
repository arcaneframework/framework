# Equilibrage de charge sur le maillage {#arcanedoc_parallel_loadbalance}

[TOC]

## Introduction {#arcanedoc_parallel_loadbalance_introduction}

%Arcane dispose d'un mécanisme d'équilibrage de la charge en
redistribuant entre sous-domaines les mailles d'un maillage. Ce
mécanisme gère l'échange des entités du maillage ainsi que les
variables associées. Il est donc en grande partie transparent pour
l'utilisateur.

La gestion de l'équilibrage se fait via deux interfaces :

- \arcane{ICriteriaLoadBalanceMng} qui permet de spécifier les critères à prendre en
  compte pour le calcul de la charge.
- \arcane{IMeshPartitioner} qui permet de déterminer les entités qui doivent
  migrer et d'effectuer la migration.

La classe \arcane{MeshCriteriaLoadBalanceMng} implémentant l'interface
\arcane{ICriteriaLoadBalanceMng} permet, lors de
l'initialisation, de spécifier une ou plusieurs variables aux mailles qui
contiendront le poids de chaque maille pour le calcul de la charge.

\deprecated L'utilisation de `ISubDomain::loadBalance()` pour définir les critères
est maintenant obsolète.

Par exemple :
```cpp
Arcane::VariableCellReal cells_weight(...);
Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh()->handle());
mesh_criteria.addCriterion(cells_weight);
```

\remark L'objet `mesh_criteria` peut être détruit sans problème après utilisation.
Les variables enregistrées le seront encore après sa destruction.

\warning L'appel à la méthode Arcane::MeshCriteriaLoadBalanceMng::reset()
concernera tous les critères ajoutés depuis le début (pour un maillage donné).
Exemple :
```cpp
Arcane::VariableCellReal cells_weight(...);
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh()->handle());
  mesh_criteria.addCriterion(cells_weight);
}
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh()->handle());
  mesh_criteria.reset(); // Ici, le critère représenté par la variable "cells_weight" est aussi retiré.
}
```

Le calcul du poids est du ressort du code utilisateur. Le
partitionneur va ensuite redistribuer le maillage en tentant
d'équilibrer au mieux les poids sur l'ensemble des sous-domaines.
Par exemple, si une méthode couteuse est appelée un nombre
différent de fois pour chaque maille, il est possible de remplir
*cells_weight* avec le nombre d'appels effectué.

En général après un repartitionnement ces variables qui servent de
critères doivent être remises à zéro.

Le repartitionnement et l'équilibrage se font via le service
d'interface IMeshPartitioner. Il est possible d'obtenir une instance
de ce service en spécifiant la ligne suivante dans le fichier 'axl':

```xml
<service-instance
 name    = "partitioner"
 type    = "Arcane::IMeshPartitioner"
 default = "DefaultPartitioner"
/>
```

Dans ce cas, le partitionneur sera accessible via la méthode suivante :

```cpp
options()->partitioner()
```

Pour programmer un repartitionnement au cours du calcul, il faut
appeler ITimeLoopMng::registerActionMeshPartition() en spécifiant le
partitionneur souhaité. Le repartitionnement et sera effectué à la fin de
l'itération courante. Dans un module, on peut donc faire comme cela :

```cpp
subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner());
```

\remark Cet appel n'est valable que pour un pas de temps. Si vous souhaitez
repartitionner à chaque pas de temps (ce qui peut être assez couteux en
nombre de calculs), il est nécessaire d'enregistrer le partitionneur à chaque
pas de temps.

Le repartionnement effectue le transfert de toutes les entités de
maillage et les variables associées. Si le code utilisateur a besoin
de faire d'autres opérations après un équilibrage, il est possible
de spécifier un point d'entrée pour cela. Dans la boucle en temps,
les points d'entrée avec l'attribut 'where="on-mesh-changed"' sont
appelés après un équilibrage. Par exemple :

```xml
<time-loop name="LoadBalanceLoop">
 <modules>...</modules>
 <entry-points where="on-mesh-changed">
  <entry-point name="MyModule.OnMeshChanged"/>
 </entry-points>
</time-loop>
```

\note Actuellement (mars 2017), l'équilibrage de charge ne
fonctionne qu'avec une seule couche de mailles fantômes.

## Multi-maillage {#arcanedoc_parallel_loadbalance_multimesh}

%Arcane gérant le multi-maillage, il est aussi possible d'équilibrer la charge de
plusieurs maillages.

L'équilibrage est indépendant pour chaque maillage (dans le futur, il sera possible
de définir des critères pour équilibrer plusieurs maillages qui nécessitent un
équilibrage "commun").

Prenons deux maillages :

```cpp
IMesh* mesh0 = subDomain().meshes()[0];
IMesh* mesh1 = subDomain().meshes()[1];
```

Et reprenons l'exemple précédent mais avec deux maillages :

```cpp
Arcane::VariableCellReal cells_weight_mesh0(...);
Arcane::VariableCellReal cells_weight_mesh1(...);
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh0->handle());
  mesh_criteria.addCriterion(cells_weight_mesh0);
}
{
  Arcane::MeshCriteriaLoadBalanceMng mesh_criteria = Arcane::MeshCriteriaLoadBalanceMng(subDomain(), mesh1->handle());
  mesh_criteria.addCriterion(cells_weight_mesh1);
}
```

\note Pour créer une variable pour le deuxième maillage, vous pouvez faire :
```cpp
Arcane::VariableCellReal cells_weight_mesh1(VariableBuildInfo(mesh1->handle(), "CellsWeight"))
```

En ce qui concerne les instances du service de partitionneur, il est possible
de spécifier un maillage dans l'axl via l'attribut `mesh-name` :

```axl
<service-instance
 name      = "partitioner0"
 type      = "Arcane::IMeshPartitioner"
 default   = "DefaultPartitioner"
 mesh-name = "Mesh0"
/>

<service-instance
 name      = "partitioner1"
 type      = "Arcane::IMeshPartitioner"
 default   = "DefaultPartitioner"
 mesh-name = "Mesh1"
/>
```

Enfin, pour la programmation du repartitionnement au cours du calcul, il est possible de
faire :

```cpp
subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner0());
subDomain()->timeLoopMng()->registerActionMeshPartition(options()->partitioner1());
```

\note Il est aussi possible de créer les instances du service de partitionneur dans le code
et de programmer leurs appels au cours du calcul :
```cpp
Ref<IMeshPartitionerBase> partitioner0 = ServiceBuilder<IMeshPartitionerBase>::createReference(subDomain(), "DefaultPartitioner", mesh0);
Ref<IMeshPartitionerBase> partitioner1 = ServiceBuilder<IMeshPartitionerBase>::createReference(subDomain(), "DefaultPartitioner", mesh1);
...
subDomain()->timeLoopMng()->registerActionMeshPartition(partitioner0.get());
subDomain()->timeLoopMng()->registerActionMeshPartition(partitioner1.get());
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_simd
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_shmem
</span>
</div>
