# Equilibrage de charge sur le maillage {#arcanedoc_parallel_loadbalance}

%Arcane dispose d'un mécanisme d'équilibrage de la charge en
redistribuant entre sous-domaines les mailles d'un maillage. Ce
mécanisme gère l'échange des entités du maillage ainsi que les
variables associées. Il est donc en grande partie transparent pour
l'utilisateur.

La gestion de l'équilibrage se fait via deux interfaces :
- ILoadBalanceMng qui permet de spécifier les critères à prendre en
  compte pour le calcul de la charge.
- IMeshPartitioner qui permet de déterminer les entités qui doivent
  migrer et d'effectuer la migration.

La méthode ISubDomain::loadBalanceMng() permet de récupèrer une
instance de ILoadBalanceMng. Le code utilisateur doit lors de
l'initialisation spécifier une ou plusieurs variables aux mailles qui
contiendront le poids de chaque maille pour le calcul de la charge.

Par exemple :
```cpp
VariableCellReal cells_weight(...);
ILoadBalanceMng* lb = subDomain()->loadBalanceMng();
lb->addCriterion(cells_weight);
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


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_simd
</span>
<!-- <span class="next_section_button">
\ref 
</span> -->
</div>
