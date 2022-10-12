# Fichier de configuration {#arcanedoc_examples_concret_example_config}

[TOC]

Pour commencer, voici le fichier de configuration :

```xml
<?xml version="1.0" ?>
 <arcane-config code-name="Quicksilver">
  <time-loops>
    <time-loop name="QAMALoop">
      <title>QS</title>
      <description>Default timeloop for code Quicksilver Arcane MiniApp</description>

      <singleton-services>
        <service name="SimpleCsvOutput" need="required" />
        <service name="SimpleCsvComparator" need="required" />
        <service name="RNG" need="required" />
      </singleton-services>

      <modules>
        <module name="QS" need="required" />
        <module name="SamplingMC" need="required" />
        <module name="TrackingMC" need="required" />
      </modules>

      <entry-points where="init">
        <entry-point name="QS.InitModule" />
        <entry-point name="SamplingMC.InitModule" />
        <entry-point name="TrackingMC.InitModule" />
        <entry-point name="QS.StartLoadBalancing" />
      </entry-points>

      <entry-points where="compute-loop">
        <entry-point name="SamplingMC.CycleSampling" />
        <entry-point name="TrackingMC.CycleTracking" />
        <entry-point name="QS.CycleFinalize" />
        <entry-point name="SamplingMC.CycleFinalize" />
        <entry-point name="TrackingMC.CycleFinalize" />
        <entry-point name="QS.LoopLoadBalancing" />
      </entry-points>

      <entry-points where="on-mesh-changed">
        <entry-point name="QS.AfterLoadBalancing" />
      </entry-points>

      <entry-points where="exit">
        <entry-point name="SamplingMC.EndModule" />
        <entry-point name="TrackingMC.EndModule" />
        <entry-point name="QS.CompareWithReference" />
        <entry-point name="QS.EndModule" />
      </entry-points>

    </time-loop>
  </time-loops>
</arcane-config>
```
Avec ce fichier, on peut déjà voir à quoi ressemble `QAMA`.
On retrouve nos trois modules `QS`, `SamplingMC` et `TrackingMC` dans
la partie `<modules>`.
On retrouve aussi les trois types de points d'entrées que l'on avait vu
dans l'exemple `HelloWorld` : `init`, `compute-loop` et `exit`
(ici : \ref arcanedoc_examples_simple_example_module_sayhelloaxl).

\note
Dans `HelloWorld`, il n'y avait qu'un seul point d'entrée par type de point d'entrée
donc il n'y avait pas à s'inquiéter de l'ordre. Ici, on en a plusieurs.
Il est donc important de noter que l'ordre des points d'entrée est important et
pris en compte. En revanche, l'ordre des types de points d'entrées n'est pas important.
```xml
<!-- 1) -->
<entry-points where="compute-loop">
  <entry-point name="SamplingMC.CycleSampling" />
  <entry-point name="TrackingMC.CycleTracking" />
</entry-points>

<entry-points where="init">
  <entry-point name="QS.InitModule" />
  <entry-point name="SamplingMC.InitModule" />
</entry-points>
```
donne la même chose que :
```xml
<!-- 2) -->
<entry-points where="init">
  <entry-point name="QS.InitModule" />
  <entry-point name="SamplingMC.InitModule" />
</entry-points>

<entry-points where="compute-loop">
  <entry-point name="SamplingMC.CycleSampling" />
  <entry-point name="TrackingMC.CycleTracking" />
</entry-points>
```
mais est different de :
```xml
<!-- 3) -->
<entry-points where="init">
  <entry-point name="SamplingMC.InitModule" /> <!-- ici -->
  <entry-point name="QS.InitModule" />         <!-- ici -->
</entry-points>

<entry-points where="compute-loop">
  <entry-point name="SamplingMC.CycleSampling" />
  <entry-point name="TrackingMC.CycleTracking" />
</entry-points>
```


____

Dans les nouveautés, on a d'abord la partie `<singleton-services>`.
On retrouve le service vu dans la section précédente : `RNG`. Nous avons
aussi deux autres services : `Arcane::SimpleCsvOutput` et `Arcane::SimpleCsvComparator`.
Ces services sont des services inclus dans le framework %Arcane et peuvent
donc être utilisés par n'importe quelle application.

Leurs interfaces sont disponibles dans cette documentation :
`Arcane::ISimpleTableOutput`, `Arcane::ISimpleTableComparator` et `Arcane::IRandomNumberGenerator`.

Comme pour le service `RNG`, il est possible de créer une implémentation
spécifique à notre application en utilisant l'interface de ces services.

____

Il y a deux façons d'utiliser un service : 
- en tant que service normal, que l'on doit déclarer dans le `.axl` 
d'un module. Dans ce cas, il y aura un objet par module l'ayant déclaré
et il ne sera pas partagé.
- en tant que singleton, que l'on doit déclarer dans le `.config` du projet.
Dans ce cas, il n'y aura qu'un seul objet par projet. Les modules pourront
récupérer un pointeur vers cet objet unique.

Dans QAMA, j'ai choisi la méthode singleton. Pour le service `SimpleCsvOutput`,
il est nécessaire de partager l'objet pour générer un unique tableau CSV.
Pour le service `RNG`, peu importe.

\warning Dans le cas du singleton, il est impossible de récupérer des données
dans le jeu de données de manière autonome. Il est néanmoins possible de faire
transiter les données par un des modules présents. Dans le cas de Quicksilver,
c'est le module QSModule qui est chargé de faire cela.
Il est possible de déterminer, pour un service, s'il est considéré comme
un singleton avec cette ligne :
```cpp
option() == null;
```
S'il y a des options, alors on est dans un service classique ; sinon c'est
que l'on est dans un singleton.

____

On a aussi un nouveau type de point d'entrée : `on-mesh-changed`.
Ce type de point d'entrée se déclenche lorsque le maillage change,
lors d'un repartitionnement par exemple.
Ce point d'entrée s'exécute après les points d'entrées de type `compute-loop`.
\warning
En pratique, si l'on récupère `m_global_iteration()` dans un point d'entrée
de type `compute-loop`, on aura l'itération **i** (exemple : `QS.LoopLoadBalancing`),
puis dans le point d'entrée suivant de type `on-mesh-changed`, on aura 
l'itération **i+1** (exemple : `QS.AfterLoadBalancing`).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example_struct
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example_rng
</span>
</div>