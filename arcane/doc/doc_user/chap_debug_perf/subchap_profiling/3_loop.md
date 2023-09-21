# Analyse de performances des boucles {#arcanedoc_debug_perf_profiling_loop}

[TOC]

Il est possible de faire du profilage des boucles gérées par
%Arcane. Cela concerne toutes les boucles telles que ENUMERATE_(),
ENUMERATE_CELL(), RUNCOMMAND_ENUMERATE(), RUNCOMMAND_LOOP(). Le
profiling fonctionne pour le code séquentiel, multi-threadé ou avec
les accélérateurs. Dans ce mode, on mesure le temps pris pour exécuter
chacune des boucles et on affiche en fin d'exécution les informations
cumulées sur ces boucles.

Pour avoir les informations de profiling, il suffit de positionner la
variable d'environnement `ARCANE_LOOP_PROFILING_LEVEL`. Les deux
valeurs possibles sont :

- `1` pour activer le profiling de base.
- `2` pour activer le profiling comme le mode `1`. La différence avec
  ce mode est qu'on utilise les évènements pour calculer le temps
  passé dans les noyaux accélérateurs. Ce mode est plus précis que le
  mode `1` mais peut occasionner un léger surcout sur le temps de
  calcul (de l'ordre du pourcent).

Il est possible aussi de positionner de manière programatique le
profilage en appelant la méthode
\arcane{ProfilingRegistry::setProfilingLevel()}. Il est possible
d'activer et de désactiver le profiling à n'importe quel moment en
dehors des boucles.

Lorsque le profiling est activé, des informations sont affichées en
fin de calcul.

Voici un exemple de résultat sur accélérateur :

```
*I-Internal   LoopStatistics:
LoopStat: global_time (ms) = 42.7763
LoopStat: global_nb_loop   =        752 time=56883.4
LoopStat: global_nb_chunk  =          0 time=0
ProfilingStat
     Ncall    Nchunk     T (ms)  Tck (ns)     %  name
        51         0     21.805         0  50.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeGeometricValues()
       100         0      8.551         0  19.9  void SimpleHydro::SimpleHydroAcceleratorService::_computePressureAndCellPseudoViscosityForces()
       300         0      4.185         0   9.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyBoundaryCondition()
        50         0      2.242         0   5.2  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyEquationOfState()
        50         0      2.026         0   4.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeViscosityWork()
        50         0      1.522         0   3.5  virtual void SimpleHydro::SimpleHydroAcceleratorService::updateDensity()
        50         0      0.866         0   2.0  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeDeltaT()
        50         0      0.679         0   1.5  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeVelocity()
        50         0      0.557         0   1.3  virtual void SimpleHydro::SimpleHydroAcceleratorService::moveNodes()
         1         0      0.337         0   0.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::hydroStartInit()
```

et un exemple en multi-thread :

```
*I-Internal   LoopStatistics:
LoopStat: global_time (ms) = 141.137
LoopStat: global_nb_loop   =       1504 time=93841.2
LoopStat: global_nb_chunk  =      35028 time=4029.27
ProfilingStat
     Ncall    Nchunk     T (ms)  Tck (ns)     %  name
       102      3264     80.491     24660  57.0  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeGeometricValues()
       200      8000     34.268      4283  24.2  void SimpleHydro::SimpleHydroAcceleratorService::_computePressureAndCellPseudoViscosityForces()
       100      3200      7.032      2197   4.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeViscosityWork()
       100      3200      6.807      2127   4.8  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyEquationOfState()
       100      4800      2.818       587   1.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeVelocity()
       100      4800      2.733       569   1.9  virtual void SimpleHydro::SimpleHydroAcceleratorService::moveNodes()
       600      1300      2.502      1925   1.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::applyBoundaryCondition()
       100      3200      2.410       753   1.7  virtual void SimpleHydro::SimpleHydroAcceleratorService::updateDensity()
       100      3200      1.875       586   1.3  virtual void SimpleHydro::SimpleHydroAcceleratorService::computeDeltaT()
         2        64      0.196      3064   0.1  virtual void SimpleHydro::SimpleHydroAcceleratorService::hydroStartInit()
```


Voici la signification des champs :

- `Ncall` : nombre de fois que la boucle est exécuté
- `Nchunk` : nombre de partitions (chunks) de boucles en mode
  multi-thread.
- `T` : temps total (en milli-seconde) passé dans l'exécution de la boucle. En
  multi-thread il s'agit du temps total cumulé sur tous les threads.
- `Tck` : temps par chunk. Cette valeur n'est valide que pour les
  exécutions en multi-thread.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_profiling_mpi
</span>
</div>
