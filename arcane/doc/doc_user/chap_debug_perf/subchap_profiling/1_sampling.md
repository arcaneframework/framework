# Analyse de performances par échantillonage {#arcanedoc_debug_perf_profiling_sampling}

[TOC]

\warning Actuellement, le profiling ne fonctionne que sur les
plateformes Linux.

\warning Actuellement, le profiling NE FONCTIONNE
PAS lorsque le multi-threading (que ce soit avec le mécanisme des
tâches ou d'échange de message) est actif.

Le profiling dans %Arcane fonctionne sur un principe
d'échantillonage: à interval régulier, le code est interrompu et on
regarde dans quelle méthode on se trouve.  Si l'interval
d'échantillonage est petit et si le code est exécuté pendant
suffisamment longtemps, on obtient une bonne représentation
statistique de la proportion de temps passé dans les méthodes les
plus couteuses. Les méthodes par échantillonage ne nécessitent pas
l'instrumentation du code et donc ralentissent très peu
l'éxecution. Par contre, elles ne permettent pas de savoir par
exemple combien de fois une méthode est appelée, ni de connaitre
facilement le graphe d'appel. Pour cela, il est
nécessaire d'utiliser des mécanismes comme gprof avec le suppport du
compilateur.


Pour activer le profiling, il faut positionner la variable
d'environnement ARCANE_PROFILING avec une des valeurs suivantes:
- \a Prof . Cela permet d'avoir un profiling peu précis mais qui fonctionne sur
toutes les machines Linux. L'échantillonage est d'une fréquence de
l'ordre de 20Hz et donc il faut faire tourner le code au moins 1
minutes pour avoir des résultats significatifs
- \a Papi. Cela permet d'avoir un profiling très précis, utilisant
les compteurs hardware du processeur. On utilise pour cela la
bibliothèque libre PAPI. Cela ne fonctionne qu'avec les noyaus Linux
récents (2.6.32 ou +) ou les noyaux patchés et il faut que %Arcane
soit compilé avec ce support. Dans ce mode l'échantillonage est
donnée en nombre de cycle d'horloge du processeur. La valeur par
défaut est de 500000 cycles. Pour un processeur à 3GHz, cela fait
donc 6000 échantillons par seconde. Il est possible de changer le
nombre de cycles via la variable d'environnement
ARCANE_PROFILING_PERIOD. Il est préférable de ne pas descendre en
dessous de la valeur par défaut.

Afin de garder des performances raisonnables lors de l'exécution, il
est préférable de ne pas dépasser 100000 échantillons. Il faut donc
ajuster la durée d'exécution du test ou la fréquence
d'échantillonage en fonction de cela.

Lorsqu'il est actif, le profiling commence à la première itération et s'arrête
à la dernière itération de l'exécution. En parallèle, le profiling se fait
pour chaque processeur et %Arcane affiche alors dans le
listing de chaque processeur les informations de profiling de la manière suivante:

```log
*I-Internal    PROCESS_ID = 17977
*I-Internal    NB ADDRESS MAP = 737
*I-Internal    NB FUNC MAP = 53
*I-Internal    NB STACK MAP = 0
*I-Internal    TOTAL STACK = 0
*I-Internal    FUNC EVENT=2798
*I-Internal    FUNC FP=0
*I-Internal    TOTAL EVENT  = 2799
*I-Internal    TOTAL FP     = 1 (nb_giga_flip=0.0005)
*I-Internal    RATIO FP/CYC = 0.000357270453733476
*I-Internal    event     %   function
*I-Internal      1113   39.7      1113      0      0 0  SimpleHydro::ModuleSimpleHydro::computeCQs(Arcane::Real3*, Arcane::Real3*, Arcane::Cell const&)
*I-Internal       613   21.9       613      0      0 0  SimpleHydro::ModuleSimpleHydro::_computeGeometricValues(Arcane::ItemVectorViewT<Arcane::Cell>)
*I-Internal       396   14.1       396      0      0 0  SimpleHydro::ModuleSimpleHydro::_computePressureAndCellPseudoViscosityForces()
*I-Internal       217    7.7       217      0      0 0  SimpleHydro::ModuleSimpleHydro::_applyEquationOfState(Arcane::ItemVectorViewT<Arcane::Cell>)
*I-Internal        87    3.1        87      0      0 0  SimpleHydro::ModuleSimpleHydro::applyBoundaryCondition()
*I-Internal        85    3.0        85      0      0 0  SimpleHydro::ModuleSimpleHydro::_computeViscosityWork(Arcane::ItemVectorViewT<Arcane::Cell>)
*I-Internal        77    2.7        77      0      0 0  SimpleHydro::ModuleSimpleHydro::updateDensity()
*I-Internal        70    2.5        70      0      0 0  SimpleHydro::ModuleSimpleHydro::computeVelocity()
*I-Internal        50    1.7        50      0      0 0  SimpleHydro::ModuleSimpleHydro::computeDeltaT()
*I-Internal        28    1.0        28      0      0 0  SimpleHydro::ModuleSimpleHydro::moveNodes()
*I-Internal        13    0.4        13      0      0 0  Arcane::VariableArrayT<Arcane::Real3>::fill(Arcane::Real3 const&, Arcane::ItemGroup const&)
*I-Internal         4    0.1         4      0      0 0  _IO_vfprintf
*I-Internal         2    0.0         2      0      0 0  Arcane::Timer::Sentry::~Sentry()
*I-Internal         1    0.0         1      0      0 0  SimpleHydro::ModuleSimpleHydro::applyEquationOfState()
```

Les premières lignes sont des informations internes à %Arcane.
Ensuite, est affiché, par ordre décroissant du temps passé, chaque
méthode. Dans l'exemple précédent, on voit qu'on passe 39.7% du temps
dans la méthode ModuleSimpleHydro::computeCQs().

Pour que les résultats soient pertinents, il faut que le code soit
compilé en mode optimisé avec l'inlining activé.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_profiling
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_profiling_mpi
</span>
</div>