# Comparaison bit à bit {#arcanedoc_debug_perf_compare_bittobit}

[TOC]

Cette page décrit les outils disponibles dans ARCANE pour effectuer
des comparaisons bit à bit des valeurs des variables gérées par
ARCANE. Les comparaisons suivantes sont possibles :
- \ref arcanedoc_debug_perf_compare_bittobit_two_exec
- \ref arcanedoc_debug_perf_compare_bittobit_synchronization
- \ref arcanedoc_debug_perf_compare_bittobit_replica

Il est important de noter que la comparaison n'est possible
que sur les variables gérées par %Arcane (comme #VariableCellReal,
#VariableScalarInt32, ...). Dans l'implémentation actuelle, seules
les variables ayant un type de donnée numérique sont comparées (donc
par exemple par les types de données 'String').

## Comparaison bit à bit entre deux exécutions {#arcanedoc_debug_perf_compare_bittobit_two_exec}

Ce mécanisme permet de déterminer la liste des variables %Arcane qui sont différentes
entre deux exécutions. Il ne fonctionne que sur les variables %Arcane.

Le principe de fonctionnement est le suivant :
- Exécution d'un cas de référence, avec sauvegarde des résultats.
- Exécution d'un deuxième cas et comparaison pendant l'exécution avec le cas de
référence.

\note Comme il faut d'abord exécuter un cas de référence et sauver les
résultats sur disque, les données sauvegardées peuvent être très
volumineuses suivant les cas.

\note avant la version 1.22.2 de %Arcane, seules les variables sur
les entités du maillage étaient comparées. Depuis cette version, les
variables tableaux qui ne reposent pas sur une entité du maillage
sont aussi comparées.

Toutes les variables %Arcane sont comparées sauf celles qui ont la
propriété IVariable::PExecutionDepend à vrai. En comparaison
parallèle/sequentiel, celles qui ont aussi la propriété
IVariable::PSubDomainDepend ne sont pas comparées.

### Exécution de la référence {#arcanedoc_debug_perf_compare_bittobit_exec}

Pour sauvegarder les informations de référence, il suffit de lancer
le cas après avoir positionné la variable d'environnement **STDENV_VERIF** à la valeur
**WRITE**. Dans ce cas, les valeurs de toutes les variables seront
sauvées à chaque itération. Il est possible de le faire pour chaque
point d'entrée en positionnant la variable d'environnement
**STDENV_VERIF_ENTRYPOINT** mais cela augmente beaucoup le volume des
informations à sauver.

Par défaut, les informations sont sauvées dans le répertoire
\c /tmp/$USER/verif  mais il est possible de changer cela en spécifiant
un autre chemin dans la variable d'environnement **STDENV_VERIF_PATH**

\note Dans le cas d'une comparaison pour un cas parallèle, il faut
être certain que le chemin utilisé pour les données soit accessible
de l'ensemble des noeuds du calculateur, ce qui n'est généralement
pas le cas du répertoire \c /tmp.

Depuis la version 3.16 de %Arcane, il est possible de chosir la
méthode de comparaison utilisée pour calculer la différence. Cela se
fait via la variable d'environnement **STDENV_VERIF_DIFF_METHOD**. Les
valeurs possibles sont :

- `RELATIVE` : calcule la différence relative `(v-ref) / ref`. C'est la
  comparaison utilisée par défaut.
- `LOCALNORMMAX` calcule la différence `(v-ref) / max_ref` avec
  `max_ref` la valeur absolue du maximum des valeurs de référence sur
  le sous-domaine. A noter qu'avec cette méthode la différence dépend
  du découpage.

### Comparaison avec la référence {#arcanedoc_debug_perf_compare_bittobit_compare}

Une fois la référence exécutée, il suffit de positionner la variable
d'environnement **STDENV_VERIF** à `READ` et de lancer une nouvelle
exécution. Il est possible de changer le nombre de sous-domaines par
rapport à l'exécution de référence et ainsi faire des comparaisons
entre parallèle et séquentiel (dans ce cas, l'exécution séquentielle
doit être celle de référence et être exécutée en premier) ou des comparaisons parallèles/parallèles.

Dans le listing de l'exécution, apparait alors pour chaque
itération (ou pour chaque point d'entrée si **STDENV_VERIF_ENTRYPOINT**
est définie) la liste des variables qui sont différentes entre cette
exécution et la référence, ainsi que leurs valeurs, comme suit :

```log
*I-TimeLoopMng Processor 3 : 50 entité(s) ayant des valeurs différentes pour la variable CaracteristicLength:
50 entité(s) ayant des valeurs différentes pour la variable CaracteristicLength
VDIFF: Variable 'CaracteristicLength' (G) uid=1264 lid=673 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1263 lid=672 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1250 lid=659 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1272 lid=681 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1251 lid=660 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (G) uid=1252 lid=661 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1264 lid=39 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1263 lid=38 val: 0.00495968464432893 réf: 0.00495967741935477 rdiff: -1.45674275618805e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1250 lid=25 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
VDIFF: Variable 'CaracteristicLength' (O) uid=1272 lid=47 val: 0.00495968464432892 réf: 0.00495967741935477 rdiff: -1.45674275601317e-06
```

Pour chaque variable et chaque sous-domaine est indiqué son nom, le Item::uniqueId() et le
Item::localId() de l'entité ainsi que si elle est fantôme ((G)) ou
appartenant au sous-domaine ((O)) , la valeur actuelle (val), la valeur de
référence (ref) et la différence relative (rdiff). Pour ne pas
alourdir le listing, seules les 10 différences les plus importantes
en valeur absolue sont affichées.

Lorsque la variable est une variable tableau et n'est pas sur une
entité du maillage, le (G) ou (O) n'apparait pas et au lieu du
numéro local de l'entité, c'est l'indice de l'élément dans le
tableau qui est affiché.

En parallèle, il peut être normal que les valeurs sur les mailles
fantômes soient différentes de la référence si la variable n'est
pas synchronisée. Comme cela peut être le cas pour de nombreuses
variables, il est possible de n'afficher les différences que sur
les mailles appartenant au sous-domaine, en positionnant la
variable d'environnement **STDENV_VERIF_SKIP_GHOSTS**.

## Vérification des comparaisons {#arcanedoc_debug_perf_compare_bittobit_verification}

### Vérification des synchronisations {#arcanedoc_debug_perf_compare_bittobit_synchronization}

De la même manière qu'il est possible de faire des comparaisons bit
à bit, il est possible de vérifier que les variables sont bien
synchronisées entre les sous-domaines. Pour cela, il suffit de
spécifier la valeur **CHECKSYNC** à la variable d'environnement
**STDENV_VERIF**. Les valeurs avec l'attribut *IVariable::PNoNeedSync* et
les variables partielles ne sont pas comparées.

### Vérification des valeurs entre replica {#arcanedoc_debug_perf_compare_bittobit_replica}

Il est aussi possible de vérifier que les valeurs d'une variable
sont les mêmes sur tous les replica d'un sous-domaine. Pour cela,
il faut spécifier la valeur **CHECKREPLICA** à la variable
d'environnement **STDENV_VERIF**. Les variables avec l'attribut
*IVariable::PNoReplicaSync* et les variables partielles ne sont pas
comparées.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_debug_perf_check_memory
</span>
<span class="next_section_button">
\ref arcanedoc_debug_perf_compare_synchronization
</span>
</div>
