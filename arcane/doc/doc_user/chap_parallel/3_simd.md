# Vectorisation {#arcanedoc_parallel_simd}

[TOC]

<!-- décrit l'utilisation de la vectorisation (SIMD). -->

## Introduction {#arcanedoc_parallel_simd_intro}

La vectorisation est un mécanisme permettant d'exécuter la même
instruction sur plusieurs données. Le terme anglais couramment
utilisé pour qualifier la vectorisation est <strong>Single
Instruction Multiple Data (SIMD)</strong>. Comme il s'agit d'une
instruction gérée directement par le processeur, les opérations
possibles sont assez limitées. En général, il s'agit des opérations
arithmétiques de base (addition, soustraction, ...) ainsi que
les fonctions mathématiques classiques (minimum, maximum, valeur
absolue, ...). Les opérations mathématiques complexes (comme le
logarithme, l'exponentielle, ...) ne sont en général pas des
instructions natives vectorielles.

Les processeurs récents permettent tous de faire de la
vectorisation. Par contre, les tailles de vecteur et les opérations
possibles sont différentes d'un processeur à l'autre.

Par exemple, la boucle simple suivante effectue \b n additions :

```cpp
using namespace Arcane;
UniqueArray<Real> a, b, c;
for( int i=0; i<n; ++i ){
  a[i] = b[i] + c[i];
}
```

Avec un processeur scalaire, les registres ne contiennent qu'un seul
réel et les instructions d'addition opèrent donc sur un seul
réel. Il faudra \b n instructions d'addition pour faire ce
calcul. Un processeur vectoriel dispose de registres contenant
plusieurs réels. Pour des registres contenant \b P réels, le nombre
d'instructions d'addition nécessaire est donc \b n/P. Si les
instructions scalaires et vectorielles prennent le même temps, on a
donc une accélération théorique d'un facteur \b P. Plus les registres
sont grands, plus l'intéret potentiel de la vectorisation est
important. Bien entendu, dans la pratique c'est souvent moins rose
et le gain réel dépend d'autres facteurs comme la bande
passante mémoire, le pipelining, ...

Pour exploiter la vectorisation, il existe deux
possibilités (qui sont compatibles) :

- laisser le compilateur gérer la vectorisation.
- utiliser des classes C++ spécifiques conçues pour cela.

La première solution est la plus simple car elle ne nécessite pas de
changer le code. Elle est directement disponible via les bonnes
options du compilateur. La contrepartie de cette simplicité est
qu'il est souvent difficile pour le compilateur de détecter les
endroits où la vectorisation est possible. Le code généré est donc
rarement vectorisé. La seconde méthode garantit l'exploitation de la
vectorisation mais elle nécessite de réécrire le code. %Arcane
propose un ensemble de classes pour exploiter cette seconde
méthode.

Le principe est donc de fournir une classe vectorielle correspondant
à une classe scalaire. La classe vectorielle contiendra donc \a N
valeurs scalaires, avec \a N dépendant du type de vectorisation disponible.

Même si en théorie la vectorisation peut s'appliquer sur tous les
types simples (short, int, long, float, ...), on se limite dans
%Arcane à fournir des classes gérant la vectorisation que pour les
types Arcane::Real et dérivés (Arcane::Real2, Arcane::Real3).

Actuellement, %Arcane fournit les types vectoriels suivants :

<table>
<tr>
<th>Type scalaire</th>
<th>Type vectoriel</th>
<th>Fichier de définition</th>
</tr>
<tr>
<td>Arcane::Real</td>
<td>Arcane::SimdReal</td>
<td>

```cpp
#include "arcane/utils/Simd.h"
```
</td>
<tr>
<td>Arcane::Real2</td>
<td>Arcane::SimdReal2</td>
<td></td>
</tr>
<tr>
<td>Arcane::Real3</td>
<td>Arcane::SimdReal3</td>
<td></td>
</tr>
<tr>
<td>Arcane::Item, Arcane::Cell, Arcane::Face, ...</td>
<td>Arcane::SimdItem, Arcane::SimdCell, Arcane::SimdFace, ...</td>
<td>

```cpp
#include "arcane/SimdItem.h"
```
</td>
</tr>
</table>

\note Pour l'instant, les classes Real2x2 et Real3x3 ne disposent
pas d'une classe vectorielle associée mais cela sera disponible dans
une version ultérieure de %Arcane.

## Utilisation des classes vectorielles {#arcanedoc_parallel_simd_usage}

L'utilisation des classes SIMD est similaire à l'usage scalaire. Il
suffit en général de changer le nom des classes scalaires par le nom
vectoriel correspondant.

\note L'utilisation de la vectorisation suppose l'utilisation des
vues sur les variables. Il n'est pas possible d'accéder directement
à une variable via les classes Arcane::SimdItem et dérivées.

L'exemple suivant montre comment passer d'une écriture scalaire à
une écriture vectorielle :

```cpp
using namespace Arcane;

// Déclaration des variables
VariableCellReal pressure = ...;
VariableCellReal density = ...;
VariableCellReal adiabatic_cst = ...;
VariableCellReal internal_energy = ...;
VariableCellReal sound_speed = ...;

// Vues en entrée (lecture)
// En C++11, il est aussi possible d'utiliser le mot clé 'auto':
// auto in_pressure = viewIn(pressure);
VariableCellRealInView in_pressure = viewIn(pressure);
VariableCellRealInView in_density = viewIn(m_density);
VariableCellRealInView in_adiabatic_cst = viewIn(adiabatic_cst);

// Vues en sortie (écriture)
VariableCellRealOutView out_internal_energy = viewOut(internal_energy);
VariableCellRealOutView out_sound_speed = viewOut(sound_speed);

// Version scalaire
ENUMERATE_CELL(icell,allCells()){
  Cell vi = *icell;
  Real pressure = in_pressure[vi];
  Real adiabatic_cst = in_adiabatic_cst[vi];
  Real density = in_density[vi];
  out_internal_energy[vi] = pressure / ((adiabatic_cst-1.0) * density);
  out_sound_speed[vi] = math::sqrt(adiabatic_cst*pressure/density);
}

// Version vectorielle
ENUMERATE_SIMD_CELL(icell,allCells()){
  SimdCell vi = *icell;
  SimdReal pressure = in_pressure[vi];
  SimdReal adiabatic_cst = in_adiabatic_cst[vi];
  SimdReal density = in_density[vi];
  out_internal_energy[vi] = pressure / ((adiabatic_cst-1.0) * density);
  out_sound_speed[vi] = math::sqrt(adiabatic_cst*pressure/density);
}
```

La vectorisation fonctionne bien tant que tous les éléments du
vecteur doivent effectuer la même opération. Les choses se
compliquent lorsque cela n'est plus le cas. Notamment, tout ce qui
dépend d'une condition est difficilement vectorisable. Il existe
aussi des cas où on souhaite faire dans une boucle vectorielle des
opérations spécifiques pour chacun des éléments. Pour gérer cette
situation, il est possible d'ajouter des sections séquentiels en
itérant sur les entités d'un Arcane::SimdItem via les macros ENUMERATE_*.
Par exemple :

```cpp
using namespace Arcane;
ENUMERATE_SIMD_CELL(ivcell,allCells()){
  SimdCell simd_cell = *ivcell; // Vecteur de mailles
  ENUMERATE_CELL(icell,ivcell){
    Cell cell = *icell;
    info() << "Cell: local_id=" << cell.localId();
  }
}
```

Enfin, il est possible de connaître le nombre de réels d'un registre
vectoriel via la constante SimdReal::BLOCK_SIZE. Cela permet par
exemple d'itérer sur les éléments d'un registre vectoriel :

```cpp
using namespace Arcane;
SimdReal3 vr;
for( Integer i=0, n=SimdReal::BLOCK_SIZE; i<n; ++i ){
  Real3 r = vr[i];
  info() << " R [" << i << "] = " << r;
}
```

## Gestion de l'alignement {#arcanedoc_parallel_simd_alignment}

En général, et c'est le cas pour les processeurs x64, l'utilisation
de la vectorisation nécessite que les données en mémoires soient
alignées d'une manière plus restrictive que pour types
scalaires. Pour le SSE, l'AVX et l'AVX512 L'alignement minimal est
égal à la taille en octet du vecteur Simd. Donc par exemple pour
l'AVX avec des vecteurs de 256 bits, soit 32 octets, l'alignement
minimal est de 32 octets. Pour simplifier la vectorisation %Arcane
garantit que les types suivants ont l'alignement minimal souhaité
pour la vectorisation :
- les localIds() des Arcane::ItemGroup.
- les données des variables tableaux et scalaires sur le maillage.
- les données des variables tableaux 2D et variables tableaux sur le
maillage. \`A noter que pour ces dernières le début du tableau est
aligné mais si la première dimension n'est pas un multiple de la
taille du vecteur alors les éléments suivants ne seront pas alignés
car il n'y a pas encore de gestion du padding).

Le C++ ne permettant pas d'allouer via new/delete avec alignement,
%Arcane fournit la classe Arccore::AlignedMemoryAllocator qui peut être
utilisée avec les classes Arcane::UniqueArray et Arcane::SharedArray pour garantir
l'alignement. Par exemple :

```cpp
using namespace Arcane;
UniqueArray x(AlignedMemoryAllocator::Simd());
x.resize(32); // Garanti que \a x à un alignement correct.
```

## Gestion des fins de boucle {#arcanedoc_parallel_simd_endloop}

La vectorisation fonctionne bien lorsque le nombre d'éléments de la
boucle est un multiple de la taille du vecteur Simd. Si ce n'est pas
le cas, il faut traiter la dernière partie de la boucle d'une
certaine manière. <strong>Afin d'offrir un mécanisme identique pour tous
les types de vectorisation, %Arcane duplique dans le vecteur Simd la
dernière valeur valide</strong>.
Par exemple, on suppose le code suivant :

```cpp
using namespace Arcane;
CellGroup cells = ...
ENUMERATE_SIMD_CELL(ivcell,cells){
  SimdCell simd_cell = *ivcell; // Vecteur de mailles
}
```

Avec \a cells un groupe de mailles qui contient 11 éléments. Si on suppose que
la taille d'un vecteur est 8, alors la boucle précédente fera deux
itérations. Pour la première on aura les valeurs suivantes de \a simd_cell
  
```cpp
// Première itération
simd_cell[0]  = cells[0];
simd_cell[1]  = cells[1];
simd_cell[2]  = cells[2];
simd_cell[3]  = cells[3];
simd_cell[4]  = cells[4];
simd_cell[5]  = cells[5];
simd_cell[6]  = cells[6];
simd_cell[7]  = cells[7];
```

Pour la deuxième itération, comme \a cells ne contient que 11
éléments, on répète dans \a simd_cell la dernière valeur valide :

```cpp
// Deuxième itération
simd_cell[8]  = cells[8];
simd_cell[9]  = cells[9];
simd_cell[10] = cells[10];
simd_cell[11] = cells[10]; // Répète la dernière valeur valide.
simd_cell[12] = cells[10];
simd_cell[13] = cells[10];
simd_cell[14] = cells[10];
simd_cell[15] = cells[10];
```

Ce mécanisme fonctionne partaitement tant que les opérations
effectuées sont bien vectorielles. Si ce n'est pas le cas, il est
possible d'itérer uniquement sur les valeurs valides comme suit :

```cpp
using namespace Arcane;
CellGroup cells = ...
ENUMERATE_SIMD_CELL(ivcell,cells){
  SimdCell simd_cell = *ivcell; // Vecteur de mailles
  ENUMERATE_CELL(icell,ivcell){
    Cell cell = *icell;
    info() << "Cell: local_id=" << cell.localId();
  }
}
```

Avec l'exemple précédent, la boucle interne ne fera que 3 itérations,
(pour les mailles \a cells[8], \a cells[9] et \a cells[10]) pour la
dernière partie de \a cells.

## Opérations supportées {#arcanedoc_parallel_simd_operation}

Les opérations mathématiquees supportés par les classes vectorielles de %Arcane
sont définies dans le fichier SimdMathUtils.h:

```cpp
#include "arcane/SimdMathUtils.h"
```

%Arcane fournit pour les classes vectorielles Arcane::SimdReal,
Arcane::SimdReal2 et Arcane::SimdReal3 les mêmes opérations que celles
disponibles dans MathUtils.h pour la version scalaire à l'exception de
\a min et \a max.

## Mécanismes de vectorisation supportés {#arcanedoc_parallel_simd_support}

Dans la version 2.2, %Arcane ne supporte que la vectorisation pour
les processeurs d'architecture x86.

Pour ces processeurs, il existe (actuellement) trois
générations de vectorisation :

- la vectorisation SSE, qui est disponible sur tous les processeurs 64
  bits et qui utilisent des registres de 128 bits.
- la vectorisation AVX, qui est disponible sur les processeurs
  depuis la génération SandyBridge (en gros depuis 2012). Ces vecteurs ont
  une taille de 256 bits.
- la vectorisation AVX512, qui est disponible sur les processeurs de
  génération SkyLake (2015+) et qui dispose de vecteurs de 512
  bits. Cette vectorisation est supportée depuis la version 2.3.9 de %Arcane.

Suivant la plateforme, plusieurs mécanismes peuvent être
disponibles. Sur les processeurs Intel les processeurs ont une
compatibilité ascendante et donc ceux qui supportent l'AVX512
supportent aussi l'AVX et le SSE. De même, les processeurs avec AVX
supportent le SSE.

%Arcane définit le mécanisme par défaut comme étant celui qui utilise
la vectorisation la plus importante. Les types Arcane::SimdInfo,
Arcane::SimdReal, Arcane::SimdReal3 sont donc des typedefs qui
dépendent de la plateforme.

%Arcane définit aussi des macros indiquant les mécanismes
disponibles :

- ARCANE_HAS_SSE si la vectorisation avec SSE est disponible
- ARCANE_HAS_AVX si la vectorisation avec AVX ou AVX2 est disponible
- ARCANE_HAS_AVX512 si la vectorisation avec AVX512 est disponible.



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_concurrency
</span>
<span class="next_section_button">
\ref arcanedoc_parallel_loadbalance
</span>
</div>
