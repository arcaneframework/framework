# Boucles sur les entités des matériaux et des milieux {#arcanedoc_materials_loop}

[TOC]

Cette page décrit la gestion des boucles sur les entités des matériaux et des milieux.

Dans le reste de cette page, on utilise le terme générique \a
composant pour décrire un matériau ou un milieu.

Les entités d'un composant peuvent se répartir en deux parties : les
entités pures et les entités impures. Par définition, les entités qui
ne sont pas pures sont impures. La notion de pure varie suivant le
type du composant :
- pour un milieu, une entité est pure s'il n'y a qu'un milieu dans
cette entité.
- pour un matériau, une entité est pure s'il n'y a qu'un
seul matériau <b>ET</b> qu'un seul milieu.

Au niveau du rangement mémoire pour une variable donnée, accéder à
une entité pure revient à accéder à la valeur globale de cette
variable.

## Généralisation des boucles {#arcanedoc_materials_loop_loop}

Depuis la version 2.7.0 de %Arcane, La macro générique
ENUMERATE_COMPONENTITEM() permet d'itérer sur les entités d'un
composant de manière globale ou par partie (pure/impure). Elle peut remplacer les
macros ENUMERATE_COMPONENTCELL(),
ENUMERATE_MATCELL() et ENUMERATE_ENVCELL().

Les valeurs suivantes sont disponibles pour l'itération :

ENUMERATE_COMPONENTITEM(MatCell,icell,container) avec container de
type IMeshMaterial* ou MatCellVector.


Il est possible d'itérer uniquement sur la partie pure ou impure d'un
composant.

\note Actuellement, l'ordre de parcours des boucles par partie pure
ou impure n'est pas défini et pourra évoluer par la suite. Cela
signifie que s'il y a des dépendances entre les itérations de la
boucle le résultat peut varier d'une exécution à l'autre.

Les exemples suivants montrent les différentes variantes de la macro
ENUMERATE_COMPONENTITEM()

### Boucles sur les milieux {#arcanedoc_materials_loop_envloop}

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateComponentItemEnv

### Boucles sur les matériaux {#arcanedoc_materials_loop_matloop}

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateComponentItemMat

### Boucles génériques sur les composants {#arcanedoc_materials_loop_componentloop}

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateComponentItemComponent

## Boucles vectorielles sur les composants {#arcanedoc_materials_loop_simdloop}

\note Dans la version actuelle de %Arcane (2.7.0), les boucles
vectorielles ne sont supportées que sur les milieux (mais pas encore sur les
matériaux).

Pour pouvoir utiliser la vectorisation sur les composants, il faut
inclure le fichier suivant :

```cpp
#include <arcane/materials/ComponentSimd.h>

using namespace Arcane::Materials;
```

Il est nécessaire d'utiliser le mécanisme des lambda du C++11 pour
itérer sur les composants via des itérateurs vectoriels. Cela se fait
via la macro suivante :

```cpp
ENUMERATE_COMPONENTITEM_LAMBDA(){
};
```

\warning Il ne faut surtout pas oublier le point virgule ';'
final. Pour plus d'informations, se reporter à la documentation de
cette macro.

\note Ce mécanisme est expérimental et pourra évoluer par la suite.

Par exemple, avec les déclarations suivantes des variables :

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateVariableDeclaration

Il est possible d'utiliser les boucles vectorielles comme suit :

\snippet MeshMaterialTesterModule_Samples.cc SampleEnumerateSimdComponentItem

\warning Pour des raisons de performance, l'ordre des itérations peut
être quelconque. Il est donc indispensable qu'il n'y ait pas de
relations entre les itérations. En particulier, si des opérations non
associatives telles que des sommes sur des réels sont utilisées,
alors le résultat peut varier entre deux exécutions.

\note L'implémentation actuelle comporte plusieurs limitations :
- Il n'est pas encore possible d'utiliser ces énumérateurs avec
les boucles concurrentes (voir page \ref arcanedoc_materials_manage_concurrency).
- Pour le SIMD, il faut obligatoirement utiliser les vues.
- Pour l'instant les vues ne sont disponibles que pour les
variables scalaires.
*/


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_materials_manage
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_materials_loop
</span> -->
</div>
