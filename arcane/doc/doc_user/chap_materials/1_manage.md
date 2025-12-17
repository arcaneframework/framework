#  Gestion des matériaux et des milieux. {#arcanedoc_materials_manage}

[TOC]

Cette page décrit la gestion des matériaux et des milieux dans %Arcane.

L'ensemble des classes associées aux matériaux et milieux se trouvent
dans le namespace Materials de Arcane (voir \ref ArcaneMaterials). Le plus simple pour utiliser
ces classes est d'utiliser le mot clé \a using. Par exemple :

```cpp
#include <arcane/materials/IMeshMaterial.h>

using namespace Arcane::Materials;
```


L'ensemble des milieux et des matériaux est gérée par la classe
\arcanemat{IMeshMaterialMng}. Une instance de cette classe est associée à un
maillage \arcane{IMesh}. Il est possible d'avoir plusieurs instances de
\arcanemat{IMeshMaterialMng} par \arcane{IMesh} mais cela n'est pas utilisé pour l'instant.

\warning Pour l'instant, la gestion des matériaux et milieux est
incompatible avec les modifications de la topologie de maillage. En
particulier, cela inclue le repartionnement dû à l'équilibrage de
charge.

\note L'utilisation des constituants est compatible avec les
changements de la topologie du maillage. Néanmoins, pour que ce
support soit actif, il faut appeler la méthode
\arcanemat{IMeshMaterialMng::setMeshModificationNotified(true)} avant
de créer les milieux et matériaux.

Une instance de \arcanemat{IMeshMaterialMng} décrit un ensemble de milieux, chaque
milieu étant composé d'un ou plusieurs matériaux.
La liste des milieux et des matériaux doit aussi être créé lors de
l'initialisation du calcul et ne plus évoluer par la suite. Il est
aussi possible d'avoir la notion de bloc au-dessus de la motion de milieu, un bloc
comprenant un ou plusieurs milieux. Cette notion de bloc est
optionnel et n'est pas indispensable à l'utilisation des milieux ou
des matériaux

Dans l'implémentation actuelle, les milieux et les matériaux ne sont
associées qu'aux mailles et pas aux autres entités du maillage. La
liste des mailles d'un milieu ou d'un matériau est dynamique et peut
évoluer au cours du calcul. De même, il est possible d'avoir
plusieurs milieux et plusieurs matériaux par maille.

\note Il ne faut pas détruire manuellement les instances de
\arcanemat{IMeshMaterialMng}. Elles seront automatiquement détruites lorsque le maillage
associé sera supprimé.

L'instance par défaut de \arcanemat{IMeshMaterialMng} pour un maillage ne se
créée pas directement. Pour la récupérer, il faut utiliser la
méthode \arcanemat{IMeshMaterialMng::getReference()} avec comme argument le maillage
concerné. L'appel à cette fonction provoque la création du
gestionnaire si ce n'est pas déjà fait. Cette fonction à un coût CPU
non négligeable et il est donc préférable de stocker l'instance retournée
plutôt que d'appeler la fonction plusieurs fois.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate

## Création des matériaux, des milieux et des blocs {#arcanedoc_materials_manage_create}

Une fois le gestionnaire créé, il faut enregistrer les matériaux et
les milieux. La première chose à faire est d'enregistrer les
caractéristiques des matériaux. Pour l'instant, il s'agit uniquement
du nom du matériau. On enregistre uniquement les caractéristiques des
matériaux et pas les matériaux eux-mêmes car ces derniers sont créés
lors de la création des milieux.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate2

Une fois les caractéristiques enregistrées, il est possible de créer
les milieux, en spécifiant leur nom et la liste de leurs matériaux.
Il faut noter que deux milieux (ou plus) peuvent être constitués du même
matériau. Dans ce cas, %Arcane créé deux instances différentes du même
matériau et ces dernières sont indépendantes. Ainsi, dans l'exemple
suivant, le matériau MAT1 est présent dans ENV1 et ENV3, ce qui fait
deux matériaux distincts avec chacun leurs valeurs partielles. Une
fois les milieux créés, il est possible de les associés dans des
blocs. Un bloc est défini par un nom, le groupe de maille associé et
sa liste des milieux, qui est donc fixe.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate3

Une fois tous les milieux et blocs créés, il faut appeler
\arcanemat{IMeshMaterialMng::endCreate()}
pour signaler au gestionnaire que l'initialisation est terminée et qu'il
peut allouer les variables. Une fois cette méthode appelée, il ne faut
plus créer de blocs, ni de milieux ni de matériaux.

Il est possible d'ajouter des milieux à un bloc entre sa création
(\arcanemat{IMeshMaterialMng::createBlock()}) et l'appel à
\arcanemat{IMeshMaterialMng::endCreate()}, via la méthode
\arcanemat{IMeshMaterialMng::addEnvironmentToBlock()}.

Il est possible de supprimer des milieux à un bloc entre sa création
(\arcanemat{IMeshMaterialMng::createBlock()}) et l'appel à
\arcanemat{IMeshMaterialMng::endCreate()}, via la méthode
\arcanemat{IMeshMaterialMng::removeEnvironmentToBlock()}.

Les instances de \arcanemat{IMeshMaterial},
\arcanemat{IMeshEnvironment} et \arcanemat{IMeshBlock}
restent valident durant toute l'existence du \arcanemat{IMeshMaterialMng}
correspondant. Elles peuvent donc être conservées et il est possible
d'utiliser l'opérateur d'égalité pour savoir si deux instances
correspondent au même matériau.

Les informations concernant les
blocs, les matériaux et les milieux sont sauvegardées et
peuvent être relues en reprise. Néanmoins, cela n'est pas
automatique pour des raisons de compatibilité. Si on souhaite
relire les informations sauvegardées, il faut appeler
\arcanemat{IMeshMaterialMng::recreateFromDump()} et ne plus créér manuellement
les informations ni appeler la méthode \arcanemat{IMeshMaterialMng::endCreate()}.

Chaque matériau \arcanemat{IMeshMaterial}, milieu
\arcanemat{IMeshEnvironment} et bloc \arcanemat{IMeshBlock} possède un
identifiant unique, de type \arcane{Int32}, qui est accessible via les
méthodes \arcanemat{IMeshMaterial::id()}, \arcanemat{IMeshEnvironment::id()} ou
\arcanemat{IMeshBlock::id()}. Ces identifiants commencent à 0 et sont incrémentés
pour chaque matériau, milieu ou bloc. Par exemple, avec la
construction de l'exemple précédent, on a :

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreate4

## Ajout ou suppression de mailles d'un matériau {#arcanedoc_materials_manage_addremovecells}

Une fois les matériaux et milieux créés, il est possible d'ajouter ou
de supprimer des mailles pour un matériau. Il n'est pas nécessaire de
modifier les mailles par milieu : Arcane se charge de recalculer
automatiquement la liste des mailles d'un milieu en fonction de
celles de ses matériaux.

Toute modification se fait via la classe \arcanemat{MeshMaterialModifier}.
Il suffit de créer une instance de cette classe et d'appeler autant
de fois que nécessaire les méthodes \arcanemat{MeshMaterialModifier::addCells()} ou
\arcanemat{MeshMaterialModifier::removeCells()}. Il faut noter que ceux deux
méthodes permettent uniquement d'indiquer les mailles à ajouter ou
supprimer. La modification effective n'a lieu que lors de l'appel à
\arcanemat{MeshMaterialModifier::endUpdate()} ou de la destruction l'instance de
\arcanemat{MeshMaterialModifier}. C'est seulement à ce moment là que les valeurs
partielles sont mises à jour et qu'il est possible d'y accéder.

\note Pour l'instant, les variables partielles des matériaux sont automatiquement
initialisées à 0 dans les nouvelles mailles d'un matériau. Pour des
raisons de performance, il est possible que cela ne soit plus le cas
et alors les valeurs ne seront pas initialisées.

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialAddMat

## Itération sur les mailles matériaux {#arcanedoc_materials_manage_iteration}

Il existe trois classes pour faire référence aux notions de maille
matériaux :
- \arcanemat{AllEnvCell} est une classe permettant d'accéder à l'ensemble des
milieux d'une maille.
- \arcanemat{EnvCell} correspond à un milieu d'une maille et permet d'accéder aux
valeurs de ce milieu pour cette maille et à l'ensemble des valeurs
de cette maille pour les matériaux de ce milieu.
- \arcanemat{MatCell} correspondant à une valeur d'un matériau d'un milieu d'une maille.

Il existe une quatrième classe \arcanemat{ComponentCell} qui n'est pas une maille
spécifique mais qui peut représenter un des trois types ci-dessus et
permet d'unifier les traitements (voir \ref arcanedoc_materials_manage_component).

Il existe deux manières d'itérer sur les mailles matériaux.

La première est d'itérer sur tous les milieux, pour chaque milieu
d'itérer sur les matériaux de ce milieu et pour chacun de ces
matériaux d'itérer sur ces mailles. Par exemple :

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialIterEnv

Il est aussi possible d'utiliser un bloc pour itérer uniquement sur
les milieux de ce bloc au lieu d'itérer sur tous les milieux :

\snippet MeshMaterialTesterModule_Samples.cc SampleBlockEnvironmentIter

La seconde manière est de parcourir toutes les mailles d'un groupe de
maille, puis pour chaque maille d'itérer sur ses milieux et sur les matériaux de ses
milieux. Pour cela, on peut utiliser la macro ENUMERATE_ALLENVCELL,
de la même manière que la macro ENUMERATE_CELL, mais en spécifiant
en plus le gestionnaire de matériaux. Par exemple, si on souhaite
itérer sur toutes les mailles (groupe allCells())

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialIterCell

De la même manière, en itérant sur toutes les mailles d'un bloc :

\snippet MeshMaterialTesterModule_Samples.cc SampleBlockMaterialIterCell

Il existe une troisième manière pour itérer sur les mailles d'un
matériau ou d'un milieu.
Les classes \arcanemat{MatCellVector} et \arcanemat{EnvCellVector} permettent d'obtenir une
liste de \arcanemat{MatCell} ou de \arcanemat{EnvCell} à partir d'un groupe de mailles
et d'un matériau ou d'un mileu. L'exemple suivant montre comment récupérer la
liste des mailles du matériau \a mat et du milieu \env correspondant au groupe \a cells
pour positionner une variable \a mat_density :

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialIterFromGroup

\note Actuellement, les classes \arcanemat{MatCellVector} et \arcanemat{EnvCellVector} ne
sont pas copiables et ne sont valides que tant que le matériau (pour
\arcanemat{MatCellVector}) ou le milieu (pour \arcanemat{EnvCellVector}) et le groupe de
maille associé ne change pas.
De plus, la conservation des informations consomme de la mémoire et
la création à un cout en temps de calcul proportionnel au nombre
de mailles du groupe. 

## Conversion Cell vers AllEnvCell, MatCell ou EnvCell {#arcanedoc_materials_manage_conversion}

La plupart des méthodes sur les entités retournent des objets de type
\arcane{Cell}. Pour convertir ces objets en le type \a \arcanemat{AllEnvCell} afin
d'avoir les infos sur les matériaux, il faut passer par une instance
de \arcanemat{CellToAllEnvCellConverter}. Ce convertisseur peut par exemple être
créé en début de fonction car son coût de création est négligeable.

```cpp
Arcane::Materials::CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);
Arcane::Cell cell = ...;
Arcane::Materials::AllEnvCell allenvcell = all_env_cell_converter[cell];
```

À partir d'une \arcanemat{AllEnvCell}, il est possible de récupérer directement une
maille pour un matériau ou un milieu donné via
\arcanemat{IMeshMaterial::findMatCell()} ou \arcanemat{IMeshEnvironment::findEnvCell()}. Ces
méthodes retournent respectivement une \arcanemat{MatCell} ou une \arcanemat{EnvCell}
correspondant à la maille matériau ou milieu souhaité. L'instance
retournée peut-être nulle si le matériau ou le milieu n'est pas
présent dans la maille. Par exemple :

```cpp
Arcane::Materials::AllEnvCell allenvcell = ...;
Arcane::Materials::IMeshMaterial* mat = ...;
Arcane::Materials::MatCell matcell = mat->findMatCell(allenvcell);
if (matcell.null())
  // Materiau absent de la maille
  ...
Arcane::Materials::IMeshEnvironment* env = ...;
Arcane::Materials::EnvCell envcell = env->findEnvCell(allenvcell);
if (envcell.null())
  // Milieu absent de la maille.
  ...
```

## Unification {#arcanedoc_materials_manage_component}

Il est possible de traiter les mailles matériaux et milieu de manière
générique. La classe \arcanemat{ComponentCell} permet cette
unification et peut s'utiliser pour tous les types de mailles
\arcanemat{MatCell}, \arcanemat{EnvCell} ou \arcanemat{AllEnvCell}. La
macro ENUMERATE_COMPONENTCELL() permet d'itérer sur des mailles de ce
type :

\snippet MeshMaterialTesterModule_Samples.cc SampleComponentIter

La classe \arcanemat{ComponentCell} peut aussi servir à indexer une variable comme
le ferait une \arcanemat{MatCell} ou une \arcanemat{EnvCell}.

De la même manière, l'interface \arcanemat{IMeshComponent} est maintenant
l'interface de base de \arcanemat{IMeshMaterial} et
\arcanemat{IMeshEnvironment} et les méthodes
\arcanemat{IMeshMaterialMng::materialsAsComponents()} et
\arcanemat{IMeshMaterialMng::environmentsAsComponents()} permettent de traiter les
listes de matériaux et milieux de la même manière sous forme de liste
de \arcanemat{IMeshComponent}. Enfin, la classe \arcanemat{ComponentCellVector} permet de
créer un vecteur de \arcanemat{ComponentCell} et s'utiliser comme un \arcanemat{EnvCellVector}
ou \arcanemat{MatCellVector}.

La méthode \arcanemat{ComponentCell::superCell()} retourne la \arcanemat{ComponentCell} de
niveau hiérarchique immédiatement supérieur. Il est aussi possible
d'itérer sur les sous-mailles d'une \arcanemat{ComponantCell} via la macro
ENUMERATE_CELL_COMPONENTCELL():

\snippet MeshMaterialTesterModule_Samples.cc SampleComponentSuperItem

Enfin, il existe une macro ENUMERATE_COMPONENT() qui permet d'itérer
sur une liste de composants, et qui peut donc se substituer à
ENUMERATE_MAT() ou ENUMERATE_ENV().

## Variables matériaux {#arcanedoc_materials_manage_variable}

Il est aussi possible de déclarer des
variables uniquement sur les milieux et qui n'ont pas de valeurs sur
les matériaux. Pour plus d'infos voir la section \ref
arcanedoc_execution_env_variables.

Les variables matériaux sont similaires aux variables de maillage
mais possèdent en plus de la valeur aux mailles
classiques une valeur par matériau et par milieu présent dans la
maille. Pour une maille qui a 3 milieux, avec 2 matériaux pour le
milieu 1, 3 pour le milieu 2 et 5 pour le milieu 3, le nombre de
valeurs est donc de 14 (10 pour les matériaux, 3 pour les
milieux et 1 pour la valeur globale). Les valeurs par matériaux et
par milieux sont appelées des valeurs partielles.

Actuellement, les variables matériaux sont uniquement disponibles aux
mailles. La classe de base gérant ces variables est
MeshMaterialVariableRef. Les types possibles sont les suivants et
sont définis dans le fichier \c "MeshMaterialVariableRef.h".

<table>
<tr><th>Nom</th><th>Description</th></th>
<tr><td>\arcanemat{MaterialVariableCellByte}</td><td>variable matériau de type \arcane{Byte}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal}</td><td>variable matériau de type \arcane{Real}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellInt16}</td><td>variable matériau de type \arcane{Int16}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellInt32}</td><td>variable matériau de type \arcane{Int32}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellInt64}</td><td>variable matériau de type \arcane{Int64}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal2}</td><td>variable matériau de type \arcane{Real2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal3}</td><td>variable matériau de type \arcane{Real3}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal2x2}</td><td>variable matériau de type \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellReal3x3}</td><td>variable matériau de type \arcane{Real3x3}</td></tr>
</table>

Il est aussi possible de définir des variables tableaux sur les
matériaux, qui ont le type suivant:

<table>
<tr><th>Nom</th><th>Description</th></th>
<tr><td>\arcanemat{MaterialVariableCellArrayByte}</td><td>variable matériau de type tableau de \arcane{Byte}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal}</td><td>variable matériau de type tableau de \arcane{Real}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayInt16}</td><td>variable matériau de type tableau de \arcane{Int16}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayInt32}</td><td>variable matériau de type tableau de \arcane{Int32}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayInt64}</td><td>variable matériau de type tableau de \arcane{Int64}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal2}</td><td>variable matériau de type tableau de \arcane{Real2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal3}</td><td>variable matériau de type tableau de \arcane{Real3}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal2x2}</td><td>variable matériau de type tableau de \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{MaterialVariableCellArrayReal3x3}</td><td>variable matériau de type tableau de \arcane{Real3x3}</td></tr>
</table>

\note Pour information, en interne
d'Arcane, ces valeurs partielles sont gérées sous forme de variable
tableau classique mais cette implémentation peut évoluer. Depuis la
version 2.0 de %Arcane, pour pouvoir distinguer ces variables des
variables classiques, elles sont taggées avec la valeur
"Material". Par exemple, la commande
```cpp
Arcane::IVariable* var = ...;
if (var->hasTag("Material")){
  // Il s'agit d'une valeur partielle.
}
```

Pour l'instant, il n'est possible de créer que des variables
scalaires aux mailles, avec l'un des types de donnée suivante : #Real, #Int32, #Int64,
Real2, Real3, Real2x2 ou Real3x3. Le nom de la classe correspondante
est le même que pour les variables classiques, mais préfixé de
'Material'. Par exemple, pour une variable de type Real3, le nom est
\a MaterialVariableCellReal3.

### Construction {#arcanedoc_materials_manage_variable_build}

La déclaration des variables matériaux se fait d'une manière
similaire à celle des variables du maillage. Il est aussi possible de
les déclarer dans le fichier axl, de la même manière qu'une variable
classique, en ajoutant l'attribut \c material avec la valeur \c
true. Les valeurs valides pour \c dimension sont \c 0 ou \c 1.

```xml
<variable field-name="mat_density"
         name="Density"
         data-type="real"
         item-kind="cell"
         dim="0"
         material="true"
/>
```

La construction se fait avec un objet du type MaterialVariableBuildInfo qui référence le IMeshMaterialMng
correspondant ou alors de la même manière qu'une variable classique,
via le VariableBuildInfo. Par exemple :

\snippet MeshMaterialTesterModule_Samples.cc SampleMaterialCreateVariable

\note La construction des variables matériaux est thread-safe.

Comme pour les variables classiques, les instructions précédentes ne
créent une variable que si aucune de même nom n'existe. Dans le cas
contraire, elles récupèrent une référence à la variable déjà créé
correspondante.

### Utilisation {#arcanedoc_materials_manage_variable_usage}

Pour accéder à une valeur d'une variable matériau, il suffit
d'utiliser l'opérateur [] avec comme argument un des types
ComponentCell, Cell, MatCell, EnvCell ou AllEnvCell.

```cpp
Arcane::Cell global_cell;
Arcane::Materials::MatCell mat_cell;
Arcane::Materials::EnvCell env_cell;
mat_density[global_cell]; // Valeur globale
mat_density[mat_cell];    // Valeur pour un matériau
mat_density[env_cell];    // Valeur pour un milieu.
```

La valeur globale est partagée avec celle des variables standards
Arcane de même nom. Par exemple :

```cpp
Arcane::Materials::IMeshMaterialMng* material_mng = ...;
Arcane::Materials::MaterialVariableBuildInfo mat_info(material_mng,"Density"))
Arcane::Materials::MaterialVariableCellReal mat_density(mat_info);
Arcane::VariableBuildInfo info(defaultMesh(),"Density"))
Arcane::VariableCellReal density(info);

mat_density[global_cell] = 3.0;
info() << density[global_cell]; // Affiche 3.0
```

Il est aussi possible de récupérer la variable globale associée à une
variable matériaux, via la méthode globalVariable():

```cpp
Arcane::Materials::IMeshMaterialMng* material_mng = ...;
Arcane::Materials::MaterialVariableCellReal mat_density(Arcane::Materials::MaterialVariableBuildInfo(material_mng,"Density"));
Arcane::VariableCellReal& density(mat_density.globalVariable());
```

L'implémentation des variables matériaux a pour but de limiter
l'utilisation mémoire. Dans cette optique, les valeurs aux milieux et
aux matériaux peuvent utiliser la même zone mémoire et dans ce cas
modifier la valeur milieu modifie aussi la valeur matériau et
réciproquement. Cela est le cas si les conditions suivantes sont
respectées dans une maille \a cell :
- un seul matériau (MAT) dans un milieu (ENV), alors var[MAT]==var[ENV]
- un seul milieu (ENV) dans la maille, alors var[ENV] == var[cell]

### Synchronisations {#arcanedoc_materials_manage_variable_synchronize}

Il est possible de synchroniser les valeurs des variables matériaux,
de la même manière que les variables classiques via la fonction
\arcanemat{MeshMaterialVariableRef::synchronize()}.

\warning Attention, il faut tout de même que
les informations sur les matériaux présents soient cohérentes entre
tous les sous-domaines : si une maille existe dans plusieurs
sous-domaines, elle doit avoir les mêmes matériaux et milieux dans
chaque sous-domaine.

Il est possible de garantir que toutes les informations sur les
matériaux et les milieux sont cohérentes entre les sous-domaines en
appelant la méthode
\arcanemat{IMeshMaterialMng::synchronizeMaterialsInCells()}. Il est aussi possible
de vérifier cette cohérence en appelant
\arcanemat{IMeshMaterialMng::checkMaterialsInCells()}.

Depuis la version 2.3.7, il existe plusieurs implémentations pour la
synchronisation. La version par défaut n'est pas optimisée et
effectue une synchronisation par matériau enregistré dans
\arcanemat{IMeshMaterialMng}. La version optimisée conseillée est la version 6.
Pour l'utiliser, il faut appeler la méthode
\arcanemat{IMeshMaterialMng::setSynchronizeVariableVersion()} avec la valeur 6. Il est aussi
possible d'effectuer plusieurs synchronisations en une seule fois via la
classe \arcanemat{MeshMaterialVariableSynchronizerList}. Cela permet d'optimiser
les communications en réduisant le nombre de messages entre les
processeurs.
Par exemple :
```cpp
Arcane::Materials::IMeshMaterialMng* material_mng = ...;
Arcane::Materials::MaterialVariableCellReal temperature = ...;
Arcane::Materials::MaterialVariableCellInt32 mat_index = ...;
Arcane::Materials::MaterialVariableCellReal pressure = ...;
// Création de la liste de variables à synchroniser.
Arcane::Materials::MeshMaterialVariableSynchronizerList mmvsl(material_mng);

// Ajoute 3 variables à la liste.
temperatture.synchronize(mmvsl);
mat_index.synchronize(mmvsl);
pressure.synchronize(mmvsl);

// Exécute la synchronisation sur les 3 variables en une fois.
mmvsl.apply();
```

Depuis la version 3.6, il existe deux autres versions de la
synchronisation qui fonctionne de manière identique à la version 6
mais avec les modifications suivantes :
- La version 7 n'effectue qu'une seule allocation mémoire pour les
buffers d'envoi et de réception (alors que la version 6 effectue
autant d'allocation qu'il y a de sous-domaines voisins)
- La version 8 qui est identique à la version 7 au détail près que
les buffers alloués sont conservés entre deux synchronisations. Cette
version permet d'éviter des allocations/désallocations successives au
prix d'une augmentation mémoire.
Ces deux versions ont pour but d'éviter de faire trop de cycles
d'allocations/désallocations qui peuvent demander une charge de
travail importante pour les cartes réseaux pour accès directs à la
mémoire (RMA).

### Gestion des dépendances {#arcanedoc_materials_manage_variable_dependencies}

Il est possible d'utiliser le mécanisme de dépendance sur les variables matériaux.
Ce mécanisme est similaire à celui des variables classiques mais
permet de gérer les dépendances par matériaux. 

\note Contrairement aux dépendances sur les variables classiques,
les dépendances sur les matériaux ne gèrent pas le temps physique et
il n'est pas possible par exemple de faire des dépendances sur le
temps physique précédent (en spécifiant \arcane{IVariable::DPT_PreviousTime}
par exemple).

Le fonctionnement est le suivant :

\snippet MeshMaterialTesterModule_Samples.cc SampleDependencies

Lors de l'appel à \arcanemat{MeshMaterialVariableRef::update()}, il est
nécessaire de spécifier en argument le matériau sur lequel on
souhaite faire la mise à jour. La méthode de calcul doit donc avoir
comme argument un matériau, et en fin de calcul appeler
\arcanemat{MeshMaterialVariableRef::setUpToDate(mat)} avec \a mat le matériau
recalculé.
Par exemple :

\snippet MeshMaterialTesterModule_Samples.cc SampleDependenciesComputeFunction

## Variables milieux {#arcanedoc_materials_manage_environment_variable}

Il est aussi possible de définir des variables qui n'ont de valeurs
partielles que sur les milieux et pas sur les matériaux. Mise à part
cette différence, elles se comportent comme les variables
matériaux. Les variables milieux disponibles sont les suivantes:

<table>
<tr><th>Nom</th><th>Description</th></th>
<tr><td>\arcanemat{EnvironmentVariableCellByte}</td><td>variable milieu de type \arcane{Byte}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal}</td><td>variable milieu de type \arcane{Real}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellInt16}</td><td>variable milieu de type \arcane{Int16}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellInt32}</td><td>variable milieu de type \arcane{Int32}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellInt64}</td><td>variable milieu de type \arcane{Int64}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal2}</td><td>variable milieu de type \arcane{Real2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal3}</td><td>variable milieu de type \arcane{Real3}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal2x2}</td><td>variable milieu de type \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellReal3x3}</td><td>variable milieu de type \arcane{Real3x3}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayByte}</td><td>variable milieu de type tableau de \arcane{Byte}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal}</td><td>variable milieu de type tableau de \arcane{Real}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayInt16}</td><td>variable milieu de type tableau de \arcane{Int16}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayInt32}</td><td>variable milieu de type tableau de \arcane{Int32}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayInt64}</td><td>variable milieu de type tableau de \arcane{Int64}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal2}</td><td>variable milieu de type tableau de \arcane{Real2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal3}</td><td>variable milieu de type tableau de \arcane{Real3}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal2x2}</td><td>variable milieu de type tableau de \arcane{Real2x2}</td></tr>
<tr><td>\arcanemat{EnvironmentVariableCellArrayReal3x3}</td><td>variable milieu de type tableau de \arcane{Real3x3}</td></tr>
</table>

Dans le fichier axl, ces variables peuvent être définies en
spécifiant l'attribut \c environment à \c true.

```xml
<variable field-name="mat_density"
         name="Density"
         data-type="real"
         item-kind="cell"
         dim="0"
         environment="true"
/>
```

\warning Comme les structures internes des mailles matériaux et
milieux sont unifiées, il est possible à la compilation d'indexer une variable milieu
avec une \arcanemat{MatCell}. Comme il n'y a pas de valeurs matériaux associées,
cela provoquera un accès mémoire invalide qui a de fortes chance de
se solder par une erreur de segmentation (SegmentationFault). Ces
erreurs sont détectées en mode CHECK et il est donc préférable
d'utiliser ce mode pour les développements.

## Parallélisation des boucles sur les matériaux et milieux {#arcanedoc_materials_manage_concurrency}

De la même manière que les boucles sur les entités (voir \ref
arcanedoc_parallel_concurrency), il est possible d'exécuter en parallèle des
boucles sur les milieux où les matériaux. Cela se fait de manière
similaire aux boucles sur les entités, en utilisant la méthode
\arcane{Parallel::Foreach}. Par exemple :

\snippet MeshMaterialTesterModule_Samples.cc SampleConcurrency

## Optimisation des modifications sur les matériaux et les milieux. {#arcanedoc_materials_manage_optimization}

La modification des mailles matériaux et milieux se fait via la
classe \arcanemat{MeshMaterialModifier}. Cette modification se fait via des
appels successifs à \arcanemat{MeshMaterialModifier::addCells()} ou
\arcanemat{MeshMaterialModifier::removeCells()}. Ces méthodes permettent
d'enregistrer la liste des modifications mais ces dernières ne sont
réellement exécutées que lors de la destruction de l'instance
\arcanemat{MeshMaterialModifier}.

Par défaut, le comportement est le suivant :
- sauvegarde des valeurs de toutes les variables matériaux
- application des modifications (qui consiste uniquement à modifier
la liste des entités des groupes de mailles associées aux matériaux
et milieux)
- restauration des valeurs de toutes les variables matériaux. Lors de
la restauration, si une nouvelle maille matériau est créée, sa valeur
dépend de \arcanemat{IMeshMaterialMng::isDataInitialisationWithZero()}.

La sauvegarde et la restauration des valeurs sont des opérations
couteuses en temps CPU et mémoire. Il est possible de désactiver ces
opérations via setKeepValuesAfterChange() mais bien entendu dans ce
cas les valeurs partielles ne sont pas conservées.

Afin d'optimiser ces modifications de matériaux, il est possible de
se passer de ces opérations de sauvegarde/restauration. Pour cela, il
faut utiliser la méthode IMeshMaterialMng::setModificationFlags(int
v). Cette méthode doit être appelée avant
\arcanemat{IMeshMaterialMng::endCreate()}.
L'argument utilisé est une combinaison de bits de l'énumération
\arcanemat{eModificationFlags}:
- \arcanemat{eModificationFlags::GenericOptimize} : indique qu'on souhaite
activer les optimisations.
- \arcanemat{eModificationFlags::OptimizeMultiAddRemove}: indique qu'on active
les optimisations dans le cas où il y a plusieurs ajouts ou
suppressions avec un même MeshMaterialModifier.
- \arcanemat{eModificationFlags::OptimizeMultiMaterialPerEnvironment} indique qu'on active
les optimisations dans le cas où il y a plusieurs matériaux
dans le milieu.

\warning La valeur \arcanemat{eModificationFlags::OptimizeMultiMaterialPerEnvironment}
n'est disponible qu'à partir de la version 2.3.2 de %Arcane. Sur
les versions antérieures, aucune optimisation n'est effectuée
si un des matériaux modifié n'est pas le seul matériau du
milieu.

Par exemple, on suppose les trois séries de modifications :
```cpp
{
  Arcane::Materials::MeshMaterialModifier m1(m_material_mng);
  m1.addCells(mat1,ids);
}
{
  Arcane::Materials::MeshMaterialModifier m2(m_material_mng);
  m2.addCells(mat1,ids1);
  m2.addCells(mat2,ids2);
  m2.removeCells(mat1,ids3);
}
{
  Arcane::Materials::MeshMaterialModifier m3(m_material_mng);
  m3.removeCells(mat2,ids);
}
```

Suivant les valeurs spécifiées lors de l'init, on aura :

```cpp
int flags1 = (int)Arcane::Materials::eModificationFlags::GenericOptimize;
m_material_mng->setModificationFlags(flags1);
// Seuls m1 et m3 sont optimisés.

int flags2 = (int)Arcane::Materials::eModificationFlags::GenericOptimize | (int)Arcane::Materials::eModificationFlags::OptimizeMultiAddRemove;
m_material_mng->setModificationFlags(flags2);
// m1, m2 et m3 sont optimisés.
```

Il est possible de surcharger les optimisations utilisées via la
variable d'environnement ARCANE_MATERIAL_MODIFICATION_FLAGS. Cette
variable doit contenir une valeur entière correspondant à celle
utilisée en argument de \arcanemat{IMeshMaterialMng::setModificationFlags()} (à
savoir 1 pour l'optimisation générale, 3 pour en plus optimiser les
ajouts/suppressions multiples, et 7 pour optimiser aussi les cas des
milieux multi-matériaux).

### Notes sur l'implémentation {#arcanedoc_materials_manage_optimization_implementation}

\htmlonly
<span style='background: red'>IMPORTANT</span>
\endhtmlonly

<h4>NOTE 1</h4>

Actuellement, les méthodes optimisées ne réutilisent pas les valeurs
partielles lorsque des mailles sont supprimées puis ajoutées à un même matériau
ce qui conduit à une augmentation progressive de la mémoire utilisée
par les valeurs partielles. Il est néanmoins possible de libérer
cette mémoire supplémentaire via l'appel à
IMeshMaterialMng::forceRecompute().

<h4>NOTE 2</h4>

Le comportement en mode optimisé lorsqu'il y a suppression puis ajout
d'une même maille dans un même matériau est différent du mode
classique. Par exemple :

```cpp
MeshMaterialModifier m1(m_material_mng);
Array<Int32> add_ids;
Array<Int32> remove_ids;

remove_ids.add(5);
remove_ids.add(9);
add_ids.add(5);
add_ids.add(7);

m1.removeCells(mat1,remove_ids);
m1.addCells(mat1,add_ids);
```

La maille de localId() \a 5 est d'abord supprimée puis remise dans le
matériau. En mode classique, la valeur de la maille sera la même
qu'avant la modification car on restaure la valeur depuis la
sauvegarde. En mode optimisé, avec
eModificationFlags::OptimizeMultiAddRemove spécifié, on supprime
d'abord la maille du matériau puis on l'a recréé. Sa valeur sera donc
celle d'une nouvelle maille matériau crée, donc 0 si
IMeshMaterialMng::isDataInitialisationWithZero() ou la valeur de la
maille globale associée sinon.

<h4>NOTE 3</h4>

Enfin, les méthodes optimisées sont plus strictes que les méthodes
classiques et certaines opérations qui étaient tolérées en mode
classique ne le sont plus :
- spécifier dans la liste des mailles à ajouter une maille qui est
déjà dans le matériau.
- spécifier dans la liste des mailles à supprimer une maille qui
n'est pas dans le matériau.
- spécifier plusieurs fois la même maille dans la liste des mailles
à supprimer ou ajouter.

```cpp
MeshMaterialModifier mm(m_material_mng);
Int32Array ids;
ids.add(5);
mm.addCells(mat1,ids); // Erreur si la maille 5 est déjà dans mat1
mm.removeCells(mat1,ids); // Erreur si la maille 5 n'est pas mat1
ids.add(6);
ids.add(6); // Erreur si \a ids contient plusieurs fois la même maille.
```

Si on se trouve dans un des cas précédent, il y a de fortes chances
que cela fasse planter le code. Pour éviter cela, le mode CHECK
détecte ces erreurs, les signale dans le listing et les filtre pour
que cela fonctionne correctement. Ces erreurs sont signalées dans le
listing par des messages du style suivant :

\verbatim
ERROR: item ... is present several times in add/remove list for material mat=...
ERROR: item ... is already in material mat=...
ERROR: item ... is not in material mat=...
\endverbatim

En mode release, la détection et la correction ne se fait que si
la variable d'environnement ARCANE_CHECK est positionnée et vaut 1.
*/


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_materials
</span>
<span class="next_section_button">
\ref arcanedoc_materials_loop
</span>
</div>
