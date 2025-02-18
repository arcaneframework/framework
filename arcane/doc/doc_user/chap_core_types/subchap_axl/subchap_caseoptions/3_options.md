# Les types d'options {#arcanedoc_core_types_axl_caseoptions_options}

[TOC]

## Les options simples {#arcanedoc_core_types_axl_caseoptions_options_simple}

Elles sont décrites par l'élément <tt>simple</tt> :

```xml
<simple name="simple-real" type="real">
  <description>Réel simple</description>
</simple>
```

L'attribut `type` peut prendre une des valeurs suivantes :
- `real` pour les réels. Ils correspondent au type Arccore::Real.
- `integer` pour les entiers. Ils correspondent au type Arccore::Integer.
- `int32` pour les entiers. Ils correspondent au type Arccore::Int32.
- `int64` pour les entiers. Ils correspondent au type Arccore::Int64.
- `bool` pour les booléens. Ils correspondent au type `bool` du C++.
- `string` pour les chaînes de caractères. Ils correspondent à la classe Arccore::String.

Pour l'exemple précédent, le jeu de données contient, par exemple, la ligne :

```xml
<simple-real>3.4</simple-real>
```

## Les options énumérées {#arcanedoc_core_types_axl_caseoptions_options_enum}

Elles sont décrites par l'élément <tt>enumeration</tt> :

```xml
<enumeration name="boundary-condition" type="eBoundaryCondition" default="X">
  <description>Type de condition aux limites</description>
  <enumvalue name="X" genvalue="VelocityX" />
  <enumvalue name="Y" genvalue="VelocityY"  />
  <enumvalue name="Z" genvalue="VelocityZ"  />
</enumeration>
```

L'attribut \c type doit indiquer le nom de
l'énumération C++ correspondante (ici \c eBoundaryCondition). Si
l'énumération appartient à un espace de nom C++ (\c namespace), il
est possible de le spécifier directement (par exemple \c MonCode::eBoundaryCondition).
	
Ce type d'option comporte une liste d'éléments fils
<tt>enumvalue</tt> qui référence chaque valeur
de l'énumération. Cet élément comporte deux attributs :
- `name` indique à quelle valeur de l'utilisateur va
  correspondre cette énumération. Ce sont ces valeurs que
  l'utilisateur va entrer.
- `genvalue` indique le nom de l'énumération correspondante
  dans le langage C++.

Dans l'exemple précédent, l'option correspond à l'énumération C++
suivante :

```cpp
enum eBoundaryCondition { VelocityX, VelocityY, VelocityZ };
```

La définition de l'énumération doit être effective avant d'inclure
le fichier *Test_axl.h*. Par exemple, si l'énumération
précédente est définie dans un fichier *BoundaryCondition.h*,
il faudra inclure les fichiers dans l'ordre suivant :
```cpp
#include "BoundaryCondition.h"
#include "Test_axl.h"
```
  
Si dans le jeu de données, on a la ligne :

```xml
<boundary-condition>X</boundary-condition>
```
 
alors l'option associée dans la classe C++ générée par le fichier aura
la valeur <tt>VelocityX</tt>.

## Les options étendues {#arcanedoc_core_types_axl_caseoptions_options_extended}

L'interface d'utilisation de ce type d'option est en cours de définition.
Pour l'instant, ce type d'option ne doit être utilisé qu'avec les 
groupes d'entités de maillage définis par %Arcane.

  Les options de type étendues sont décrites par l'élément <tt>extended</tt> :

```xml
<extended name="surface" type="Arcane::FaceGroup">
   <description>
     Surface sur laquelle s'applique la condition aux limites
   </description>
</extended>
```

L'attribut <b>type</b> peut prendre une des valeurs suivantes :
  
- *Arcane::NodeGroup* pour un groupe de noeuds
- *Arcane::FaceGroup* pour un groupe de faces
- *Arcane::CellGroup* pour un groupe de mailles

Si une option de ce type existe, sa valeur est validée après lecture
du maillage et doit correspondre à un groupe existant et du bon type.

Pour le développeur, une option de type \c extended génère un objet du type
<tt>Arcane::NodeGroup</tt>, <tt>Arcane::FaceGroup</tt> 
ou <tt>Arcane::CellGroup</tt> qui s'utilise donc comme un groupe. Par exemple,
avec l'option précédente de nom <tt>surface</tt>, la méthode
<tt>surface()</tt> de la classe gérant le jeu de donnée retourne un
groupe de faces de type <tt>Arcane::FaceGroup</tt> et la ligne suivante est valide :

```cpp
ENUMERATE_FACE(i,surface())
```

Pour l'exemple précédent, si XMIN est un surface définie sur le maillage,
le jeu de données contient, par exemple, la ligne :

```xml
<surface>XMIN</surface>
```

## Les options complexes {#arcanedoc_core_types_axl_caseoptions_options_complex}

Une option complexe est composée de plusieurs autres options, y compris
d'autres options complexes. L'option complexe est décrite par l'élément <tt>complex</tt>.

L'exemple suivant définit une option <tt>pair-of-int</tt> composée
de deux options simples <tt>first</tt> et <tt>second</tt> :

```xml
<complex name="pair-of-int" type="PairOfInt">
  <simple name="first" type="integer">
  <simple name="second" type="integer">
</complex>
```

Le jeu de données correspondant est :

```xml
<pair-of-int>
  <first>5</first>
  <second>6</second>
</pair-of-int>
```

## Les options de type service {#arcanedoc_core_types_axl_caseoptions_options_service}

La notion de service %Arcane est expliqué dans la section \ref arcanedoc_core_types_service.
Rappelons simplement qu'un service est un composant externe utilisé par un module
et donc nécessaire au fonctionnement du module. Pour que le module puisse
fonctionner, son jeu de données doit référencer le service nécessaire grâce à
l'élément <tt>service-instance</tt>.

L'exemple suivant définit un service nommé *eos-model* 
de type \c IEquationOfState :

```xml
<service-instance name="eos-model" type="IEquationOfState">
  <description>Service d'equation d'état</description>
</service-instance>
```

La classe \c IEquationOfState correspond à l'interface définie pour le service.
Si la classe appartient à un espace de nom C++ (\c namespace), il est
possible de le spécifier directement (par exemple \c
MonCode::IEquationOfState).

Le jeu de données correspondant est:

```xml
<eos-model name="StiffenedGas">
  <limit-tension>0.01</limit-tension>
</eos-model>
```

Il est aussi possible d'ajouter le nom du maillage dans le jeu de données :

```xml

<eos-model name="StiffenedGas" mesh-name="Mesh0">
  <limit-tension>0.01</limit-tension>
</eos-model>
```

Le code pour lire l'option est :
  
```cpp
IEquationOfState* eos = options()->eosModel();
```
  
Les services peuvent avoir les attributs suivants :
<table>
<tr>
<th>Attribut</th>
<th>Type</th>
<th>Description</th>
</tr>
<tr>
<td>allow-null</td>
<td>bool</td>
<td>si \c false (le défaut), le service spécifié dans le jeu de
données doit exister. Si ce n'est pas le cas, une erreur se produira
lors de la lecture. Si cet attribut vaut \c true, et que le service
spécifié n'existe pas, l'option correspondante sera nulle et il
faudra tester sa valeur. Par exemple :

```cpp
IEquationOfState* eos = options()->eosModel();
if (!eos)
  info() << "Service spécifié indisponible";
```
</td>
</tr>

<tr>
<td>optional</td>
<td>bool</td>
<td>si \c false (le défaut), le service est toujours créé et si
l'élément correspondant n'existe pas dans le jeu de données, le
service avec le nom par défaut sera créé. Si cet attribut vaut \c
true et que l'élément n'est pas présent dans le jeu de données,
aucun service n'est créé.
</td>
</tr>

<tr>
<td>mesh-name</td>
<td>string</td>
<td>Nom du maillage auquel le service sera associé. Lors de la
construction du service, la valeur de Arcane::ServiceBuildInfo::mesh()
sera le maillage de nom \a mesh-name. Si aucune valeur n'est spécifiée pour ce
champ, alors c'est le maillage du service parent qui sera
utilisé. S'il n'y a pas de service parent, alors c'est le maillage par
défaut (Arcane::ISubDomain::defaultMesh()) qui sera utilisé.

Si le maillage associé à ce service n'existe pas, alors l'option est
ignorée ainsi que les sous-options éventuelles.

Cette option est disponible à partir de la version 2.0.7 de %Axlstar
et 3.8.4 de %Arcane.

\warning Il est possible de spécifier l'attribut `mesh-name` dans l'AXL
ET/OU dans le jeu de données (.ARC). On peut voir celui de l'AXL comme
un `mesh-name` par défaut pour une instance de service donnée. Si
le `mesh-name` est spécifié dans le jeu de données (.ARC), il écrasera
le `mesh-name` spécifié dans l'AXL.
</td>
</tr>

</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_common_struct
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_usage
</span>
</div>
