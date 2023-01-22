# Variable {#arcanedoc_core_types_axl_variable}

[TOC]

Une variable est une valeur manipulée par le code et gérée par
%Arcane. Par exemple le volume, la vitesse, sont des variables. Elles
sont caractérisées par un **nom**, un **type de donnée**, un **support**
et une **dimension**.

Les variables sont déclarées à l'intérieur d'un module/service, au sein 
du descripteur de module/service (pour la suite de cette page, on va dire 
que module = service, pour alléger l'écriture).

Si deux modules utilisent des variables de même nom,
leurs valeurs seront implicitement partagées. C'est par ce moyen que les 
modules communiquent leurs informations.

## Caractéristiques

### Type de données {#arcanedoc_core_types_axl_variable_types}

Le **type de donnée** des variables est à choisir parmi les suivants :

| Nom C++            |  Type
|--------------------|-----------------------------------------
| \arccore{Integer}  | entier signé sur 32 bits
| \arccore{Int16}    | entier signé sur 16 bits
| \arccore{Int32}    | entier signé sur 32 bits
| \arccore{Int64}    | entier signé sur 64 bits
| \arcane{Byte}      | représente un caractère sur 8 bits
| \arccore{Real}     | réel IEEE 754
| \arcane{Real2}     | coordonnée 2D, vecteur de deux réels
| \arcane{Real3}     | coordonnée 3D, vecteur de trois réels
| \arcane{Real2x2}   | tenseur 2D, vecteur de quatre réels
| \arcane{Real3x3}   | tenseur 3D, vecteur de neufs réels
| \arccore{String}   | chaîne de caractères unicode

Les flottants (*Real*, *Real2*, *Real2x2*, *Real3*, *Real3x3*) sont des réels double précisions (stockés
sur 8 octets).

### Support {#arcanedoc_core_types_axl_variable_support}

Le **support** correspond à l'entité qui porte la variable,
sur laquelle la variable est définie. Ces variables qui s'appliquent sur des éléments du maillage
sont appelées des **grandeurs**.

| Nom C++           | Support
|-------------------|-----------------------------------------------------
| (vide)            | variable définie globalement (ex : pas de temps)
| \arcane{Node}     | noeud du maillage
| \arcane{Face}     | face du maillage
| \arcane{Cell}     | maille du maillage
| \arcane{Particle} | particule du maillage
| \arcane{DoF}      | degré de liberté

### Dimension {#arcanedoc_core_types_axl_variable_dim}

La **dimension** peut être :

| Nom C++     | Dimension
|-------------|------------
| **Scalar**  | scalaire
| **Array**   | tableau 1D
| **Array2**  | tableau 2D

Depuis la version 3.8.11 de %Arcane il est aussi possible de déclarer
des variables muti-dimensionnelles sur le maillage.

### Classe C++ des variables scalaires, 1D ou 2D {#arcanedoc_core_types_axl_variable_cppclass}

Pour les variables scalaires, 1D ou 2D, il est aisé d'obtenir la
classe C++ correspondant à un type de donnée, un support et à une dimension
donnée. Le nom de la classe est construit de la manière suivante :

**Variable** + \ref arcanedoc_core_types_axl_variable_support +
\ref arcanedoc_core_types_axl_variable_dim +
\ref arcanedoc_core_types_axl_variable_types

Par exemple, pour une variable représentant un tableau d'entiers
**VariableArrayInteger**, pour une variable représentant un 
réel **VariableScalarReal**.

Quand une variable scalaire est définie sur une entité du maillage
le support (*Scalar*) n'est pas précisé. Par exemple, pour une variable 
représentant un réel aux mailles **VariableCellReal**.

Toutes les combinaisons sont possibles aux exceptions suivantes :
- les variables de type *chaîne de caractères* qui n'existent que pour les genres
  scalaires et tableaux mais pas sur les éléments du maillage (pour
  des raisons de performances).
- Les variables de dimension 2 ne peuvent pas avoir de support (il
  n'est pas possible d'avoir des variables 2D aux mailles par exemple).
	
Le tableau suivant donne quelques exemples de variables :

Nom C++                           | Description
----------------------------------|------------
\arcane{VariableScalarReal}       | un réel
\arcane{VariableScalarInteger}    | un entier
\arcane{VariableArrayInteger}     | Tableau d'entiers
\arcane{VariableArrayReal3}       | Tableau de coordonnées 3D
\arcane{VariableNodeReal2}        | Coordonnée 2D aux noeuds
\arcane{VariableFaceReal}         | Réel aux faces
\arcane{VariableFaceReal3}        | Coordonnée 3D aux faces
\arcane{VariableFaceArrayInteger} | Tableau d'entiers aux faces
\arcane{VariableCellArrayReal}    | Tableau de réels aux mailles
\arcane{VariableCellArrayReal3}   | Tableau de coordonnées 3D aux mailles
\arcane{VariableCellArrayReal2x2} | Tableau de tenseurs 2D aux mailles

Pour les variables multi-dimensionnelles, il n'existe pas de `typedef`
car le nombre de possibilités est infini.

## Déclaration {#arcanedoc_core_types_axl_variable_declare}

La déclaration des variables se fait par l'intermédiaire du 
descripteur de module ou du service (fichier AXL) ou directement dans
le code.

### Déclaration dans les fichiers AXL

\note Les variables déclarées dans le fichier AXL sont toujours
associées au maillage par défaut du module ou du service. Il n'est pas
possible de déclarer de variables de sous-domaine dans ces fichiers
(mais il est toujours possible de les déclarer explicitement dans la
classe du module ou du service).

Dans le fichier AXL, la déclaration des variables se fait dans la
balise \c variables. L'exemple suivant montre la déclaration dans le
module \c Test d'une variable de type réel aux mailles appelée \c
Pressure et une variable de type réel aux noeuds appelée \c
NodePressure.

```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
    <description>Descripteur du module Test</description>
  <variables>
    <variable field-name="pressure" name="Pressure" data-type="real"
              item-kind="cell" dim="0" dump="true" need-sync="true" />
    <variable field-name="node_pressure" name="NodePressure" data-type="real"
              item-kind="node" dim="0" dump="true" need-sync="true" />
  </variables>

  <entry-points>
  </entry-points>

  <options>
  </options>
</module>
```

#### Nom, type de donnée et support de la variable

Les attributs suivants sont obligatoires et permettent de définir le
nom, le type de donnée et le support de la variable :

<table>
<tr>
<th>Attribut</th>
<th>Description</th>
</tr>
<tr>
<td> **name** </td>
<td>nom de la variable. Par convention, elle commence par une
majuscule. Les caractères valides sont les caractères alphabétiques
**[a-zA-Z]**, les chiffres (sauf au début du nom) et le caractère
souligné. Les variables dont le nom commence par *Global* et *%Arcane*
sont réservées pour %Arcane.
</td>
</tr>
<tr>
<td> **field-name** </td>
<td>nom informatique du champ de la variable dans la classe générée. Ce nom doit
être un nom valide en C++. Par convention, si la variable s'appelle
 *NodePressure* par exemple, le champ sera **node_pressure**. Dans la
 classe générée, le nom spécifié sera préfixé de **m_** (le préfixage \c m_ correspond aux 
  \ref arcanedoc_general_codingrules "règles de codage" dans %Arcane pour les attributs de classe).
</td>
</tr>
<tr>
<td> **item-kind** </td>
<td>support de la variable. Les valeurs possibles sont  \c node, \c
edge, \c face, \c cell, \c particle, \c dof ou \c none. La valeur \c none indique qu'il ne s'agit
pas d'une variable du maillage. Si le type est \c particle ou \c dof,
il faut spécifier l'attribut **family-name**
</td>
</tr>

<tr>
<td> **data-type** </td>
<td>type de données de la variable. Il s'agit au choix de *integer*, *int16*,
 *int32*, *int64*, *real*, *string*, *real2*, *real3*, *real2x2*,
 *real3x3*. Le type *string8 n'est pas utilisable pour les variables
 du maillage (celles ayant un support).
</td>
</tr>

<tr>
<td> **family-name** </td>
<td>nom de la famille d'entité (\arcane{IItemFamily} auquel la
variable est associée. Cet attribut n'est valide que si **item-kind**
vaut \c particle ou \c dof.
</td>
</tr>

</table>

#### Dimensions de la variable

Les attributs suivants permettent de définir la dimension de la
variable.

\note Les valeurs **shape-dim**, **extent0** et **extent1** sont
uniquement disponibles à partir de la version 3.8.11 de %Arcane.

Si aucun des attributs suivants n'est défini, alors on considère que
la variable est une variable scalaire comme si la valeur **dim="0"**
avait été spécifiée.

Il faut spécifier soit **dim** (pour les variables classiques
scalaires, 1D ou 2D, soit **shape-dim** pour les variables
multi-dimensionnelles.

<table>
<tr>
<th>Attribut</th>
<th>Description</th>
</tr>

<tr>
<td> **dim** </td>
<td>La dimension de la variable pour les variables scalaires, 1D ou
2D.Les valeurs possibles sont *0* pour
une variable scalaire, *1* pour une variable tableau à une dimension et
*2* pour un tableau à deux dimensions. Les tableaux à deux dimensions
ne sont pas supportés pour les variables du maillage.
</td>
</tr>

<tr>
<td> **shape-dim** </td>
<td>La dimension de la variable pour les variables
multi-dimensionnelles. Les valeurs possibles sont 0, 1, 2 ou 3.
</td>
</tr>

<tr>
<td> **extent0** </td>
<td>Nombre d'élément pour une variable multi-dimension
dont les éléments sont des vecteurs ou nombre de lignes pour une variable multi-dimension
dont les éléments sont des matrices. Cet attribut ne peut être utilisé
que si **shape-dim** est présent.
</td>
</tr>

<tr>
<td> **extent1** </td>
<td>Nombre de colonnes pour une variable multi-dimension
dont les éléments sont des matrices. Cet attribut ne peut être utilisé
que si **shape-dim** est présent.
</td>
</tr>

</table>

#### Propriétés de la variable

Les attributs suivants sont optionnels et permettent de spécifier des
propriétés sur la variable:

<table>
<tr>
<th>Attribut</th>
<th>Description</th>
</tr>

<tr>
<td> **dump** </td>
<td>permet de choisir si les valeurs d'une variable sont sauvegardées
lors d'un arrêt du code. Dans ce cas, les valeurs sauvegardées seront,
bien entendu, relues lors d'une reprise d'exécution. Certaines
variables recalculées n'ont pas besoin d'être sauvegardées ; dans ce
cas l'attribut *dump* vaut *false*. C'est le cas lorsque la valeur
d'une variable n'est pas utile d'une itération à l'autre.
</td>
</tr>

<tr>
<td> **need-sync** </td>
<td>indique si la variable doit être synchronisé entre
sous-domaines. Il s'agit juste d'une indication qui peut être utilisée
lors de vérifications.
</td>
</tr>

</table>

Lors de la compilation du descripteur de module par %Arcane (avec \c axl2cc - cf précédemment), 
les variables sont enregistrées au sein de la base de données de l'architecture.

#### Exemple de déclaration de variables dans les fichiers AXL

```xml

<!-- Variable scalaire aux noeuds de 'Real' -->
<variable field-name="node_mass" name="NodeMass"
          data-type="real" item-kind="node" dim="0" />

<!-- Variable scalaire aux mailles de 'Int64' -->
<variable field-name="cell_unique_id" name="UniqueId"
          data-type="int64" item-kind="cell" dim="0" />

<!-- Variable 1D aux mailles de 'Real3' -->
<variable field-name="cell_cqs" name="CellCQS"
          data-type="real3" item-kind="cell" dim="1" />

<!-- Variable scalaire aux particules de 'Real' sur la famille 'ArcaneParticles' -->
<variable field-name="particle_temperature" name="Temperature" family-name="ArcaneParticles"
          data-type="real" item-kind="particle" dim="0" />

<!-- Variable multi-dim 0D aux mailles de 'Real' -->
<variable field-name="mdvar0d" name="MDVar0D"
          data-type="real" item-kind="cell" shape-dim="0" />

<!-- Variable multi-dim 1D aux mailles de 'Real' -->
<variable field-name="mdvar1d" name="MDVar1D"
          data-type="real" item-kind="cell" shape-dim="1" />

<!-- Variable multi-dim 2D aux mailles de 'Real' -->
<variable field-name="mdvar2d" name="MDVar2D"
          data-type="real" item-kind="cell" shape-dim="2" />

<!-- Variable multi-dim 2D aux mailles de 'Real' vue comme une variable 3D -->
<variable field-name="mdvar2d_as_3d" name="MDVar2D"
          data-type="real" item-kind="cell" shape-dim="3" />

<!-- Variable multi-dim 0D aux mailles de NumVector<Real,2> -->
<variable field-name="mdvar0d_vector2" name="MDVar0DVector2"
          data-type="real2" item-kind="cell" shape-dim="0" />

<!-- Variable multi-dim 1D aux faces de NumVector<Real,3> -->
<variable field-name="mdvar1d_vector3" name="MDVar1DVector3"
          data-type="real3" item-kind="face" shape-dim="1" />

<!-- Variable multi-dim 2D aux faces de NumVector<Real,4> -->
<variable field-name="mdvar2d_vector4" name="MDVar2DVector4"
          data-type="real" item-kind="face" shape-dim="2" extent0="4" />

<!-- Variable multi-dim 0D aux noeuds de NumMatrix<Real,2,2> -->
<variable field-name="mdvar0d_matrix2x2" name="MDVar0DMatrixReal2x2"
          data-type="real2x2" item-kind="node" shape-dim="0" />

<!-- Variable multi-dim 1D aux mailles de NumMatrix<Real,3,3> -->
<variable field-name="mdvar1d_matrix3x3" name="MDVar1DMatrixReal3x3"
          data-type="real3x3" item-kind="cell" shape-dim="1" />

<!-- Variable multi-dim 1D aux mailles de NumMatrix<Real,2,6> -->
<variable field-name="mdvar1d_matrix2x6" name="MDVar1DMatrix2x6"
          data-type="real" item-kind="cell" shape-dim="1" extent0="2" extent1="6"/>

<!-- Variable multi-dim 0D aux mailles de NumMatrix<Real,3,2> -->
<variable field-name="mdvar0d_matrix3x2" name="MDVar0DMatrix3x2"
          data-type="real" item-kind="cell" shape-dim="0" extent0="3" extent1="2"/>
```

## Utilisation des variables {#arcanedoc_core_types_axl_variable_use}

### Utilisation des variables scalaires, 1D et 2D

La manière d'utiliser une variable est identique quel que soit son
type et ne dépend que de son genre.

#### Variables scalaires
	
Les variables scalaires sont utilisées par l'intermédiaire de la classe template
\arcane{VariableRefScalarT}.

Il n'y a que deux possibilités d'utilisation :

- "lire la valeur" : cela se fait par
  l'opérateur() (\arcane{VariableRefScalarT::operator()()}). Cet opérateur retourne une
  référence constante sur la valeur stockée dans la variable. Il peut être utilisé
  partout ou l'on veut utiliser une valeur du type de la variable.
- "changer la valeur" : cela se fait par l'opérateur = (\arcane{VariableRefScalarT::operator=()})

Par exemple, avec la variable \c m_time de type \c VariableScalarReal :

```cpp
m_time = 5.0;        // affecte la valeur 5. à la variable m_time
double z = m_time(); // récupère la valeur de la variable et l'affecte à z.
cout << m_time();    // imprime la valeur de m_time
```

L'important est de ne pas oublier les parenthèses lorsqu'on veut
accéder à la valeur de la variable.

#### Variables tableaux
	
Les variables tableaux sont utilisées par l'intermédiaire de la classe template
\arcane{VariableRefArrayT}.

Leur fonctionnement est assez similaire à la classe \c vector
de la STL. Le dimensionnement du tableau se fait par la méthode
\arcane{VariableRefArrayT::resize()} et chaque élément du tableau peut être accédé par l'opérateur
\arcane{VariableRefArrayT::operator[]()} qui retourne une référence sur le type des éléments du
tableau.

Par exemple, avec la variable *m_times* de type \arcane{VariableArrayReal} :

```cpp
Arcane::VariableArrayReal m_times = ...;
m_times.resize(5);         // redimensionne le tableau pour contenir 5 éléments
m_times[3] = 2.0;          // affecte la valeur 2.0 au 4ème élément du tableau
cout << m_times[0];        // imprime la valeur du premier élément
```

#### Variables scalaires sur le maillage
	
Il s'agit des variables sur les éléments du maillage (noeuds,
faces ou mailles) avec une valeur par élément. Ces variables sont
définies par la classe template \arcane{MeshVariableScalarRefT}.

Leur fonctionnement est assez similaire à celui d'un tableau C
standard. On utilise l'opérateur \arcane{VariableRefArrayT::operator[]()}
pour récupérer une référence sur le type de la variable pour un
élément du maillage donné. Cet opérateur est surchargé pour prendre
en argument un itérateur sur un élément du maillage.

Les grandeurs se déclarent et s'utilisent de manière similaire
quel que soient le type d'élément du maillage. Elles sont dimensionnées
automatiquement lors de l'initialisation au nombre d'éléments du
maillage du genre de la variable.

Par exemple, avec la variable *m_volume* de type \arcane{VariableCellReal}:

```cpp
Arcane::VariableCellReal m_volume = ...;
ENUMERATE_CELL(icell,allCells()) {
  m_temperature[icell] = 2.0;     // Affecte la valeur 2.0 au volume de la maille courante
  cout << m_temperature[icell];   // Imprime le volume de la maille courante

  // il est possible de faire les mêmes opérations avec la maille
  // ATTENTION c'est moins performant
  m_temperature[icell] = 2.0;  // Affecte la valeur 2.0 au volume de la maille 'cell'
  cout << m_temperature[icell];    // Imprime le volume de la maille 'cell'
}
```

#### Variables tableaux sur le maillage

Il s'agit des variables sur les éléments du maillage (noeuds,
faces ou mailles) avec un tableau de valeurs par éléments. Ces variables sont
définies par la classe template \arcane{MeshVariableArrayRefT}.

Le fonctionnement de ces variables est identique à celui des
variables scalaires sur le maillage mais l'opérateur
\arcane{MeshVariableArrayRefT::operator[]()} retourne un tableau de valeurs
du type de la variable.

Il est possible de changer le nombre d'éléments de la deuxième
dimension de ce tableau par la méthode \arcane{MeshVariableArrayRefT::resize()}.

Par exemple, avec la variable *m_temperature* de type \arcane{VariableCellArrayReal}:

```cpp
Arcane::VariableCellArrayReal m_temperature = ...;
m_temperature.resize(3); // Chaque maille aura 3 valeurs de temperature
ENUMERATE_CELL(icell,allCells()) {
  // Affecte la valeur 2.0 à la première temperature de la maille courante
  m_temperature[icell][0] = 2.0;

  // Imprime la 2-ème température de la maille courante
  info() << m_temperature[icell][1];

  // Il est possible de faire les mêmes opérations avec la maille

  // Déclare une référence à une maille.
  Cell cell = *icell;
  // Affecte la valeur 2.0 à la première temperature de la maille 'cell'
  m_temperature[cell][0] = 2.0;

  // Affecte la valeur 4.0 à la 3-ème temperature de la maille 'cell'
  m_temperature(cell,2) = 4.0;

  // Imprime la 2-ème température de la maille 'cell'
  info() << m_temperature[cell][1];
}
```

### Utilisation des variables multi-dimensionnelles sur le maillage {#arcanedoc_core_types_axl_md_variable_use}

\warning Les variables multi-dimensionnelles sont une fonctionnalité
expérimentale et l'API n'est pas encore figé. Le code actuel n'est pas
garanti de pouvoir être utilisé dans une version ultérieure de
%Arcane. De même, certains mécanismes impliquants ces variables (comme
le dépouillement) peuvent ne pas être complètement opérationnels.

A partir de la version 3.8.11 de %Arcane il est possible d'utiliser
des variables multi-dimensionnelles sur la maillage. Ces variables
unifient l'utilisation des variables quelle que soit leur dimension et
permettent de supporter jusqu'à 3 dimensions (à noter que cette valeur
pourra probablement être augmentée dans une future version.

\note Pour l'instant ces variables ont forcément le type de donnée
\arcane{Real}.

Il existe trois types d'éléments pour les variables multi-dimensionnelles :
- les scalaires \arcane{MeshMDVariableRefT}
- les vecteurs \arcane{MeshVectorMDVariableRefT}
- les matrices \arcane{MeshMatrixMDVariableRefT}

| Nom C++                            |  Type d'élément  | Dimension maximale
|------------------------------------|------------------|----------------------
| \arcane{MeshMDVariableRefT}        | scalaire         | 3
| \arcane{MeshVectorMDVariableRefT}  | vecteur          | 2
| \arcane{MeshMatrixMDVariableRefT}  | matrice          | 1

Ces trois variables ont une classe de base commune \arcane{MeshMDVariableRefBaseT}.

L'utilisation est similaire quel que soit le type de
l'élément. L'accès se fait via l'opérateur `operator()` car il permet
d'avoir plusieurs arguments. A partir du C++23 il sera aussi possible
d'utiliser l'opérateur `operator[]`.

Avant d'utiliser ces variables, il est nécessaire d'appeler la méthode
\arcane{MeshMDVariableRefT::reshape()}. Le nombre de valeurs à
spécifier doit être identique à la dimension de la variable.

Exemple pour les variables multi-dimensionnelles scalaires:

\snippet MDVariableUnitTest.cc SampleMDVariableScalar

Exemple pour les variables multi-dimensionnelles vectorielles:

\snippet MDVariableUnitTest.cc SampleMDVariableVector

Exemple pour les variables multi-dimensionnelles matricielles:

\snippet MDVariableUnitTest.cc SampleMDVariableMatrix

En interne, toutes ces variables sont similaires à une variable
tableau sur le maillage dont le nombre d'éléments est égale au produit
des composantes de la forme retournée par
\arcane{MeshMDVariableRefBaseT::fullShape()}.
____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_entrypoint
</span>
</div>
