# Jeu de données (.ARC) {#arcanedoc_core_types_casefile}

[TOC]

## Introduction {#arcanedoc_core_types_casefile_intro}

Ce chapître décrit la structure d'un jeu de données. Le jeu de
données est un fichier au format [XML](https://www.w3.org/TR/xml) qui
contient les valeurs permettant de paramétrer l'exécution d'un code de
calcul.

Voici un exemple de jeu de donnée en anglais :

```xml
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
 <arcane>
  <title>Tube à choc de Sod</title>
  <description>Ce JDD permet de tester le module Hydro simplifié de
     Arcane</description>
  <timeloop>ArcaneHydroLoop</timeloop>
 </arcane>

 <!-- Liste des maillages (toujours en anglais) -->
 <meshes>
  <mesh>
   <filename>tube5x5x100.ice</filename>
  </mesh>
 </meshes>

 <functions>
  <table name="table-dt" parameter="time" value="real" interpolation="linear">
   <value><x>0.0</x><y>1.0e-3</y></value>
   <value><x>1.0e-2</x><y>1.0e-5</y></value>
  </table>
 </functions>

 <simple-hydro>
  <deltat-init>0.001</deltat-init>
  <deltat-min>0.0001</deltat-min>
  <deltat-max>0.01</deltat-max>
  <final-time>0.2</final-time>
 </simple-hydro>

</case>
```

Un jeu de données se compose de quatre parties :

- l'élément `<arcane>` permettant entre autre de configurer la boucle
  en temps et les modules actifs (voir \ref
  arcanedoc_core_types_casefile_arcaneelement)
- l'élément maillage (`<meshes>`) permettant de décrire le
  maillage (voir \ref arcanedoc_core_types_casefile_meshes).
- l'élément contenant les fonctions (`<functions>` ou
  `<fonctions>`)(voir \ref
  arcanedoc_core_types_casefile_functions)
- les éléments restants concernent les options des différents
  modules du code.

Seule la balise `<arcane>` est requise. Les autres balises sont
optionnelles

\warning Les caractères blancs dans les attributs sont interdits dans
la norme XML. Le jeu de données est donc invalide s'il y en
a. Cependant les analyseurs peuvent les tolérer. Les caractères
blancs en début et fin de balises sont
significatifs et peuvent changer la signification ou rendre invalide
un jeu de données. Par exemple, <file> Toto</file> indique que le
fichier de maillage contient un blanc avant les caractères 'Toto'.

\note Pour des raisons historiques, il est toléré d'avoir des
espaces en début ou fin de texte d'une balise dans le cas des
options simples du jeu de données (voir \ref
arcanedoc_core_types_axl_caseoptions_options_simple). Dans ce cas, ces espaces sont
ignorés. Par exemple `<deltat>  25.0  </deltat>` est valide. Ces
blancs ne sont autorisés que dans les options des modules et
services, pas dans les balises spécifiques à %Arcane (comme
`<arcane>`, `<functions>`, `<meshes>`, ...).

## Élément <arcane> {#arcanedoc_core_types_casefile_arcaneelement}

Cet élément contient les informations sur la boucle en temps
utilisée et la liste des modules actifs. Le contenu de cet élément
est la première chose lue dans le jeu de données. Les éléments
suivants sont possibles :

```xml
<arcane>
  <title>Tube à choc de Sod</title>
  <description>Ce JDD permet de tester le module Hydro simplifié de Arcane</description>
  <timeloop>ArcaneHydroLoop</timeloop>
  <modules>
    <module name="Hydro" actif="true" />
    <module name="PostProcessing" actif="false" />
  </modules>
  <configuration>
    <parameter name="NotParallel" value="false" />
    <parameter name="NotCheckpoint" value="true" />
  </configuration>
</arcane>
```

Le tableau suivant donne la liste des éléments possibles :

<table>
<tr>
<th>élément anglais</th>
<th>élément francais</th>
<th>description</th>
</tr>
<tr>

<td><b>timeloop</b></td>
<td><b>boucle-en-temps</b></td>
<td> nom de la boucle en temps utilisée. Ce nom doit correspondre à
une boucle en temps disponible dans le fichier de configuration du code.
</td>
</tr>

<tr>
<td><b>title</b></td>
<td><b>titre</b></td>
<td> titre du jeu de données. Purement informatif.
</td>
</tr>

<tr>
<td><b>description</b></td>
<td><b>description</b></td>
<td> description du cas test. Purement informatif.
</td>
</tr>

<tr>
<td><b>modules</b></td>
<td><b>modules</b></td>
<td> Liste des modules avec leur état d'activation. Cette balise est
utilisée pour indiquer si un module optionnel est activé ou non. Par
défaut les modules optionnels ne sont pas actifs. Il s'agit d'une
liste d'éléments `<module>` comme suit :

```xml
<module name="Module1" active='true' />
<module name="Module2" active='false' />
```
</td>
</tr>

<tr>
<td><b>services</b></td>
<td><b>services</b></td>
<td> Liste de services singletons avec leur état d'activation (qui
vaut *vrai* par défaut). Il s'agit d'une liste d'éléments <service>
comme suit :

```xml
<service name="Service1" active='true' />
<service name="Service2" active='false' />
<service name="Service3" />
```
Dans l'exemple précédent, les services de nom 'Service1' et 'Service3' seront chargés.
</td>
</tr>

<tr>
<td><b>configuration</b></td>
<td><b>configuration</b></td>
<td> Liste de paramêtres de configuration. Ces paramètres ne sont pas
lus par %Arcane mais peuvent être utilisés par exemple par la procédure
de lancement du code. Chaque paramètre est de la forme suivante :

```xml
<parameter name="Param1" value='value1' /> <!-- Anglais -->
<parametre name="Param1" value='value1' /> <!-- Francais -->
```
</td>
</tr>
</table>

## Maillages (balise <meshes>) {#arcanedoc_core_types_casefile_meshes}

Les maillages sont gérées par le service `ArcaneCaseMeshService`. Les
valeurs possibles sont décrites dans la page \ref
axldoc_service_ArcaneCaseMeshService_arcane_impl. Il est possible de
spécifier plusieurs maillages. Par exemple :

~~~xml
<meshes>
  <mesh>
    <filename>sod.vtk</filename>
  </mesh>
  <mesh>
    <filename>plancher.msh</filename>
  </mesh>
</meshes>
~~~

Il existe une autre possibilité pour spécifier les maillages. Cette
possibilité est déclarée obsolète et ne doit être utilisée que par les
codes existants. Pour ces codes, on utilise la balise `<mesh>` (ou
`<maillage>` en francais) pour spécifier les informations du
maillage. Par exemple :

~~~xml
<maillage>
  <fichier>sod.vtk</fichier>
</maillage>
<maillage>
  <fichier>plancher.msh</fichier>
</maillage>
~~~

## Fonctions (balises <fonctions> ou <functions>) {#arcanedoc_core_types_casefile_functions}

Il est possible de définir dans le jeu de données des fonctions qui
sont utilisées pour faire varier les valeurs d'une option en fonction
du temps physique ou de l'itération. L'ensemble des fonctions est
défini dans la balise `<fonctions>` si la langue est le francais ou
`<functions> si la langue est l'anglais.

\note Dans la suite du document on utilisera uniquement les termes en
anglais pour améliorer la lisibilité.

Une fonction doit avoir un nom unique qui est utilisé par l'option
pour la référencer. L'exemple suivante montre comment définir une
fonction `table-dt` et l'utiliser en référence dans l'option
`<my-option>` :

~~~{xml}
<?xml version="1.0"?>
<case codename="ArcaneTest" xml:lang="en" codeversion="1.0">
 <arcane>
 </arcane>

 <functions>
  <table name="table-dt" parameter="time" value="real" interpolation="linear">
   ...
  </table>
 </functions>
 <my-module>
  <my-option function="table-dt">0.01</function>
 </my-module>
~~~

Les fonctions prennent un seul argument en paramètre et peuvent
retourner le type attendu par l'option qui les utilise. Les deux
valeurs possibles pour les paramètres sont :

- le temps physique. Dans ce cas le type du paramètre est un `reel`.
- le numéro de l'itération. Dans ce cas le type du paramètre est un entier.

Au début de chaque itération, %Arcane met automatiquement à jour les
options du jeu de données qui référencent une fonction (via la méthode
\arcane{ICaseMng::updateOptions()}).

La balise `<functions>` permet de définir deux types de fonction:

- les tables de marche qui sont spécifiées directement dans le jeu de
  données. Il s'agit de fonctions linéaires continues ou linéaires par
  morceau.
- des fonctions externes écrite en C#. Dans ce cas il est possible de
  définir n'importe quel type de fonction dont la signature correspond
  au type attendu par l'option. Le chapître \ref
  arcanedoc_wrapping_csharp_casefunction indique comment définir et
  utiliser ces fonctions.

### Syntaxe des tables de marche

Une table de marche est une fonction linéaire continue ou linéaire par
morceau définie par un ensemble de couples `(X,Y)`. Par exemple :

~~~{xml}
 <!-- Exemple en anglais -->
 <functions>
  <table name="table-dt" parameter="time" value="real" interpolation="linear">
   <value><x>0.0</x><y>1.0e-3</y></value>
   <value><x>1.0e-2</x><y>1.0e-5</y></value>
  </table>
 </functions>
~~~

~~~{xml}
 <!-- Exemple en francais -->
 <table nom='test-time-real-2' parametre='temps' valeur='reel' interpolation='lineaire'>
  <valeur> <x>0.0</x> <y>3.0</y> </valeur>
  <valeur> <x>4.0</x> <y>9.0</y> </valeur>
  <valeur> <x>5.0</x> <y>7.</y> </valeur>
  <valeur> <x>6.0</x> <y>2.0</y> </valeur>
  <valeur> <x>10.0</x><y>-1.0</y> </valeur>
  <valeur> <x>14.0</x><y>-3.0</y> </valeur>
 </table>
~~~

Les couples `(X,Y)` doivent être rangés par valeur croissante de
`X`. Si la valeur du paramètre est plus petite ou plus grande que la
première ou la dernière valeur de la tablea de marche, on prend cette
dernière. Dans l'exemple précédent pour la table de marche
`test-time-real-2`, si `X<0.0` on retourne `3.0` et si `X>14.0` on
retourne `-3.0`.

Les tables de marche ont les attributes suivants :

<table>
<tr>
<th>Nom Anglais</th>
<th>Nom Francais</th>
<th>Type</th>
<th>Description</th>
</tr>

<tr>
<td>nom</td>
<td>name</td>
<td>string</td>
<td>Nom de la table de marche.
</td>

</tr>

<tr>
<td>parameter</td>
<td>parametre</td>
<td>string</td>
<td>Type du paramètre. Les valeurs possibles sont `time` (`temps` en
francais) pour un paramètre qui est le temps physique ou `iteration`
pour un paramètre qui est le numéro de l'itération courante.
</td>
</tr>

<tr>
<td>value</td>
<td>valeur</td>
<td>string</td>
<td>Type de la valeur de retour de la table de marche. Les valeurs
possibles sont `real`, `integer`, `real3`, `string` ou `bool`
(respectivement `reel`, `entier`, `reel3`, `string` et `bool` en francais).
</td>
</tr>

<tr>
<td>interpolation</td>
<td>interpolation</td>
<td>string</td>
<td>Les valeurs possibles sont `linear` ou `constant` (`lineaire` ou
`constant-par-morceaux` en francais). Si l'interpolation est
constante, la valeur retournée est celle correspondante de `Y` qui
correspond au `X` immédiatement inférieur à la valeur du paramètre. Si
l'interpolation est linéaire, on réalise une interpolation linéaire
entre `(X1,Y1)` et `(X2,Y2)`, avec `X1` la valeur de `X` immédiatement
inférieure au paramètre et `X2` la valeur suivante dans la table de
marche. Pour la table de marche précedente (`test-time-real-2`) en
exemple, si `X` vaut `4.5` alors on retourne `9.0` si l'interpolation
est constante par morceaux et `8.0` (soit `Y1 +
(X-X1)*(Y2-Y1)/(X2-X1)` <=> `9.0 + (4.5-4.0)*(7.0-9.0)/(5.0-4.0)`) si
l'interpolation est linéaire.
</td>
</tr>

<tr>
<td>comul</td>
<td>comul</td>
<td>string</td>
<td>Coefficient multiplicateur pour la valeur. Cet attribut est
optionnel et doit être du même type que la valeur de la fonction. S'il
est présent, la valeur de la fonction est multipliée par la valeur de
cet attribut (pour un `Real3`, la multiplication est faite composante
par composante)
</td>
</tr>

<tr>
<td>deltat-coef</td>
<td>deltat-coef</td>
<td>real</td>
<td>Coefficient multiplicateur pour le temps physique. Cet attribut est
optionnel et est utilisé pour multiplier la valeur du pas de temps
courant (voir \arcane{ICaseMng::updateOptions()}).
</td>
</tr>

</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_default_values
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_codeconfig
</span>
</div>
