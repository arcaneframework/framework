# Jeu de données (.ARC) {#arcanedoc_core_types_casefile}

[TOC]

## Introduction {#arcanedoc_core_types_casefile_intro}

Ce chapître décrit la structure d'un jeu de données. Le jeu de
données est un fichier au format XML qui contient les valeurs
permettant de paraméter l'exécution d'un code de calcul.

Voici un exemple de jeu de donnée en anglais:

```xml
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
 <arcane>
  <title>Tube à choc de Sod</title>
  <description>Ce JDD permet de tester le module Hydro simplifié de
     Arcane</description>
  <timeloop>ArcaneHydroLoop</timeloop>
 </arcane>

 <mesh>
  <file>tube5x5x100.ice</file>
 </mesh>

 <simple-hydro>
  <deltat-init>0.001</deltat-init>
  <deltat-min>0.0001</deltat-min>
  <deltat-max>0.01</deltat-max>
  <final-time>0.2</final-time>
 <simple-hydro>

</case>
```

Un jeu de données se compose de quatres parties:
- l'élément <arcane> permettant entre autre de configurer la boucle
  en temps et les modules actifs.
- l'élément maillage (<mesh> ou <maillage>) permettant de décrire le
  maillage. Cet élément peut-être présent plusieurs fois.
- l'élément contenant les tables de marche.
- les éléments restants concernent les options des différents
  modules du code.

  TODO indiquer syntaxe pour les tables de marche et le maillage TODO

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
ignorés. Par exemple <deltat>  25.0  </deltat> est valide. Ces
blancs ne sont autorisés que dans les options des modules et
services, pas dans les balises <arcane> ou <maillage>/<mesh>.

## Élément <arcane> {#arcanedoc_core_types_casefile_arcaneelement}

Cet élément contient les informations sur la boucle en temps
utilisée et la liste des modules actifs. Le contenu de cet élément
est la première chose lue dans le jeu de données. Les éléments
suivants sont possibles:

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

 Le tableau suivant donne la liste des éléments possibles:

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
<td> Liste des modules avec leur état d'activation. Il s'agit d'une
liste d'éléments <module> comme suit:

```xml
<module name="Module1" active='true' />
<module name="Module2" active='false' />
```
</td>
</tr>

<tr>
<td><b>services</b></td>
<td><b>services</b></td>
<td> Liste de services singletons avec leur état d'activation (qui vaut *vrai* par défaut). Il s'agit d'une
liste d'éléments <service> comme suit:

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
<td> Liste de paramêtres de configuration. Ces paramêtres ne sont pas
lus par %Arcane mais peuvent être utilisés par exemple par la procédure
de lancement du code. Chaque paramêtre est de la forme:
liste d'éléments <module> comme suit:

```xml
<parameter name="Param1" value='value1' /> <!-- Anglais -->
<parametre name="Param1" value='value1' /> <!-- Francais -->
```
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