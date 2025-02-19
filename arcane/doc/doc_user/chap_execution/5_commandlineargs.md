# Ligne de commande et jeu de données {#arcanedoc_execution_commandlineargs}

[TOC]

## Introduction {#arcanedoc_execution_commandlineargs_intro}

[//]: # (Le jeu de données `.arc` &#40;\ref arcanedoc_core_types_casefile&#41; contenant les options des modules et des services)

Il y a deux possibilités de personnalisation des options du jeu de données
via les arguments de la ligne de commande :
- par remplacement de symboles,
- par adresse de l'option *(méthode recommandée)*.

Il est possible de personnaliser tous les types d'options %Arcane listées
ici : \ref arcanedoc_core_types_axl_caseoptions_options

## Personnalisation par symboles {#arcanedoc_execution_commandlineargs_symbol}

### Les symboles dans le jeu de données {#arcanedoc_execution_commandlineargs_symbol_dataset}

Ce type de personnalisation nécessite de modifier le jeu de données pour y inclure
des symboles. Ce jeu de données peut donc devenir inutilisable sans les bons
arguments de la ligne de commande.

Un symbole est une chaîne de caractères entourée d'arrobases.

Exemple : `@UneValeur@`

Si on met ce symbole dans un jeu de données, on pourrait avoir, par exemple :

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <deltat-cp>@UneValeur@</deltat-cp>
  </simple-hydro>
  
</case>
```

Que l'option `deltat-cp` soit une *option simple*, une *option énumérée* ou
une *option étendue*, le fonctionnement est le même, y compris si ces options
sont dans des *options complexes* ou des *options services*.

Pour les *options services*, le remplacement des symboles fonctionne aussi dans
les attributs `name` et `mesh-name`. Exemple :

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <post-processor2 name="@NamePostProcessor@" mesh-name="@MeshPostProcessor@">
      <fileset-size>@NbTimeInOneFile@</fileset-size>
      <binary-file>false</binary-file>
    </post-processor2>
  </simple-hydro>
  
</case>
```

Trois symboles peuvent être remplacés : `@NamePostProcessor@`, `@MeshPostProcessor@`
et `@NbTimeInOneFile@`.

\remark Ici, on peut voir une première limite au remplacement de symboles
pour les *options services* : si on remplace le symbole `@NamePostProcessor@`
par `Ensight7PostProcessor`, parfait.
En revanche, si on le remplace par `VtkHdfV2PostProcessor`, ça va être problématique
car ces deux services n'utilisent pas les mêmes options ! Les options `fileset-size`
et `binary-file` n'existent pas dans `VtkHdfV2PostProcessor`.

Le remplacement de symboles peut aussi être utilisé dans les *options simples* ayant
un type tableau.

Prenons l'option suivante :

```xml
<!--Fichier AXL-->
<simple name="simple-real-array" type="real[]">
  <description>Tableau de réel</description>
</simple>
```

Dans le jeu de données, ajoutons un symbole :

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <simple-real-array>3.0 @DeuxiemeElement@ 3.2 3.3</simple-real-array>
  </simple-hydro>
  
</case>
```

On peut remplacer le symbole `@DeuxiemeElement@` par `3.1`,
ou par plusieurs valeurs : `3.1 3.11`.


### Attribution d'une valeur à un symbole dans la ligne de commande {#arcanedoc_execution_commandlineargs_symbol_command}

\note Pour l'instant, il est nécessaire de définir la variable
d'environnement `ARCANE_REPLACE_SYMBOLS_IN_DATASET`.
```shell
export ARCANE_REPLACE_SYMBOLS_IN_DATASET=1
```

Lorsque le jeu de données contient les symboles voulus, on peut attribuer leurs
valeurs lors de l'exécution.

Cette attribution reprend la syntaxe des arguments %Arcane (`-A,`).

Admettons que l'on ait, dans le jeu de données, les symboles `@NamePostProcessor@`,
`@MeshPostProcessor@` et `@NbTimeInOneFile@`.

Pour attribuer une valeur à ces symboles, on peut lancer l'application comme ceci :

<div class="tabbed">

- <b class="tab-title">Multiple `-A,`</b>
<div>
  ```sh
  ./app \
  -A,NamePostProcessor=Ensight7PostProcessor \
  -A,MeshPostProcessor=Mesh1 \
  -A,NbTimeInOneFile=10 \
  dataset_with_symbols.arc
  ```
</div>

- <b class="tab-title">Unique `-A,`</b>
<div>
  ```sh
  ./app \
  -A,NamePostProcessor=Ensight7PostProcessor,MeshPostProcessor=Mesh1,NbTimeInOneFile=10 \
  dataset_with_symbols.arc
  ```
</div>

</div>


Il est aussi possible d'entourer les valeurs par des guillemets `""`.
C'est particulièrement utile pour les types tableaux :
```sh
./app \
-A,DeuxiemeElement="3.1 3.11" \
dataset_with_symbols.arc
```

Lorsqu'un symbole est présent dans le jeu de données mais absent de la ligne
de commande, le symbole est simplement remplacé par une chaîne de caractères vide.

\warning Dans %Arcane, une différence est faite entre une option
vide `<deltat-cp></deltat-cp>` et une option absente du jeu de données. Dans
le premier cas, la valeur de l'option est **vide** (`String("")`)
mais présente donc n'est pas remplacée par la valeur par défaut.
Dans le second cas, la valeur de l'option est **null** (`String()`) donc est remplacée
par la valeur par défaut.



## Personnalisation par adresse de l'option {#arcanedoc_execution_commandlineargs_addr}

### Les options uniques {#arcanedoc_execution_commandlineargs_addr_unique}

Par rapport au remplacement de symboles, cette méthode permet de conserver un
jeu de données valide sans arguments obligatoires (mais est un peu plus verbeuse).

Ces deux méthodes agissent aussi aux mêmes endroits en interne, donc les
possibilités sont les mêmes : il est possible de modifier la valeur
d'une *option simple*, d'une *option énumérée* ou d'une *option étendue*, même
si ces options sont dans des *options complexes* ou des *options services*.

Pour les *options services*, comme précédemment, on peut agir sur
les attributs `name` et `mesh-name`.

Reprenons le premier exemple :
```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <deltat-cp>3.0</deltat-cp>
  </simple-hydro>
  
</case>
```

Si l'on souhaite modifier la valeur de l'option `deltat-cp`, il suffit de
lancer l'application comme ceci :

```sh
./app \
-A,//simple-hydro/deltat-cp=3.1 \
dataset.arc
```

Il est nécessaire d'avoir l'adresse (XPath) de l'option. Pour la retrouver, il faut
dérouler les éléments XML. Ici, l'option `deltat-cp` est à l'adresse :
`//case/simple-hydro/deltat-cp`.
Ensuite, on retire le `case/` au début (ou `cas/` pour les jeux de données en français).
Enfin, on peut construire l'argument à ajouter : 

`-A,``//simple-hydro/deltat-cp``=3.1`

Comme pour le remplacement de symboles, on peut ajouter des guillemets :

`-A,//simple-hydro/simple-real-array="3.1 3.11 3.12"`

Dans le cas des attributs, ils sont désignés par un `@` (pour l'attribut `name`, on
doit mettre `@name` dans l'adresse).
Si l'on souhaite modifier l'attribut `name` d'un service, on peut le faire comme ceci :

`-A,//simple-hydro/post-processor/@name=VtkHdfV2PostProcessor`

### Les options multiples {#arcanedoc_execution_commandlineargs_addr_multi}

Mais un problème apparait assez vite : comment faire si on a des options multiples ?

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <deltat-cp>3.0</deltat-cp>
    <deltat-cp>6.0</deltat-cp>
    <deltat-cp>7.0</deltat-cp>
  </simple-hydro>
  
</case>
```

En XML, dans ce genre de cas, on utilise des indices.
Ainsi, pour les trois valeurs de `deltat-cp`, on les adresse comme ceci :

`//simple-hydro/deltat-cp[1]`<br>
`//simple-hydro/deltat-cp[2]`<br>
`//simple-hydro/deltat-cp[3]`

\warning Il n'y a pas d'erreur au niveau des indices. En XML, les indices commencent
à 1, et non 0.

On peut aussi les écrire comme ceci :

`//simple-hydro[1]/deltat-cp[1]`<br>
`//simple-hydro[1]/deltat-cp[2]`<br>
`//simple-hydro[1]/deltat-cp[3]`

Ces syntaxes sont gérées par %Arcane. Ainsi, pour modifier la seconde option,
on peut écrire :

```sh
./app \
-A,//simple-hydro/deltat-cp[2]=6.1 \
dataset.arc
```

Ou bien :

```sh
./app \
-A,//simple-hydro[1]/deltat-cp[2]=6.1 \
dataset.arc
```

### L'ajout d'options {#arcanedoc_execution_commandlineargs_addr_add_option}

Une chose que l'on peut faire aussi est d'ajouter des options.

Si l'on souhaite ajouter une quatrième option `deltat-cp`, il est possible d'ajouter l'argument :

`-A,//simple-hydro/deltat-cp[4]=9.0`

Il est aussi possible d'ajouter des options non présentes dans le jeu de données (mais présentes dans l'AXL) :

```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>

  <simple-hydro>
  </simple-hydro>

</case>
```

On pourrait ajouter les options via les arguments de la ligne de commande :

```sh
./app \
-A,//simple-hydro/deltat-cp[1]=3.0 \
-A,//simple-hydro/deltat-cp[2]=6.0 \
-A,//simple-hydro/deltat-cp[3]=7.0 \
dataset.arc
```

Ça fonctionne aussi pour les *options services*. Dans ce cas, il est nécessaire d'ajouter au moins l'attribut `name` :

```sh
./app \
-A,//simple-hydro/post-processor/@name=VtkHdfV2PostProcessor \
dataset.arc
```

Puis, on peut, par exemple, modifier les *options simples* de ce service ou l'attribut `mesh-name` :

```sh
./app \
-A,//simple-hydro/post-processor/@name=VtkHdfV2PostProcessor \
-A,//simple-hydro/post-processor/@mesh-name=Mesh1 \
-A,//simple-hydro/post-processor/max-write-size=50 \
dataset.arc
```

\warning Il n'est pas encore possible d'ajouter des *options complexes*.

#### Et si on commence à l'indice 2 ? {#arcanedoc_execution_commandlineargs_addr_add_option_default}

Si, au lieu de lancer notre application avec ces arguments :

```sh
./app \
-A,//simple-hydro/deltat-cp[1]=3.0 \
-A,//simple-hydro/deltat-cp[2]=6.0 \
-A,//simple-hydro/deltat-cp[3]=7.0 \
dataset.arc
```

on lançait ceci :

```sh
./app \
-A,//simple-hydro/deltat-cp[2]=6.0 \
-A,//simple-hydro/deltat-cp[3]=7.0 \
dataset.arc
```

Dans ce cas, l'option `//simple-hydro/deltat-cp[1]`, qui n'est ni présente dans le jeu de
données, ni présente dans les arguments de la ligne de commande, sera ajouté avec la valeur
par défaut.

### Les syntaxes spéciales {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax}

#### L'indice ANY {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_index_any}

L'indice ANY permet de traiter plusieurs options d'indices différents en une fois.
Il est représenté par des crochets vides : `[]`.

Si l'on reprend l'exemple au-dessus et que l'on souhaite modifier toutes les
valeurs de `deltat-cp`, on peut utiliser l'adresse :

`//simple-hydro/deltat-cp[]`

Exemple de lancement :

```sh
./app \
-A,//simple-hydro/deltat-cp[]=2.0 \
dataset.arc
```

Et dans ce cas, les trois options `simple-hydro/deltat-cp` auraient la valeur `2.0`.

#### La balise ANY {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_tag_any}

La balise ANY permet de changer la valeur d'une option présente à plusieurs endroits.
Elle est représentée par une partie d'adresse vide (deux `/` sans rien entre)
(donc `//module1/option1/option11`, si on veut remplacer `option1` par ANY, on fait `//module1//option11`).

\note La balise ANY ne remplace qu'une seule partie de l'adresse. Pour remplacer
plusieurs parties, il faut en mettre plusieurs. Reprenons l'exemple du dessus en
remplaçant `module1` par ANY : `////option11`

\todo Doit-on garder cette syntaxe ou préférer utiliser `*` ? (exemple : `//*/*/option11`)
Le `*` pouvant prêter à confusion (ça pourrait sous-entendre que l'on pourrait
faire `//*/option*/option11` (comme une regex) alors que non), et la partie d'adresse vide
paraissant plus claire, ce dernier fut choisi.

En reprenant l'exemple au-dessus, si l'on souhaite modifier toutes les options `deltat-cp[2]`,
quelque soit le module dans lequel cette option apparait, on peut utiliser l'adresse :

`///deltat-cp[2]`

Exemple de lancement :

```sh
./app \
-A,///deltat-cp[2]=2.0 \
dataset.arc
```

\warning La balise ANY ne peut pas être présente à la fin de l'adresse
(l'adresse `//module1/option11/` est invalide).


#### Le mélange des deux ANY {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_mix_any}

Admettons que l'on ait le jeu de données :
```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <checkpoint>
      <deltat-cp>3.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <checkpoint>
      <deltat-cp>6.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <final-time>10.2</final-time>

    <post-processor name="Ensight7PostProcessor">
      <fileset-size>10</fileset-size>
      <binary-file>false</binary-file>
    </post-processor>

    <noidea-service name="StillNoIdea">
      <duration>-1</duration>
      <rewrite>false</rewrite>
    </noidea-service>
  </simple-hydro>

  <pas-simple-hydro>
    <checkpoint>
      <deltat-cp>1.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <checkpoint>
      <deltat-cp>3.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <checkpoint>
      <deltat-cp>7.0</deltat-cp>
      <print-details-before-cp>true</print-details-before-cp>
    </checkpoint>

    <final-time>8.9</final-time>

    <post-processor name="Ensight7PostProcessor">
      <fileset-size>10</fileset-size>
      <binary-file>false</binary-file>
    </post-processor>
  </pas-simple-hydro>
  
</case>
```

On pourrait commencer par vouloir modifier l'option `deltat-cp` du second
checkpoint du module `pas-simple-hydro`.
Pour cela, on peut utiliser l'argument : <br>
`-A,//pas-simple-hydro/checkpoint[2]/delta-cp=4.0`.

Puis, on pourrait modifier le premier `deltat-cp` des deux modules : <br>
`-A,///checkpoint[1]/delta-cp=2.0`.

Ensuite, on pourrait ne pas vouloir d'écriture entre les checkpoints : <br>
`-A,///checkpoint[]/print-details-before-cp=false`.

Enfin, les services du module `simple-hydro` doivent agir sur le `Mesh0`
et les services `pas-simple-hydro` sur le `Mesh1` : <br>
`-A,//simple-hydro//@mesh-name=Mesh0` <br>
`-A,//pas-simple-hydro//@mesh-name=Mesh1`

Finalement, on se retrouve avec la commande :

```sh
./app \
-A,//pas-simple-hydro/checkpoint[2]/delta-cp=4.0 \
-A,///checkpoint[1]/delta-cp=2.0 \
-A,///checkpoint[]/print-details-before-cp=false \
-A,//simple-hydro//@mesh-name=Mesh0 \
-A,//pas-simple-hydro//@mesh-name=Mesh1 \
dataset.arc
```

### Les syntaxes invalides {#arcanedoc_execution_commandlineargs_addr_invalid_syntax}

Pour cette partie, quelques containtes d'utilisations ont été mises en places :

- Les arguments commençants par `-A,//` sont réservés à cet usage.
- Les adresses ne peuvent pas terminer par un `/`.
  Exemple invalide :
```sh
./app \
-A,//simple-hydro/deltat-cp/=3.1 \
dataset.arc
```
- Les indices doivent être des entiers supérieurs ou égals à 1.
- Lorsqu'un indice est présent (y compris l'indice ANY), une balise doit aussi être présente.
  Exemple invalide :
```sh
./app \
-A,//simple-hydro/[2]/option=3.1 \
dataset.arc
```



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_traces
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_general_codingrules
</span> -->
</div>

