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

Si on met ce symbole dans un jeu de données, on aurait, par exemple :

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
mais présente donc n'est pas remplacé par la valeur par défaut.
Dans le second cas, la valeur de l'option est **null** (`String()`) donc est remplacé
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

Il est nécessaire d'avoir l'adresse de l'option. Pour la retrouver, il faut
dérouler les éléments XML. Ici, l'option `deltat-cp` est à l'adresse :
`//case/simple-hydro/deltat-cp`.
Ensuite, on retire le `case/` au début (ou `cas/` pour les jeux de données en français).
Enfin, on peut construire l'argument à ajouter : 

`-A,``//simple-hydro/deltat-cp``=3.1`

Comme pour le remplacement de symboles, on peut ajouter des guillemets :

`-A,//simple-hydro/simple-real-array="3.1 3.11 3.12"`

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
à 1, et non 0 !

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

Et dans ce cas, les trois options `simple-hydro/deltat-cp` auront la valeur `2.0`.


#### Le tag ANY {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_tag_any}

Le tag ANY permet de traiter plusieurs options ayant des adresses différentes en une fois.
Il est représenté par une partie d'adresse vide (donc `//module1/option1/option11`,
si on veut remplacer `option1` par ANY, on fait `//module1//option11`).

\warning Le tag ANY ne remplace qu'une seule partie de l'adresse. Pour remplacer
plusieurs parties, il faut en mettre plusieurs. Reprenons l'exemple du dessus en
remplaçant `module1` par ANY : `////option11`

En reprenant l'exemple au-dessus, si l'on souhaite modifier toutes les options `deltat-cp[2]`,
on peut utiliser l'adresse :

`///deltat-cp[2]`

Exemple de lancement :

```sh
./app \
-A,///deltat-cp[2]=2.0 \
dataset.arc
```

\warning Le tag ANY ne peut pas être présent à la fin de l'adresse.


#### Le mélange des deux ANY {#arcanedoc_execution_commandlineargs_addr_multi_special_syntax_mix_any}

Admettons que l'on ait le jeu de données :
```xml
<!--Fichier ARC-->
<?xml version="1.0"?>
<case codename="ArcaneTest" codeversion="1.0">
  <arcane/>
  <meshes/>
  
  <simple-hydro>
    <deltat-cp>3.0</deltat-cp>
    <deltat-cp>6.0</deltat-cp>
    <final-time>10.2</final-time>
    <post-processor name="Ensight7PostProcessor">
      <fileset-size>10</fileset-size>
      <binary-file>false</binary-file>
    </post-processor>
  </simple-hydro>

  <pas-simple-hydro>
    <deltat-cp>1.0</deltat-cp>
    <deltat-cp>3.0</deltat-cp>
    <final-time>9.9</final-time>
    <post-processor name="Ensight7PostProcessor">
      <fileset-size>10</fileset-size>
      <binary-file>false</binary-file>
    </post-processor>
  </pas-simple-hydro>
  
</case>
```

TODO Continuer ici


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




____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_execution_traces
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_general_codingrules
</span> -->
</div>

