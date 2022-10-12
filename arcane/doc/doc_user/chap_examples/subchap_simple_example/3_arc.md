# Jeu de données {#arcanedoc_examples_simple_example_arc}

[TOC]

Notre application `HelloWorld` a besoin d'un jeu de données pour fonctionner. 
Voici donc le jeu de données généré par `arcane_template`, agrémenté d'une `option`.
\note
Les fichiers `.arc` ne sont pas compilés avec l'application. Ils peuvent
donc être modifiés après coup. De plus, on peut en avoir plusieurs pour
faire varier la simulation (c'est un peu l'objectif d'ailleurs).

## HelloWorld.arc {#arcanedoc_examples_simple_example_arc_helloworldarc}
```xml
<?xml version="1.0"?>
<case codename="HelloWorld" xml:lang="en" codeversion="1.0">

  <arcane>
    <title>3steps</title>
    <timeloop>HelloWorldLoop</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>
        <origin>0.0 0.0</origin>
        <x><n>20</n><length>2.0</length></x>
        <y><n>20</n><length>2.0</length></y>
      </generator>
    </mesh>
  </meshes>

  <say-hello>
    <nSteps>3</nSteps>
  </say-hello>

</case>
```
Plusieurs choses sont à noter ici.
D'abord, on retrouve le nom de notre application à la seconde ligne.

Ensuite :
```xml
<arcane>
  <title>3steps</title>
  <timeloop>HelloWorldLoop</timeloop>
</arcane>
```
On a le titre de notre jeu de données. Il doit décrire le jeu de données.
Ce jeu de données permet de faire 3 tours de boucle donc `3steps`.

Puis, on a `<timeloop>HelloWorldLoop</timeloop>`. C'est le nom de la boucle en
temps de notre programme. Ce nom doit être identique à celui donné
dans le fichier `.config` (voir prochaine section \ref arcanedoc_examples_simple_example_config).

____

```xml
<meshes>
  <mesh>
    <generator name="Cartesian2D" >
      <x><n>20</n><length>2.0</length></x>
      <y><n>20</n><length>2.0</length></y>
      <origin>0.0 0.0</origin>
      <nb-part-x>1</nb-part-x> 
      <nb-part-y>1</nb-part-y>
    </generator>
  </mesh>
</meshes>
```
Cette partie permet d'appeler le service `Cartesian2DMeshGenerator` (documentation développeur uniquement) 
qui permet de créer un maillage cartesien 2D. Dans cet exemple, on génère un maillage de 20x20 mailles, de
taille totale de 2.0x2.0 (donc chaque maille fait une taille de (0.5, 0.5)) avec une origine à (0.0, 0.0)
et un découpage en sous-domaine de 1x1 (donc il y a 1x1=1 sous-domaine).

____

```xml
<say-hello>
  <nSteps>3</nSteps>
</say-hello>
```
Cette partie concerne uniquement le module `SayHello`.
En effet, chaque module peut avoir des options enregistrées dans le jeu de données.
Le nom du module est transformé en low-case et les majuscules sont précédées d'un tiret.
\remarks
Plusieurs exemples pour comprendre :
```log
SayHello -> say-hello
SAYHello -> s-a-y-hello
sayhello -> sayhello
```

Au milieu, on retrouve notre option `<nSteps>3</nSteps>` que l'on a vu
lors de la section précédente. On lui donne la valeur 3, ce qui fait
que l'instruction `options()->getNStep()` renverra la valeur 3.

\remarks
Pour rappel, dans le `.axl` du module `SayHello`, on avait ça :
```xml
<options>
  <simple name="nSteps" type="integer" default="10">
    <description>Nombre de boucle à effectuer.</description>
  </simple>
</options>
```
On voit bien le lien entre `.axl` et `.arc`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_module
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_config
</span>
</div>