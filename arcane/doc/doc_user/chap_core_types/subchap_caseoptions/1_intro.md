# Introduction {#arcanedoc_core_types_caseoptions_intro}

[TOC]

Ce chapître décrit les options possibles pour le fichier *axl*. Ces options
s'appliquent de manière identiques aux modules et aux services. Afin
d'éviter des répétitions inutiles, on utilisera le terme module
seulement, en sachant que cela s'applique aussi aux services.

Chaque module possède des options qui peuvent être spécifiées
par l'utilisateur lors du lancement d'une exécution. Ces options sont en
général dictées par le *jeu de données* que fournit
l'utilisateur pour lancer son cas. Le document \ref arcanedoc_core_types_module 
montre que chaque module possède un fichier de configuration nommé
*descripteur de module* composé de 3 parties : 
les variables, les points d'entrée et les options de configuration.
Le présent document s'intéresse à la la partie concernant les
options de configuration qui vont permettre de définir la grammaire
du jeu de données du module.

Le descripteur de module est un fichier au format XML. Ce fichier 
est utilisé par %Arcane pour générer des classes C++. Une de ces
classes se charge de la lecture des informations dans le jeu
de données.

Par convention, pour un module appelé *Test*, le descripteur
de module s'appelle *Test.axl*. Ce fichier `axl` permet de générer un fichier
*Test_axl.h*. Ce fichier sera inclu par la classe implémentant le module.

Dans le jeu de données, les options d'un module sont fournies
dans l'élément de nom `<options>`. Par exemple, les options
du module *Test* sont :

```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <description>Module Test</description>
  <options>
    <!-- contient les options du module Test -->
    ...
  </options>
</module>
```


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_caseoptions
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_caseoptions_struct
</span>
</div>