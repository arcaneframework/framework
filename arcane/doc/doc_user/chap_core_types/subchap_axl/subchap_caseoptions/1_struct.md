# Structure du fichier {#arcanedoc_core_types_axl_caseoptions_struct}

[TOC]

Le descripteur de module est au format XML. Nous allons nous
intéresser à la partie configuration des options contenue dans 
l'élément \c options de ce fichier. En voici un exemple :

```xml
<options>
  <simple name = "simple-real" type = "real">
    <name lang='fr'>reel-simple</name>
    <description>Réel simple</description>
  </simple>
</options>
```

Cet exemple définit une option de configuration appelée
*simple-real*. Cette option est une variable simple de type
`real` sans valeur par défaut.

La structure de tout élément de configuration des options d'un
module est similaire à celle-ci. Toutes les options possibles doivent
apparaître dans des éléments fils de \c options.

Les différentes possibilités sont les suivantes :
- les options simples, de type `real`, `bool`,
  `integer` ou `string`.
- les options énumérées, qui doivent correspondre à un type
  `enum` du C++.
- les options de types dit étendus. Il s'agit de types créés
  par l'utilisateur (classes, structures...).  Cela comprend 
  par exemple les groupes d'entités du maillage.
- les options complexes, qui sont composées elles-mêmes d'options.
  Les options complexes peuvent s'imbriquer.
- les options services, qui permettent de référencer un service (voir le document \ref arcanedoc_core_types_service).


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_common_struct
</span>
</div>