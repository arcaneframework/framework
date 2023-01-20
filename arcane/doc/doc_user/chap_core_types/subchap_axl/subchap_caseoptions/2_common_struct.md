# Attributs et propriétés communs à toutes les options {#arcanedoc_core_types_axl_caseoptions_common_struct}

[TOC]

Quelle que soit l'option, l'élément la définissant doit comporter les
deux attributs suivants :

<table>
<tr>
<th>attribut</th>
<th>occurence</th>
<th>description</th>
<tr>
<td>`name`</td>
<td>requis</td>
<td>nom de l'option. Il doit être composé uniquement de caractères
alphabétiques en minuscules avec le '-' comme séparateur. Ce nom est
utilisé pour générer dans la classe C++ un champ de même nom où les
'-' sont remplacés par des majuscules. Par exemple, une option nommée
`max-epsilon` sera récupérée dans le code par la méthode `maxEpsilon()`,
</td>
</tr>
<tr>
<td>`type`</td>
<td>requis</td>
<td>type de l'option. La valeur et la signification de cet attribut dépendent du type de l'option.</td>
</tr>
<tr>
<td>`default`</td>
<td>optionnel</td>
<td>valeur par défaut de l'option. Si l'option n'est pas présente
dans le jeu de données, %Arcane fera comme si la valeur de cet attribut
avait été entrée par l'utilisateur. L'option vaudra alors la valeur
fournie dans l'attribut \c default. Si cet attribut n'est pas présent,
l'option n'a pas de valeur par défaut et doit toujours être présente
dans le jeu de données.</td>
</tr>
<tr>
<td>`minOccurs`</td>
<td>optionnel</td>
<td>entier qui spécifie le nombre minimum d'occurrences
possible pour l'élément. Si cette valeur vaut zéro, l'option peut être
omise même si l'attribut `default` est absent. Si cet attribut
est absent, le nombre minimum d'occurrence est 1.</td>
</tr>
<tr>
<td>`maxOccurs`</td>
<td>optionnel</td>
<td>entier qui spécifie le nombre maximum d'occurences
possible pour l'élément. Cette valeur doit être supérieure ou égale à `minOccurs`. La valeur spéciale `unbounded` signifie que le
nombre maximum d'occurrences n'est pas limité. Si cet attribut est
absent, le nombre maximum d'occurrence est 1.</td>
</tr>
</table>
  
Pour chaque option, il est possible d'ajouter les éléments fils
suivants :

<table>
<tr>
<th>élément</th>
<th>occurence</th>
<th>description</th>
</tr>
<tr>
<td>`description`</td>
<td>optionnel</td>
<td> qui est utilisé pour décrire l'utilisation de
      l'option. Cette description peut utiliser des éléments HTML. Le contenu
      de cet élément est repris par %Arcane pour la génération de la documentation
      du jeu de données.
</td>
</tr>
<tr>
<td>`userclass`</td>
<td>optionnel</td>
<td> indique la classe d'appartenance de l'option.
      Cette classe spécifie une catégorie d'utilisateur ce qui permet par
      exemple de restreindre certaines options à une certaine catégorie. Par
      défaut, si cet élément est absent, l'option n'est utilisable que
      pour la classe utilisateur. Il est possible de spécifier plusieurs
      fois cet élément avec une catégorie différente à chaque fois. Dans ce
      cas, l'option appartient à toutes les catégories spécifiées.
</td>
</tr>
<tr>
<td>`defaultvalue`</td>
<td>0..infini</td>
<td> permet d'indiquer une valeur par défaut pour une catégorie
donnée. Par exemple :

```xml
<simple name="simple-real" type="real">
  <defaultvalue category="Code1">2.0</defaultvalue>
  <defaultvalue category="Code2">3.0</defaultvalue>
</simple>
```

Dans l'exemple précédent, si la catégorie est 'Code1' alors la
valeur par défaut sera '2.0'. Il est possible de spécifier autant de
catégorie qu'on le souhaite. La catégorie utilisée lors de
l'exécution est positionnée via la méthode
Arcane::ICaseDocument::setDefaultCategory().
</td>
</tr>
<tr>
<td>`name`</td>
<td>0..infini</td>
<td>
permet d'indiquer une traduction pour le nom de
l'option. La valeur de cet élément est le nom traduit pour l'option
et correspondant au langage spécifié par l'attribut <tt>lang</tt>.
Par exemple :

```xml
<simple name="simple-real" type="real">
  <name lang='fr'>reel-simple</name>
</simple>
```

indique que l'option 'simple-real' s'appelle en francais 'reel-simple'.
Plusieurs éléments <tt>name</tt> sont possibles, chacun spécifiant une
traduction. Le jeu de données devra être fourni dans la langue par défaut,
le français dans notre cas. Si aucune traduction n'est donnée, c'est
la valeur de l'attribut \c name qui est utilisée.
</td>
</tr>
</table>


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_struct
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_options
</span>
</div>
