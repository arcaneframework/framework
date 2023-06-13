# Tableaux {#arcanedoc_core_types_array_usage}

[TOC]

## Types tableaux {#arcanedoc_core_types_array_usage_type}

\note Même si on fait référence à %Arcane dans cette partie,
les classes gérant les tableaux et les vues sont dans %Arccore et
sont donc définies dans le namespace Arccore. Il y a néanmoins
un `using` dans %Arcane pour ces classes ce qui permet de les
utiliser comme étant dans le namespace Arcane.

L'utilisation des tableaux dans %Arcane utilise deux types de
classes : les \a conteneurs et les \a vues:

- Les conteneurs tableaux permettent de stocker des éléments et gèrent la
  mémoire pour ce stockage à la manière de `std::vector`. Ils possèdent
  des opérations permettant d'ajouter ou supprimer des éléments. La
  mémoire nécessair est automatiquement gérée en cas d'ajout d'éléments
  par exemple.
- Les vues représentent un sous-ensemble d'un conteneur et sont des
  objets <strong>temporaires</strong> : les vues ne doivent pas être
  conservées entre deux évolutions du nombre d'éléments du conteneur
  associé.

Les classes gérant les conteneurs ont un nom qui finit par
\a %Array (par exemple \arccore{UniqueArray} ou \arccore{SharedArray}.
Les conteneurs ont les caractéristiques suivantes :

- ils gèrent la mémoire nécessaire pour conserver leurs éléments.
- les éléments sont conservés de manière contigüe en mémoire. Il est
  donc possible d'utiliser ces conteneurs à des fonctions en langage C par
  exemple qui prennent en argument des pointeurs. 

Dans %Arcane, il existe deux types de classes pour gérer les vues :
- les classes dont le nom finit par \a View (\arccore{ArrayView},
  \arccore{ConstArrayView}. Il s'agit des classes
  utilisées historiquement dans %Arcane. Pour ces classes le nombre
  d'éléments est conservé dans un \arccore{Int32}.
- les classes dont le nom finit par \a Span (\arccore{Span}, \arccore{SmallSpan}). Ces
  classes ont été ajoutées à partir de 2018 et sont similaires à la
  classe `std::span` (https://en.cppreference.com/w/cpp/container/span)
  du C++20. Pour ces classes le nombre d'éléments
  est conservé dans un \arccore{Int64} pour \arccore{Span} et dans un
  \arccore{Int32} pour \arccore{SmallSpan}.

La différence majeure entre les vues historiques et \arccore{Span} est qu'il
n'y a qu'une seule classe pour gérer les éléments constant ou non constant
et que les opérateurs d'accès (\arccore{Span::operator[]()}) sont `const`
pour les `Span` ce qui permet de les utiliser dans les lambda.
Il est donc préférable d'utiliser \arccore{Span} ou \arccore{SmallSpan}
à la place de \arccore{ArrayView} ou \arccore{ConstArrayView}.

Les vues ont les caractéristiques suivantes :
- elles ne gèrent aucune mémoire et sont toutes issues d'un
  conteneur (qui n'est pas nécessairement une classe %Arcane)
- elles ne sont valides que tant que le conteneur associé existe et
  le nombre de ces éléments n'est pas modifié.
- elles s'utilisent en général par valeur plutôt que par référence (on
  ne leur applique pas l'opérateur &).
- leur taille est petite (en général 16 octets) et on peut donc les
  conserver et les copier facilement.

\warning Pour des raisons de performance, les classes tableaux ne gèrent par
l'initialisation des éléments de la même manière si le type est considéré
comme un type POD (Plain Object Data) poure %Arcane.
La macro ARCCORE_DEFINE_ARRAY_PODTYPE(type) permet
d'indiquer que \a type est un type POD pour \arccore{Array}. L'utilisation
de cette macro doit se faire avant la définition d'une instance de tableau
pour le type \a type. Tout les types de base du C++ (`char`, `int`, `double`, ...)
sont considérés comme des types POD pour %Arcane.

Le tableau suivant donne la liste des classes gérant les tableaux et
les vues associées :

<table>
<tr>
<th>Description</th>
<th>Classe de base</th>
<th>Sémantique par référence</th>
<th>Sémantique par valeur</th>
<th>Vue modifiable</th>
<th>Vue constante</th>
</tr>
<tr>
<td>Tableau 1D</td>
<td>\arccore{Array}</td>
<td>\arccore{SharedArray}</td>
<td>\arccore{UniqueArray}</td>
<td>\arccore{ArrayView} <br/> \arccore{Span<T>} <br/> \arccore{SmallSpan<T>}</td>
<td>\arccore{ConstArrayView} <br/> \arccore{Span<const T>} <br/> \arccore{SmallSpan<const T>}</td></tr>
<tr>
<td>Tableau 2D classique</td>
<td>\arccore{Array2}</td>
<td>\arccore{SharedArray2}</td>
<td>\arccore{UniqueArray2}</td>
<td>\arccore{Array2View} <br/> \arccore{Span2<T>} <br/> \arccore{SmallSpan2<T>}</td>
<td>\arccore{ConstArray2View} <br/> \arccore{Span2<const T>} <br/> \arccore{SmallSpan2<const T>}</td>
</tr>
<tr>
<td>Tableau 2D avec 2-ème dimension variable</td>
<td>\arcane{MultiArray2}</td>
<td>\arcane{SharedMultiArray2}</td>
<td>\arcane{UniqueMultiArray2}</td>
<td>\arcane{MultiArray2View}</td>
<td>\arcane{ConstMultiArray2View}</td>
</tr>
</table>

Pour chaque type de tableau, il existe une classe de base dont hérite
une implémentation avec sémantique par référence et une
implémentation avec sémantique par valeur. La classe de base n'est
ni copiable ni affectable. La différence de
sémantique concerne le fonctionnement des opérateurs de recopie et
d'affectation :
- la sémantique par référence signifie que lorsqu'on fait <em>a =
b</em>, alors \a a devient une référence sur \a b et toute modification de \a b modifie
aussi \a a.

```cpp
Arcane::SharedArray<int> a1(5);
Arcane::SharedArray<int> a2;
a2 = a1; // a2 et a1 font référence à la même zone mémoire.
a1[3] = 1;
a2[3] = 2;
std::cout << a1[3]; // affiche '2'
```

- la sémantique par valeur signifie que lorsqu'on fait <em>a =
b</em>, alors \a a devient une copie des valeurs de \a b et par la suite les
tableaux \a a et \a b sont indépendants.

```cpp
Arcane::UniqueArray<int> a1(5);
Arcane::UniqueArray<int> a2;
a2 = a1; // a2 devient une copie de a1.
a1[3] = 1;
a2[3] = 2;
std::cout << a1[3]; // affiche '1'
```

## Passage de tableaux en arguments {#arcanedoc_core_types_array_usage_argument}

Voici les règles de bonnes pratiques à respecter pour le passage de tableaux en argument :

<table>

<tr>
<th>Argument</th>
<th>Besoin</th>
<th>Opérations possibles</th>
</tr>
<tr>
<td>\arccore{ConstArrayView} <br/> \arccore{Span<const T>} <br/> \arccore{SmallSpan<const T>}</td>
<td>Tableau 1D en lecture seule</td>
<td>

```cpp
x = a[i];
```

</td>
</tr>
<tr>
<td>\arccore{ArrayView} <br/> \arccore{Span<T>} <br/> \arccore{SmallSpan<T>}</td>
<td>Tableau 1D en lecture et/ou écriture mais dont la taille n'est
pas modifiable</td> 
<td>

```cpp
x = a[i];
a[i] = y;
```

</td>
</tr>
<tr>
<td>\arccore{Array}&</td>
<td>Tableau 1D modifiable et pouvant changer de nombre d'éléments</td>
<td>

```cpp
x = a[i];
a[i] = y;
a.resize(u);
a.add(v);
```

</td>
</tr>
<tr>
<td>const \arccore{Array}&</td>
<td>Interdit. Utiliser \arccore{ConstArrayView} ou \arccore{Span<const T>} à la place</td>
<td></td>
</tr>
<tr>
<td>\arccore{ConstArray2View} <br/> \arccore{Span2<const T>} <br/> \arccore{SmallSpan2<const T>}</td>
<td>Tableau 2D en lecture seule</td>
<td>

```cpp
x = a[i][j];
```

</td>
</tr>
<tr>
<td>\arccore{Array2View} <br/> \arccore{Span2<T>} <br/> \arccore{SmallSpan2<T>}</td>
<td>Tableau 2D en lecture et/ou écriture mais dont la taille n'est pas modifiable</td>
<td>

```cpp
x = a[i][j];
a[i][j] = y;
```

</td>
</tr>
<tr>
<td>\arccore{Array}&</td>
<td>Tableau 2D modifiable et pouvant changer de nombre d'éléments</td>
<td>

```cpp
x = a[i][j];
a[i][j] = y;
a.resize(u,v);
```

</td>
</tr>
<tr>
<td>const \arccore{Array2}&</td>
<td>Interdit. Utiliser \arccore{ConstArray2View} ou \arccore{Span2<const T>} à la place</td>
<td></td>
</tr>
</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_timeloop
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_numarray
</span>
</div>
