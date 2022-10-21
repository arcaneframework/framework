# Tableaux {#arcanedoc_core_types_array_usage}

[TOC]

## Types tableaux {#arcanedoc_core_types_array_usage_type}

L'utilisation des tableaux dans %Arcane utilise deux types de
classes: les \a conteneurs et les \a vues.

Les conteneurs tableaux permettent de
stocker des éléments et gèrent la
mémoire pour ce stockage. Les vues représentent un sous-ensemble
d'un conteneur et sont des objets <strong>temporaires</strong>.

Les classes gérant les conteneurs ont un nom qui finit par \a
%Array. Les conteneurs ont les caractéristiques suivantes:
- ils gèrent la mémoire nécessaire pour conserver leurs éléments.
- les éléments sont conservés de manière contigüe en mémoire. Il est
donc possible d'utiliser ces conteneurs à des fonctions en langage C par
exemple qui prennent en argument des pointeurs. 

Les classes gérant les vues ont un nom qui finit par \a View, comme ArrayView
ou ConstArrayView. Les vues ont les caractéristiques suivantes:
- elles ne gèrent aucune mémoire et sont toutes issues d'un
conteneur (qui n'est pas nécessairement une classe Arcane)
- elles ne sont valide que tant que le conteneur associé existe et
n'est pas modifié.
- elles s'utilisent toujours par valeur et jamais par référence (on
ne leur applique pas l'opérateur &).

En général, il ne faut donc pas utiliser de vue comme champ d'une classe.

Le tableau suivant donne la liste des classes gérant les tableaux et
les vues associées:

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
<td>\arccore{ArrayView}</td>
<td>\arccore{ConstArrayView}</td></tr>
<tr>
<td>Tableau 2D classique</td>
<td>\arccore{Array2}</td>
<td>\arccore{SharedArray2}</td>
<td>\arccore{UniqueArray2}</td>
<td>\arccore{Array2View}</td>
<td>\arccore{ConstArray2View}</td>
</tr>
<tr>
<td>Tableau 2D avec 2ème dimension variable</td>
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
d'affectation:
- la sémantique par référence signifie que lorsqu'on fait <em>a =
b</em>, alors \a a devient une référence sur \a b et toute modification de \a b modifie
aussi \a a.

```cpp
SharedArray<int> a1(5);
SharedArray<int> a2;
a2 = a1; // a2 et a1 font référence à la même zone mémoire.
a1[3] = 1;
a2[3] = 2;
std::cout << a1[3]; // affiche '2'
```

- la sémantique par valeur signifie que lorsqu'on fait <em>a =
b</em>, alors \a a devient une copie des valeurs de \a b et par la suite les
tableaux \a a et \a b sont indépendants.

```cpp
UniqueArray<int> a1(5);
UniqueArray<int> a2;
a2 = a1; // a2 devient une copie de a1.
a1[3] = 1;
a2[3] = 2;
std::cout << a1[3]; // affiche '1'
```

## Passage de tableaux en arguments {#arcanedoc_core_types_array_usage_argument}

Voici les règles de bonnes pratiques à respecter pour le passage de tableaux en argument:

<table>

<tr>
<th>Argument</th>
<th>Besoin</th>
<th>Opérations possibles</th>
</tr>
<tr>
<td>ConstArrayView</td>
<td>Tableau 1D en lecture seule</td>
<td>

```cpp
x = a[i];
```

</td>
</tr>
<tr>
<td>ArrayView</td>
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
<td>Array&</td>
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
<td>const Array&</td>
<td>Interdit. Utiliser ConstArrayView à la place</td>
<td></td>
</tr>
<tr>
<td>ConstArray2View</td>
<td>Tableau 2D en lecture seule</td>
<td>

```cpp
x = a[i][j];
```

</td>
</tr>
<tr>
<td>Array2View</td>
<td>Tableau 2D en lecture et/ou écriture mais dont la taille n'est pas modifiable</td>
<td>

```cpp
x = a[i][j];
a[i][j] = y;
```

</td>
</tr>
<tr>
<td>Array2&</td>
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
<td>const Array&</td>
<td>Interdit. Utiliser ConstArrayView à la place</td>
<td></td>
</tr>
</table>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_timeloop
</span>
<!-- <span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span> -->
</div>
