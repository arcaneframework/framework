# Utilisation de la classe NumArray {#arcanedoc_core_types_numarray}

[TOC]

La classe Arcane::NumArray permet de gérer des tableaux à plusieurs
dimensions de valeurs numériques. La version actuelle de %Arcane gère
des tableaux jusqu'à la dimension 4. Le nombre de dimensions du
tableau est aussi appelé le rang (Arcane::NumArray::rank()) du tableau.

Cette classe est similaire à la classe `std::mdarray` prévue pour le
C++26: https://isocpp.org/files/papers/D1684R0.html.

La sémantique est une sémantique par valeur (comme `std::vector`) et
donc les opérateurs d'affectation provoquent une recopie des valeurs
du tableau.

Le prototype est le suivant :

~~~{cpp}
template<typename DataType,typename Extents,typename LayoutPolicy>
class NumArray;
~~~

Avec:
- \a DataType: le type de donnée du tableau. Il s'agit obligatoirement
  d'un type numérique (`std::is_arithmetic<DataType>==true`) qui doit
  être copiable trivialement
  (`std::is_trivially_copyable<DataType>==true`)
- \a Extents: indique le nombre d'éléments (extent()) de chaque
  dimension. La valeur peut être dynamique (Arcane::DynExtent) ou
  statique si une valeur positive est utilisée.
- \a LayoutPolicy: indique la politique d'agencement. Actuellement
  deux valeurs sont possibles : Arcane::RightLayout ou
  Arcane::LeftLayout. La valeur par défaut est Arcane::RightLayout qui
  correspond à l'agencement classique d'un tableau C multidimensionnel.

## Création

Les types Arcane::MDDim1, Arcane::MDDim2, Arcane::MDDim3,
Arcane::MDDim4, permettent de spécifier des instances dont toutes les
dimensions sont dynamiques. Par exemple :

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarations

Si on souhaite spécifier une ou plusieurs dimensions statiques, on
peut faire comme cela :

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarationsExtented

\note Les valeurs de l'instance ne sont pas initialisées lors de la
construction. Il faut appeler la méthode Arcane::NumArray::fill()
(uniquement si la mémoire est accessible depuis l'hôte) si
on souhaite remplir le tableau avec une valeur donnée.

Il est possible de spécifier lors de la construction ou avec le
méthode Arcane::NumArray::resize() le nombre d'éléments de chaque
dimension. Dans ce cas le nombre d'arguments correspond au nombre de
dimensions dynamiques de l'instance :

\snippet NumArrayUnitTest.cc SampleNumArrayResize

\warning Le redimensionnement ne conserve pas les valeurs actuelles du tableau

## Gestion mémoire

Le type Arcane::eMemoryRessource permet de spécifier dans quel espace
mémoire le tableau sera alloué. Par défaut, on utilise
Arcane::eMemoryRessource::UnifiedMemory ce qui permet au tableau
d'être accessible à la fois sur l'hôte et l'accélérateur. Il est
possible de spécifier à la construction la ressource mémoire
associée. Si on utilise la zone mémoire
Arcane::eMemoryRessource::Device alors les données seront uniquement
accessibles sur accélérateur et il ne faudra pas tenter d'accéder aux
valeurs du tableau (que ce soit en lecture ou en écriture) depuis
l'hôte.

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarationsMemory

## Indexation

L'indexation des valeurs de Arcane::NumArray se fait via l'opérateur
Arcane::NumArray::operator(). On peut soit utiliser une instance de
Arcane::ArrayIndex (`Arcane::ArrayIndex<N>` avec `N` le rang du
tableau), soit utiliser une surcharge qui prend `N` valeurs en argument.

Pour chaque dimension, la valeur de l'index commence à zéro. Les
valeurs valides vont donc de `[0,extentP()[` avec `P` la `P-ème`
dimension.

Par exemple :

\snippet NumArrayUnitTest.cc SampleNumArrayDeclarationsIndexation
