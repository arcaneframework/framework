# Structures et types de base {#arcanedoc_getting_started_basicstruct}

[TOC]

## Types de bases {#arcanedoc_getting_started_basicstruct_types}

%Arcane fournit un ensemble de types de base, correspondant soit à un
type existant du C++ (comme *int*, *double*), soit à une classe (comme
 \arcane{Real2}). Ces types sont utilisés pour toutes les opérations courantes
mais aussi pour les variables. Par exemple, lorsqu'on souhaite
déclarer un entier, il faut utiliser \arccore{Integer} au lieu de
*int* ou *long*. Cela permet de modifier la taille de ces types
(par exemple, utiliser des entiers sur 8 octets au lieu de 4)
sans modifier le code source.

Les types de bases sont :

<table>
<tr><td><b>Nom de la classe</b></td><td><b>Correspondance dans les spécifications</b></td></tr>
<tr><td>\arccore{Integer}   </td><td> entier signé sur 32 bits </td></tr>
<tr><td>\arccore{Int16}     </td><td> entier signé sur 16 bits </td></tr>
<tr><td>\arccore{Int32}     </td><td> entier signé sur 32 bits </td></tr>
<tr><td>\arccore{Int64}     </td><td> entier signé sur 64 bits </td></tr>
<tr><td>\arcane{Byte}       </td><td> représente un caractère sur 8 bits </td></tr>
<tr><td>\arccore{Real}      </td><td> réel IEEE 754 </td></tr>
<tr><td>\arcane{Real2}      </td><td> coordonnée 2D, vecteur de deux réels </td></tr>
<tr><td>\arcane{Real3}      </td><td> coordonnée 3D, vecteur de trois réels </td></tr>
<tr><td>\arcane{Real2x2}    </td><td> tenseur 2D, vecteur de quatre réels </td></tr>
<tr><td>\arcane{Real3x3}    </td><td> tenseur 3D, vecteur de neufs réels </td></tr>
<tr><td>\arccore{String}    </td><td> chaîne de caractères au format UTF-8 </td></tr>
</table>

Les flottants (\arccore{Real}, \arcane{Real2}, \arcane{Real2x2},
\arcane{Real3}, \arcane{Real3x3}) utilisent des réels double précision de la norme IEEE
754 et sont stockés sur 8 octets.

## Entités du maillage {#arcanedoc_getting_started_basicstruct_meshitem}
Il existe 4 types d'entités de base dans un maillage : les noeuds, les
arêtes, les faces et les mailles. \`A chacun de ces types correspond une
classe C++ dans %Arcane. Pour chaque type d'entité, il existe un type
*groupe* qui gère un ensemble d'entités de ce type. La classe qui gère
un groupe d'une entité a pour nom celui de l'entité suffixée par
*Group*. Par exemple, pour les noeuds, il s'agit de \arcane{NodeGroup}.

<table>
<tr><td><b>Nom de la classe</b></td><td><b>Correspondance dans les spécifications</b></td></tr>
<tr><td>\arcane{Node}      </td><td> un noeud </td></tr>
<tr><td>\arcane{Cell}      </td><td> une maille </td></tr>
<tr><td>\arcane{Face}      </td><td> une face en 3D, une arête en 2D</td></tr>
<tr><td>\arcane{Edge}      </td><td> une arête en 3D</td></tr>
<tr><td>\arcane{Particle}  </td><td> une particule</td></tr>
<tr><td>\arcane{DoF}       </td><td> un degré de liberté</td></tr>
<tr><td>\arcane{NodeGroup} </td><td> un groupe de noeuds </td></tr>
<tr><td>\arcane{CellGroup} </td><td> un groupe de mailles </td></tr>
<tr><td>\arcane{FaceGroup} </td><td> un groupe de faces </td></tr>
<tr><td>\arcane{EdgeGroup} </td><td> un groupe d'arêtes </td></tr>
<tr><td>\arcane{ParticleGroup} </td><td> un groupe de particules</td></tr>
<tr><td>\arcane{DoFGroup} </td><td> un groupe de degrés de liberté</td></tr>
</table>

\note
Les faces \arcane{Face} sont les entités de dimension N-1 avec N la dimension
des mailles. En dimension 2, les faces correspondent donc aux arêtes et en dimension 3 aux
faces des polyèdres. Ceci permet aux algorithmes numériques de parcourir le maillage
indépendamment de sa dimension. L'entité arête (\arcane{Edge}) n'existe que
pour les maillages 3D et correspond alors à une arête.

Chaque entité du maillage correspond à une instance d'une classe. Par
exemple, si le maillage contient 15 mailles, il y a 15 instances du
type \arcane{Cell}. Chaque classe fournit un certain nombre d'opérations
permettant de relier les instances entre elles. Par exemple, la méthode
\arcane{Cell::node}(\arccore{Int32}) de la classe \arcane{Cell} permet de récupérer le
\a i-ème noeud de cette maille. De même, la méthode \arcane{Cell::nbNode()} permet de
récupérer le nombre de noeuds de la maille. Pour plus de
renseignements sur les opérations supportées, il est nécessaire de se
reporter à la documentation en ligne des classes correspondantes
(\arcane{Node}, \arcane{Edge}, \arcane{Face}, \arcane{Cell}, \arcane{Particle}, \arcane{DoF}).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_getting_started_about
</span>
<span class="next_section_button">
\ref arcanedoc_getting_started_iteration
</span>
</div>
