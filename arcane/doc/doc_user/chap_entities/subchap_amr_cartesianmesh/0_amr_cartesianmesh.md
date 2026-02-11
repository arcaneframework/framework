# AMR pour les maillages cartésiens {#arcanedoc_entities_amr_cartesianmesh}

[TOC]

Ce sous-chapitre explique la partie AMR de %Arcane pour les maillages cartésiens.

## Introduction {#arcanedoc_entities_amr_cartesianmesh_intro}

Deux types d'AMR sont disponibles dans %Arcane :

- AMR par maille pour les maillages non-structurées (et cartésiens) (amr-type=1),
- AMR par patch pour les maillages cartésiens (amr-type=3).

L'AMR type 1 permet de raffiner une ou plusieurs mailles en fournissant un tableau de localId.

```cpp
mesh()->modifier()->flagCellToRefine(cells_local_id);
mesh()->modifier()->adapt();
```
Pour les maillages cartésiens, plusieurs méthodes sont disponibles pour encapsuler les méthodes de l'AMR non-structuré
et permettant de créer des patchs.

\note Les noeuds des mailles communs entre les niveaux ne sont pas dupliqué en AMR type 1. Ils le sont en AMR type 3.

\image html amr_0.webp

Pour les maillages cartésiens, l'AMR type 3 est aussi disponible et apporte une nouvelle API.
L'utilisateur marque des mailles à raffiner puis %Arcane s'occupe de déterminer des patchs réguliers ainsi que les
mailles de recouvrement.
Le fonctionnement est plus guidé.

Pour utiliser l'AMR type 3, la numérotation des faces doit être changé. Aujourd'hui, le changement est effectué lors du
premier appel à `Arcane::ICartesianMesh::computeDirections()`. De plus, la numérotation des patchs peut être gérée par
la classe `Arcane::CartesianMeshNumberingMng`.

Le choix du type d'AMR à utiliser se fait dans le jeu de données :

```xml

<mesh amr-type="3">
   <meshgenerator>
      <cartesian>
         <nsd>2 2</nsd>
         <origine>0.0 0.0</origine>
         <lx nx='16'>64.0</lx>
         <ly ny='16'>64.0</ly>
      </cartesian>
   </meshgenerator>
</mesh>
```

\note Il n'est pas encore possible de spécifier le type d'AMR avec la nouvelle version du générateur de maillage
cartésien.

<br>

Sommaire de ce sous-chapitre :

1. \subpage arcanedoc_entities_amr_cartesianmesh_notions <br>
   Explication de quelques notions apportées par l'AMR.</br>

2. \subpage arcanedoc_entities_amr_cartesianmesh_working <br>
   Explication du fonctionnement de l'AMR dans %Arcane.</br>



____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_entities_snippet_cartesianmesh
</span>
<span class="next_section_button">
\ref arcanedoc_entities_amr_cartesianmesh_notions
</span>
</div>
