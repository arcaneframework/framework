# Comment écrire de la documentation {#arcanedoc_doxygen}

[TOC]

## Introduction {#arcanedoc_doxygen_intro}

Lors de la réécriture/restructuration de cette documentation,
certaines conventions ont été établies pour avoir une
doncumentation homogène, simple à lire (à la fois son code source
et son rendu final).

## Structure {#arcanedoc_doxygen_struct}

La documentation se présente ainsi :
(commande `tree` avec quelques coupes)
```sh
framework/arcane/doc/
├── cea_ifpen_logo.png
├── changelog.md
├── chap_core_types
│   ├── 0_core_types.md
│   ├── 1_module.md
│   ├── 2_variable.md
│   ├── ...
│   └── subchap_caseoptions
│       ├── 0_caseoptions.md
│       ├── 1_intro.md
│       ├── 2_struct.md
│       └── ...
├── chap_debug_perf
│   ├── 0_debug_perf.md
│   ├── 1_check_memory.md
│   ├── 2_profiling.md
│   └── ...
├── chap_examples
│   ├── 0_examples.md
│   ├── 1_simple_example.goto
│   ├── 2_concret_example.goto
│   ├── subchap_concret_example
│   │   ├── 0_concret_example.md
│   │   ├── 1_struct.md
│   │   ├── 2_config.md
│   │   ├── ...
│   │   └── img
│   │       ├── QAMA_schema.odg
│   │       └── QAMA_schema.svg
│   └── subchap_simple_example
│       ├── 0_simple_example.md
│       ├── 1_struct.md
│       ├── 2_module.md
│       ├── ...
│       └── img
│           ├── HW_schema.odg
│           ├── HW_schema.svg
│           └── ...
├── chap_getting_started
│   ├── 0_getting_started.md
│   ├── 1_about.md
│   ├── 2_basicstruct.md
│   └── ...
├── how_write_doc.md
├── theme
│   ├── custom.css
│   ├── doxygen-awesome-theme
│   │   └── ...
│   ├── header.html
│   ├── script-num-lines-code.js
│   └── script-resize.js
├── user
│   ├── cleanup_v2.dox
│   └── usermanual.md
└── userdoc.doxyfile
```

Toute la partie thème pour Doxygen se trouve dans le dossier `theme`.
La documentation générale est répartie en chapitre, sous-chapitre, sous-sous-chapitre, &c.

Prenons le chapitre `getting_started` pour l'exemple.
Tout ce chapitre se trouve dans le dossier `chap_getting_started`. Il n'y a pas
de numérotations sur ces dossiers pour éviter d'avoir à tout changer dans
le fichier `userdoc.doxyfile` en cas de changement de position.

Dans un chapitre, nous avons plusieurs fichiers markdown (`.md`).
Le premier doit être un sommaire et nommé `0_nom_du_chapitre.md`.
Ce sommaire permet de lister les `\subpage` pour que Doxygen puisse créer
les liaisons et la structure de la documentation.

Exemple, le fichier `0_getting_started.md` :
```md
# Débuter avec %Arcane {#arcanedoc_getting_started}

Vous débutez sur %Arcane ? Ce chapitre devrait vous donner les notions de bases.

Sommaire de ce chapitre :
1. \subpage arcanedoc_getting_started_about
2. \subpage arcanedoc_getting_started_basicstruct


____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_getting_started_about
</span>
</div>
```

Première ligne, on a le titre de notre chapitre et son tag (pour pouvoir être référencé
dans d'autres pages).
On utilise les `#` pour désigner les titres (comme dans les markdowns).
La première ligne est toujours un titre `h1` (1 seul `#`).
Le tag est entre acolades et débute par un croisillon.
Le nom du tag est structuré comme ceci :
`arcane`\_`nom_du_chapitre`.

Troisième ligne, on a une description sommaire du chapitre.

Les trois lignes suivantes représentent le sommaire du chapitre.
On a une liste numérotée de `\subpage`. 
À coté de `\subpage`, on a le tag d'une page du chapitre.

À la ligne 10, on a une ligne de séparation (voir doc de markdown).

Enfin, on a une partie de html pour faire apparaitre des boutons.
Deux boutons maximum : un bouton `back` et un bouton `next`.
Ici, on n'a qu'un bouton `next`. Entre les `<span>`, on doit mettre
une référence à une page (et pas un `\subpage` !). La partie CSS
s'occupera de faire apparaitre les boutons correctement.

Prenons maintenant le fichier suivant : `1_about.md` (un peu modifié pour
cette explication) :

```md
# Qu'est-ce que %Arcane ? {#arcanedoc_getting_started_about}

[TOC]

## Arcon, Arccore ?  {#arcanedoc_getting_started_about_arcall}
### Arcon  {#arcanedoc_getting_started_about_arcall_arcon}

...Texte...

## Framework ?  {#arcanedoc_getting_started_about_framework}

...Texte...


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_getting_started
</span>
<span class="next_section_button">
\ref arcanedoc_getting_started_basicstruct
</span>
</div>
```

D'abord, la première ligne est semblable à la première ligne
du fichier précédent.
Titre de niveau 1 (1 seul `#`). On commence avec le nom de la page
puis son tag.
Le tag est structuré comme ceci :
`arcane`\_`nom_du_chapitre`\_`nom_de_la_page`.

Ensuite, on retrouve la table des matières (`[TOC]`) qui sera généré
par doxygen si besoin.

Après, on a des titres de niveau 2 et 3.
Le tag est toujours construit de la même manière :

`arcane`\_`nom_du_chapitre`\_`nom_de_la_page`\_`titre_de_la_section_niveau2`\_`titre_de_la_section_niveau3`,...

Pour finir, on retrouve la ligne horizontale et les boutons `back` et `next` avec des `\ref` entre les `<span>`.

____

Enfin, les sous-chapitres. C'est à peu près le même principe.

Dans le chapitre `example`, on a :
```sh
├── chap_examples
│   ├── 0_examples.md
│   ├── 1_simple_example.goto
│   ├── 2_concret_example.goto
│   ├── subchap_concret_example
│   │   ├── 0_concret_example.md
│   │   ├── 1_struct.md
```

Quand on veut faire un sous-chapitre, on créé un dossier `subchap_nom_du_sous_chapitre`
dans le dossier du chapitre parent et on créé un fichier `1_nom_du_sous_chapitre.goto`.

Ce fichier permet de situer le sous-chapitre par rapport aux autres pages (pour nous,
doxygen n'en a pas besoin).

Voyons le fichier `0_concret_example.md` :
```md
# Exemple concret {#arcanedoc_examples_concret_example}


Ce sous-chapitre présente un exemple concret appelé `Quicksilver %Arcane Mini-App`.

Il est recommendé d'avoir lu le sous-chapitre précédent \ref arcanedoc_examples_simple_example
pour bien comprendre ce qui suit, car certaines choses ne seront pas répétées ici.

Sommaire de ce sous-chapitre :
1. \subpage arcanedoc_examples_concret_example_struct


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example_struct
</span>
</div>
```

Les tag seront structurés comme ça :
`arcane`\_`nom_du_chapitre`\_`nom_du_sous_chapitre`\_`nom_de_la_page`\_`titre_de_la_section_niveau2`,...

A part ça, on reprend les règles précédentes.

____

Pour les images, elles doivent être mises avec le (sous-) chapitre correspondant,
dans un dossier `img`. Ce dossier devra être répertorié dans la partie
`IMAGE_PATH` du `userdoc.doxyfile`.

____

Les morceaux de code sont à mettre entre trois "`" (AltGr+7 sur clavier azerty) ou 
entre trois "~" (AtlGr+2 sur clavier azerty).
(Voir doc markdown pour plus d'infos).

____

Une fois votre chapitre ajouté, il faudra ajouter le ou les dossiers des chapitres
et des sous-chapitres dans le userdoc.doxyfile, partie `INPUT`.

____

Deux macros ont été défini : `\arcane{}` et `\arccore{}`.
Elles permettent de lier une classe ou une méthode sans mettre le namespace `Arcane::` ou `Arccore::`
et sans l'afficher.

Exemple :

`Sans namespace : Cell -- Sans macro : Arcane::Cell -- Avec macro : \\arcane{Cell}` 

donne : 

Sans namespace : Cell -- Sans macro : Arcane::Cell -- Avec macro : \arcane{Cell}

---

Dans la partie Changelog, il est possible d'utiliser la macro `\pr{}`
pour rediriger créer un lien vers la pull request sur GitHub. Exemple :

\pr{530}

---

Avec la dernière version du thème "Doxygen Awesome", nous avons la possibilité
d'ajouter des onglets dans la documentation. Exemple :

<div class="tabbed">
 
- <b class="tab-title">Onglet 1</b>
Coucou ! Voici l'onglet n°1 !

- <b class="tab-title">Onglet 2</b>
Recoucou ! Voici l'onglet n°2 !
 
</div>

Voici le code des onglets affichés au-dessus :
```md
<div class="tabbed">
 
- <b class="tab-title">Onglet 1</b>
Coucou ! Voici l'onglet n°1 !

- <b class="tab-title">Onglet 2</b>
Recoucou ! Voici l'onglet n°2 !
 
</div>
```

Doxygen est assez capricieux pour le html. Pour mettre plusieurs lignes
dans un onglet, il faut soit utiliser la balise `<br>`, soit mettre
le tout dans une `<div>`, avec tout le contenu de la div indentée.
Exemple :

<div class="tabbed">
 
- <b class="tab-title">Onglet 1</b>
Coucou ! Voici l'onglet n°1 !<br>
Coucou ! Voici l'onglet n°1 !<br>
Coucou ! Voici l'onglet n°1 !


- <b class="tab-title">Onglet 2</b>
<div>
  Recoucou ! Voici l'onglet n°2 !

  Recoucou ! Voici l'onglet n°2 !



  Recoucou ! Voici l'onglet n°2 !
</div>

</div>

```md
<div class="tabbed">
 
- <b class="tab-title">Onglet 1</b>
Coucou ! Voici l'onglet n°1 !<br>
Coucou ! Voici l'onglet n°1 !<br>
Coucou ! Voici l'onglet n°1 !


- <b class="tab-title">Onglet 2</b>
<div>
  Recoucou ! Voici l'onglet n°2 !

  Recoucou ! Voici l'onglet n°2 !



  Recoucou ! Voici l'onglet n°2 !
</div>

</div>
```

---

Il est possible d'utiliser les balises `<details><summary>` dans la documentation
pour réduire un bout de texte.

Exemple :
```xml
<details>
  <summary>Titre</summary>
  Contenu réduit.
</details>
```
<details>
  <summary>Titre</summary>
  Contenu réduit.
</details>
