# Principes du format ARCT {#arcanetdoc_arct_basis}

[TOC]

\note Le fichier de spécification est disponible dans le dépôt framework/arctools/arcanet/arct. Cette page résume ces spécifications.

Un fichier ARCT décrit un ou plusieurs cas d'utilisation, pour un projet donné.

Pour obtenir le résultat souhaité, il est possible que plusieurs programmes doivent être lancés à la chaine.
Ce fichier permet de stocker des paramètres pour chaque programme.
Par contre, ce fichier ne stocke pas l'ordre d'exécution de ces programmes.

Pour Arcane par exemple (ou pour les programmes utilisant Arcane), il est possible de lui donner en paramètre le chemin d'un fichier ARCT et le nom d'un cas
d'utilisation pour qu'Arcane l'exécute.

Évidemment, tous les programmes ne supporteront pas ce format de fichier. Pour ceux-la,
le lanceur ArcCher s'occupera de lire le fichier ARCT et de leur passer les bons paramètres, de manière a ce que ce soit transparent pour l'utilisateur.

Le format de fichier ARCT a été conçu initialement pour remplacer les fichiers de variations existants dans certains codes utilisant Arcane.

Un fichier de variations décrit une ou plusieurs variations de propriétés d'un jeu de données Arcane (.arc).

```xml
<?xml version='1.0' ?>
<variations>
  <variation>
    TODO
  </variation>
</variations>
```

Avec un programme intermédiaire, il est possible de générer un ou plusieurs fichiers de jeu de données, un par variation possible.

```txt
[dataset.arc] + [auto_variations.xml]
->
- [variant1/dataset.arc]
- [variant2/dataset.arc]
- ...
```

L'approche adoptée par le format ARCT est différente.

Au lieu de générer des variations de jeux de données, on a cherché à générer des variations d'exécutions.


## Briques de base

Un fichier ARCT possède cette structure :
```json5
{
  "versions": {
    "_": 1,
    "arcane": 1,
    "arccher": 1
  },
  "general": {
    // <config_block>
  },
  "cases": {
    "cas_1": {
      // <config_block>
    }
  },
  "commons": {
    "common_config_1": {
      // <config_block>
    }
  },
  "variations": {
    "variation_1": {
      "variant_1": {
        // <config_block>
      }
    }
  }
}
```

Un bloc de configurations (`<config_block>`) est un objet JSON permettant d'apporter un bout de configuration à un ou plusieurs programmes.

Un bloc de configurations est lui-même composé de blocs réservés aux programmes (qui sont aussi des objets JSON).
Par exemple :

```json5
{
  "_": {
    // <reserved_block>
  },
  "arcane": {
    // <reserved_block>
  },
  "arccher": {
    // <reserved_block>
  }
}
```

Le bloc `arcane` est réservé à Arcane. C'est à Arcane de définir son contenu et de le versionner. Ça ne dépend pas du format ARCT.

Un programme demandeur est un programme qui va prendre un fichier ARCT en argument et qui va utiliser un lecteur de fichier ARCT pour
obtenir la configuration qu'il va devoir appliquer.

Le lecteur de fichier ARCT va fournir une ou plusieurs listes de blocs de configurations au programme demandeur.

Le programme demandeur devra dire au lecteur s'il supporte le multi-cas (lancement de plusieurs cas en une fois) ou non.
S'il ne le supporte pas, alors une seule liste de bloc sera fourni par le lecteur (sinon, erreur).

```txt
[lecteur de fichier ARCT] + [fichier.arct]
->
[
  [<config_block_1>, <config_block_2>, ...],
  [<config_block_3>, <config_block_2>, ...],
  ...
]
```

Le programme demandeur devra appliquer la configuration contenue dans chaque bloc. Il choisira comment fusionner les informations de ces blocs.

Exemple :
On peut avoir un bloc qui donne le nombre d'itérations à faire :
```json
{
  "_": {
    "name": "10 itérations"
  },
  "arcane": {
    "options": {
      "MaxIteration": "10"
    }
  }
}
```

Un autre qui lance Arcane avec 4 sous-domaines en mémoire partagée :
```json
{
  "_": {
    "name": "4 threads"
  },
  "arcane": {
    "options": {
      "S": "4"
    }
  }
}
```

Arcane applique les options une par une et si une option apparait dans deux blocs, la plus récente est conservée.


## Ordre des blocs

Dans un lecteur de fichier ARCT, une liste de blocs de configurations est simplement une liste d'objets JSON triée selon
l'ordre de lecture qu'il est nécessaire de suivre.

Pour écrire ces listes de manière humainement lisible, cette syntaxe a été définie :

- `{}` désigne un bloc ainsi que ses dépendances,
- `[]` désigne le bloc parent ; ainsi, tous les blocs avant sont des dépendances "before" et les blocs après sont des
  dépendances "after",
- `->` est simplement un séparateur de bloc qui montre l'ordre de lecture.

Exemple :

```txt
{ nom_config_1 -> [general] } -> { { [nom_config_2] -> nom_config_4 } -> [nom_du_cas_1] -> nom_config_3 }
```

On peut voir que l'on a deux blocs principaux : `[general]` et `[nom_du_cas_1]` :

- `[general]` a une dépendance "before" qui est `nom_config_1`.
- `[nom_du_cas_1]` a deux dépendances :
  - une dépendance "after" : `nom_config_3`
  - une dépendance "before" : `[nom_config_2]` qui a une dépendance "after" :
    - `nom_config_4`

La liste d'objet JSON est simplement :

```txt
array = [nom_config_1, general, nom_config_2, nom_config_4, nom_du_cas_1, nom_config_3];
```

Si l'on met de côté les dépendances, l'ordre de lecture est systématiquement celui-ci :

```txt
{ [general] } -> { [cas_1] }
```

`general` est toujours le premier bloc (sauf s'il n'est pas présent). Ensuite, le bloc représentant le cas de base choisi par l'utilisateur (ici : `cas_1`) est ajouté à la liste.



## Nom des cas d'utilisation

Il y a deux types de nom (de cas d'utilisation) : les noms complets et les noms incomplets.

Un nom complet est un nom qui, associé à un fichier ARCT, désigne un unique cas d'utilisation (une seule liste de blocs de configurations).

Un nom incomplet est un nom qui, associé à un fichier ARCT, désigne plusieurs cas d'utilisation (plusieurs listes de blocs de configurations).

Ces deux types de noms sont liés aux variations.

Une variation résolue est une variation dont on connait le variant qu'il est nécessaire de mettre dans la liste des
blocs de configurations.

Une variation non résolue est une variation dont on ne connait pas le variant à choisir à partir des informations du
fichier ARCT.

Si un bloc contient une variation non résolue, on aura plusieurs exécutions possibles. Il est possible de choisir, dans le nom de cas d'utilisation, le variant
à utiliser. On résout donc la variation avec un nom complet.
Si l'on utilise un nom incomplet, alors on aura plusieurs listes de blocs de configurations.

Pour résumer, dans un nom de cas d'utilisation, on doit retrouver le nom du cas de base en premier, puis les différentes résolutions de variation si c'est nécessaire.


Pour construire ces noms, nous utilisons les symboles suivants : `:`, `~`, `=`, `!` et `+`.

Le symbole `:` permet de séparer les différents éléments du nom de cas. Il doit être utilisé uniquement pour séparer les
différents éléments constituant le nom de cas. Il ne doit pas être mis au début ou à la fin du nom de cas (s'il n'y a
pas de cas de base à exécuter, seulement des variations à résoudre, pas de `:` au début).

Le symbole `=` permet de résoudre une variation.

Le symbole `!` permet d'exclure des variants de l'ensemble des variants disponibles à lancer.

Le symbole `+` permet d'inclure des variants à lancer, parmi tous les variants disponibles.

Les symboles `!` et `+` permettent de restreindre le nom de variants à traiter. Avec ces symboles, il est impossible de
résoudre totalement une variation. Ces symboles peuvent donc être utilisés uniquement dans le cas de noms incomplets.

Exemple : `cas_1:nb_proc=4:nb_iteration+10+20:eos-model!StiffenedGas`


## Le format

Dans les pages suivantes, plusieurs exemples seront expliqués, du plus simple au plus complet. Deux programmes seront utilisés : Arcane et ArcCher.
Leurs spécifications sont disponibles tout en haut de la page.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_general
</span>
</div>
