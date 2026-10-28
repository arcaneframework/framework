# Exemple : cas de base {#arcanetdoc_arct_cases}

[TOC]

Fichier `example.arct` :

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  },

  "general": {
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc"
    }
  },

  "cases": {
    "algo1": {
      "_": {
        "name": "Algorithme inter-boundingbox"
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "false"
        }
      }
    },
    "algo2": {
      "_": {
        "name": "Algorithme intra-boundingbox"
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "true"
        }
      }
    }
  }
}
```

On reste sur un exemple simple, mais en ajoutant deux cas de base : `algo1` et `algo2`.

On laisse le `dataset` dans `general` : on est sur un ARCT dédié à `config_2D_diagonal_circle`. On pourra simplement
choisir deux algorithmes différents.

On pourrait très bien répéter `dataset` dans les deux blocs `cases`, ça revient au même ici. Tous les blocs de
configurations peuvent contenir les mêmes éléments.

Pour lancer l'algo 1, on peut faire ainsi :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1
```
```txt
{ [general] } -> { [algo1] }
```

Si l'on ne met pas `-A,Case=algo1`, seul le bloc `general` sera pris en compte.

Voici les blocs reçus par Arcane :

```shell
./PotatoesSample1 -A,ParamFile=example.arct
```
```txt
{ [general] }
```

donne :

```txt
// Bloc 1
"arcane": {
  "dataset": "config_2D_diagonal_circle.arc"
}
```

----

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo1
```
```txt
{ [general] } -> { [algo1] }
```

donne :

```txt
// Bloc 1
  "arcane": {
    "dataset": "config_2D_diagonal_circle.arc"
  }




// Bloc 2
  "_": {
    "name": "Algorithme inter-boundingbox"
  },
  "arcane": {
    "options": {
      "//nonreg/contact-pairs[]/use-contact-intra": "false"
    }
  }
```

----

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2
```
```txt
{ [general] } -> { [algo2] }
```

donne :

```txt
// Bloc 1
  "arcane": {
    "dataset": "config_2D_diagonal_circle.arc"
  }


// Bloc 2
  "_": {
    "name": "Algorithme intra-boundingbox"
  },
  "arcane": {
    "options": {
      "//nonreg/contact-pairs[]/use-contact-intra": "true"
    }
  }
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_general
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_commons
</span>
</div>
