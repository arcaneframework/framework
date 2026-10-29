# Exemple : blocs commons {#arcanetdoc_arct_commons}

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
        "name": "Algorithme intra-boundingbox",
        "depend_b": ["enable_post_processing"]
      },
      "arcane": {
        "options": {
          "//nonreg/contact-pairs[]/use-contact-intra": "true"
        }
      }
    }
  },

  "commons": {
    "enable_post_processing": {
      "_": {
        "name": "Activation de la sortie VtkHdf"
      },
      "arcane": {
        "options": {
          "//nonreg/post-processor-write": "true",
          "//nonreg/mesh-contact-param[]/div-nb-cell": "1"
        }
      }
    }
  }
}
```

On ajoute ici un bloc `commons`. Ce bloc, dans cet exemple, permet d'activer les sorties sur notre programme.

On a deux types de dépendances : `depend_b` (**before**) et `depend_a` (**after**). L'unique différence est que les
blocs des dépendances **before** sont ajoutés avant notre bloc, alors que les blocs des dépendances **after** sont
ajoutés après.

\note Les blocs de dépendances sont ajoutés dans le même ordre que présenté dans les tableaux `depend_b`/`depend_a` du
fichier ARCT.
<br>
Si :
```jsonc
...
"algo2": {
...
"depend_b": ["aaa", "bbb"],
"depend_a": ["ccc", "ddd"],
...
```
alors on aura :
```txt
... -> { aaa -> bbb -> [algo2] -> ccc -> ddd }
```

Si l'on exécute `algo1`, le résultat sera le même que sur l'exemple précedent. Par contre, pour l'algo 2 :

```shell
./PotatoesSample1 -A,ParamFile=example.arct -A,Case=algo2


{ [general] } -> { enable_post_processing -> [algo2] }
```

On a un `depend_b` dans `algo2` donc le bloc `enable_post_processing` sera exécuté avant `algo2`.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_cases
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_variations1
</span>
</div>
