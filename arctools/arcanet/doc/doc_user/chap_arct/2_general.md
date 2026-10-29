# Simple exemple {#arcanetdoc_arct_general}

[TOC]

Fichier `example.arct` :

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  },
  
  "general": {
    "_": {
      "name": "Lancement basique"
    },
    "arcane": {
      "dataset": "config_2D_diagonal_circle.arc",
      "options": {
        "MaxIteration": "10"
      }
    }
  }
}
```

Cet exemple est très simple. Il permet de donner deux informations à Arcane : le jeu de données à utiliser et le nombre
d'itérations à réaliser.

Arcane supportant ce type de fichier, il suffit de le mettre en argument pour qu'il soit lu :

```shell
./PotatoesSample1 -A,ParamFile=example.arct


{ [general] }
```

Il n'y a pas de cas de base dans le fichier, donc pas besoin d'en spécifier.

À noter qu'un fichier comme celui-ci fonctionne aussi et est parfaitement valide (mais ne fait strictement rien) :

```json
{
  "versions": {
    "_": 1,
    "arcane": 1
  }
}
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanetdoc_arct_basis
</span>
<span class="next_section_button">
\ref arcanetdoc_arct_cases
</span>
</div>
