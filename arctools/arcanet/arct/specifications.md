# ARCT v1 {#arct_specs}

ArcaNet File. Version 1.

## Extension de nom de fichier

Les fichiers ArcaNet utilisent l'extension `.arct`.

## Objectifs

Ce fichier a pour objectif de décrire comment lancer un ou plusieurs cas d'utilisation pour un projet donné.

Tous les programmes de la chaine de calcul de ce projet peuvent prendre ce fichier en paramètre, ainsi que le nom du cas
à exécuter et charge aux programmes de se configurer selon les paramètres donnés.

La construction du nom de cas sera aussi spécifiée dans ce fichier.

Ce fichier ne décrit pas l'ordre de lancement des programmes, uniquement les paramètres de lancement/de configuration.

Dans ce fichier, chaque programme peut avoir un ou plusieurs objets JSON qui lui sont totalement réservés, à sa charge
(à lui de les mettre à jour, par exemple).

En pratique, tous les programmes de la chaine de calcul ne supporteront pas ce type de fichier. Un lanceur de programme,
supportant ce type de fichier, pourra prendre à sa charge la configuration de ces programmes, afin de rendre cette
incompatibilité invisible pour l'utilisateur final. Ce lanceur de programme ne sera pas spécifié ici.

## Fonctionnement général

Un bloc de base de ce schéma de fichier est un objet JSON appelé **config_block** ("bloc de configurations") contenant
des objets JSON destinés aux programmes demandeurs appelés **reserved_block** ("bloc réservé (pour un programme
demandeur)").

Le principe de base de ce fichier est de construire une ou plusieurs listes de **config_block**, listes qui dépendront
du nom de cas donné par l'utilisateur.

Un programme demandeur peut supporter ou non le lancement de plusieurs cas d'utilisation en une fois. Si le programme
ne peut lancer qu'un seul cas à la fois, alors il n'obtiendra qu'une seule liste de **config_block**, et l'utilisateur
devra s'assurer que son fichier ARCT et son nom de cas, une fois traités, ne désignera qu'un seul cas d'utilisation (une
erreur devra être affichée si une incompatibilité est détectée).

L'ordre des blocs de ces listes dépend des différentes dépendances entre les blocs de configurations.

Le programme demandeur, via un lecteur de fichier ARCT, obtiendra ces listes et devra les traiter, bloc par bloc. Ce
traitement est dépendant du programme demandeur (exemple : si un tableau est présent dans deux de ces blocs, il pourra
les fusionner ou les écraser ou...).

Le programme demandeur aura des listes de blocs de configurations, avec les blocs réservés de tous les programmes. Il
peut lire les blocs réservés d'autres programmes, même les mettre à jour s'il le souhaite, mais il devra tenir compte
des spécifications imposées par le programme propriétaire du bloc.

Le lecteur de fichier ARCT devra supporter toutes les fonctionnalités et les contraintes exposées ici pour afficher le
support de ce schéma de fichier, dans la version affichée en première ligne de ce fichier.

## Spécifications

- ARCT est un schéma de fichier JSON standard,
- Son encodage doit être en UTF-8 (avec ou sans BOM),
- Les noms des propriétés suivront la convention `snake_case`.

### Commentaires JSON (JSONC)

Les commentaires en JSON (`//` et `/* */`) sont une extension non-officielle du format JSON. Ainsi, tous les
lecteurs/écrivains JSON ne les supportent pas forcément.

Ce schéma n'impose pas le support des commentaires pour les lecteurs JSON.

Pour les écrivains non plus. Ainsi, une mise à jour de fichier ARCT peut provoquer une perte de commentaires.

Pour ces raisons, l'utilisation du JSONC et des commentaires est déconseillé (les exemples de ce document utilisent des
commentaires par convention documentaire).

Pour contourner ça, il est possible de commenter un fichier ARCT/JSON classique avec les noms de propriétés `"//X"`, `X`
désignant un nombre quelconque ou autres (les propriétés commençant par `//` ne seront pas utilisées).

Exemple :

```json
{
  "//1": "Commentaire quelconque."
}
```

## Schéma général

Le schéma ARCT est composé de cinq objets JSON à la racine :

- `general` : configuration générale,
- `cases` : configurations des cas d'utilisation,
- `commons` : configurations communes,
- `variations` : variations de configurations avec plusieurs variants,
- `versions` : versions des **reserved_block**.

Le fichier ARCT minimal nécessite uniquement l'objet `versions` rempli ainsi :

```jsonc
{
  "versions": {
    "_": 1
  }
}
```

### General

Cet objet contient uniquement un bloc de configurations :

```jsonc
{
  "general": {
    // <config_block>
  }
}
```

Cet objet est facultatif.

### Cases

Cet objet doit contenir tous les cas d'utilisation pour le projet.

Exemple de cet objet :

```jsonc
{
  "cases": {
    "nom_du_cas_1": {
      // <config_block>
    },
    "nom_du_cas_2": {
      // <config_block>
    }
  }
}
```

Cet objet est facultatif.

Le format du nom des cas de base est libre (sauf certains caractères qui sont réservés par le schéma). D'usage, on
utilise soit le `camelCase`, soit le `snake_case`, sans espaces.

Il ne peut pas y avoir deux cas avec le même nom.

Aussi, le nom des cas de base doit être différent du nom des configurations communes et du nom des variations de
configurations. Si ce n'est pas le cas, et comme le format JSON ne l'interdit pas, une erreur doit être émise par le
lecteur de fichier ARCT.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom des cas de base :

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Commons

Cet objet doit contenir toutes les configurations communes (configurations qui peuvent être demandées par d'autres
configurations).

Exemple de cet objet :

```jsonc
{
  "commons": {
    "nom_de_la_config_1": {
      // <config_block>
    },
    "nom_de_la_config_2": {
      // <config_block>
    }
  }
}
```

Cet objet est facultatif.

Le format du nom des configurations est libre (sauf certains caractères qui sont réservés par le schéma). D'usage, on
utilise soit le `camelCase`, soit le `snake_case`, sans espaces.

Il ne peut pas y avoir deux configurations avec le même nom.

Le nom des configurations doit être différent du nom des cas d'utilisation et du nom des variations de configurations.
Si ce n'est pas le cas, et comme le format JSON ne l'interdit pas, une erreur doit être émise par le lecteur de fichier
ARCT.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom de la configuration :

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Variations/Variants

Cet objet doit contenir toutes les variations de configurations, contenant elles-mêmes toutes les variantes de la
configuration.

Exemple de cet objet :

```jsonc
{
  "variations": {
    "nom_de_la_variation_1": {
      "nom_du_variant_1_de_la_variation_1": {
        // <config_block>
      },
      "nom_du_variant_2": {
        // <config_block>
      }
    },
    "nom_de_la_variation_2": {
      "nom_du_variant_1_de_la_variation_2": {
        // <config_block>
      },
      "nom_du_variant_2": {
        // <config_block>
      }
    }
  }
}
```

Cet objet est facultatif.

Le format du nom des variations et des variants est libre (sauf certains caractères qui sont réservés par le schéma).
D'usage, on utilise soit le `camelCase`, soit le `snake_case`, sans espaces.

Il ne peut pas y avoir deux variations avec le même nom. Au sein d'une variation, il ne peut pas y avoir deux variants
avec le même nom. En revanche, deux variants de deux variations distinctes peuvent avoir le même nom.

Le nom des variations doit être différent du nom des cas d'utilisation et du nom des configurations communes. Si ce
n'est pas le cas, et comme le format JSON ne l'interdit pas, une erreur doit être émise par le lecteur de fichier ARCT.
Le nom des variants peut être identique aux noms des variations, aux noms des cas d'utilisation et aux noms des
configurations communes.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom des variations/variants :

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Versions

Cet objet doit contenir la version de chaque **reserved_block**.

Exemple de cet objet :

```jsonc
{
  "versions": {
    "_": 1,
    "programme_1": 1,
    "programme_2": [1, 0]
  }
}
```

La clef `"_"` est réservée au numéro de version du schéma du fichier lui-même, décrit dans ce fichier, ainsi qu'à la
version du bloc réservé ARCT.

Le seul élément qui ne pourra pas évoluer dans le schéma de fichier ARCT est la position de ce numéro de version. Ainsi,
tous les lecteurs de fichier ARCT passés et futurs pourront déterminer leurs capacités à lire le fichier uniquement à
l'aide de ce numéro.

Le format de la valeur de version est imposé. Ce doit être soit un entier positif, soit un tableau d'entiers positifs.

Si un programme demandeur possède un bloc réservé classique et/ou un bloc réservé alternatif dans un bloc de
configurations, il doit le versionner, donc mettre le numéro de version dans l'objet `versions`. Les blocs réservés
classiques et alternatifs partagent le même numéro de version. Le nom de l'alternative ne doit donc pas apparaitre dans
l'objet `versions`.

Le programme demandeur doit définir une spécification pour chaque version, qu'il peut rendre public s'il souhaite que
d'autres programmes puissent lire correctement son bloc réservé. La sémantique du numéro de version doit aussi être
spécifiée (pour pouvoir lire et pouvoir comparer les numéros de version correctement).

Ce programme ne peut pas mélanger plusieurs versions de ces blocs.

## Schéma du bloc de configurations

### Schéma général

Le bloc de configurations est un objet JSON contenant des blocs réservés (eux-mêmes des objets JSON).

Un bloc de configurations peut être un objet vide.

Exemple de cet objet :

```jsonc
{
  "_": {
    // <reserved_block>
  },
  "programme_1": {
    // <reserved_block>
  },
  "programme_2": {
    // <reserved_block>
  }
}
```

Chaque objet du bloc de configurations doit être facultatif.

Si un objet est présent, même vide, un numéro de version doit être présent dans la partie prévue à cet effet.

Le format du nom des programmes est libre (sauf certains caractères qui sont réservés par le schéma). Le seul nom
réservé est `"_"`.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom des blocs réservés :

- `:`,
- `=`,
- `~` (sauf pour définir un bloc réservé alternatif),
- `!`,
- `+`.

### Blocs réservés alternatifs

Le format de fichier JSON impose que les noms soient uniques. Néanmoins, il peut être nécessaire de faire apparaitre
plusieurs blocs réservés pour un même programme, suivant le mode de lancement par exemple. Ce sont les blocs réservés
alternatifs (en opposition aux blocs réservés classiques).

Un des caractères réservés est justement prévu pour ce cas de figure : `~`.

Exemple :

```jsonc
"nom_du_cas_1": {
  "_": {
    // <reserved_block>
  },
  "programme_1~init": {
    // <reserved_block>
  },
  "programme_1": {
    // <reserved_block>
  }
}
```

Pour demander le lancement de ce chemin alternatif, il faut le spécifier dans le nom de cas lors du lancement du
programme.

Un lancement classique se ferait avec le nom de cas `nom_du_cas_1`. Pour lancer un chemin alternatif, il faut ajouter le
nom de l'alternative au nom de cas en les séparant du symbole `~` : `nom_du_cas_1~init`.

On sait que le programme demandeur recevra une liste de blocs de configurations. Donc, en théorie, ce serait à lui de
s'occuper de récupérer le bon bloc selon l'alternative choisie. Pour faciliter cette récupération, le lecteur de fichier
ARCT devra proposer une méthode prenant en argument un bloc de configuration et retournant le bloc réservé adéquat.

Pour un bloc de configuration donné et pour une alternative demandée :

- si le bloc réservé alternatif avec le bon nom existe, il est utilisé,
- si le bloc réservé alternatif avec le bon nom n'existe pas, le bloc réservé classique est utilisé,
- si le bloc réservé classique n'existe pas, un bloc vide est retourné (pas d'erreur, les blocs réservés étant
  facultatifs).

Il est important de noter que le nom du cas de base est facultatif (il peut être inutile si l'on souhaite lancer
uniquement le bloc `general`). Un nom de cas peut donc commencer par `~`. Par contre, `~` seul n'est pas autorisé.

### Bloc ARCT réservé

Ce bloc réservé est décrit par ce fichier de spécification et peut être utilisé par les programmes demandeurs (à eux de
faire attention au numéro de version du fichier).

Ce bloc a plusieurs raisons d'être :

- apporter des précisions sur l'utilité de ce bloc de configurations (avec un nom plus précis, par exemple),
- apporter des paramètres communs pour tous les programmes demandeurs,
- étendre facilement le schéma de fichier sans trop de difficultés,
- décrire les dépendances entre les blocs de configurations.

Voici un exemple exhaustif de ce bloc :

```jsonc
{
  "name": "Exemple de cas n°1",
  "depend_b": ["nom_bloc_common_1"],
  "depend_a": ["nom_variation_1"]
}
```

Le nom peut donner plus de précisions sur la fonction de ce bloc.

La propriété `depend_b` décrit les dépendances du bloc à lire avant ce bloc ("b" signifie "before").

La propriété `depend_a` décrit les dépendances du bloc à lire après ce bloc ("a" signifie "after").

Le programme demandeur peut lire ces propriétés s'il le souhaite. La résolution des dépendances est une tâche réservée
au lecteur de fichier ARCT. Le programme demandeur obtient une liste de blocs de configurations dans le bon ordre.

## Ordre de lecture des blocs de configurations et dépendances

### Syntaxe d'une liste de blocs de configurations

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

### Cas général (sans dépendances)

Prenons cet exemple simple :

```jsonc
// file.arct
{
  "versions": {
    "_": 1,
    "programme_1": 1
  },
  "general": {
    "programme_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "nom_du_cas_1": {
      "programme_1": {
        // <reserved_block>
      }
    }
  }
}
```

Et supposons que `programme_1` prenne en argument le fichier ARCT via l'option `-j` et le nom du cas via l'option `-c`.

#### Cas 1

Si nous spécifions uniquement le chemin du fichier ARCT, sans nom de cas, seul le bloc de configurations `general` sera
lu :

`./programme_1 -j file.arct`

```txt
{ [general] }
```

Si le bloc `general` n'est pas présent, alors aucun bloc ne sera présenté au programme demandeur. Au programme demandeur
de décider s'il doit émettre une erreur ou non. Le fichier sera, lui, conforme.

#### Cas 2

Ajoutons maintenant le nom du cas `nom_du_cas_1` :

`./programme_1 -j file.arct -c nom_du_cas_1`

```txt
{ [general] } -> { [nom_du_cas_1] }
```

Sans dépendances, le bloc `general` est toujours lu en premier et le bloc `case` en second.

Mais attention : tous les blocs de configurations sont facultatifs. Si le bloc `general` n'existe pas, il ne sera,
évidemment, pas lu. Seul le bloc `case` le sera (en premier, donc).

En revanche, si le cas `nom_du_cas_1` n'est pas présent dans le fichier ARCT alors qu'il est demandé par le programme
demandeur, alors le lecteur doit émettre une erreur.

Rappel : c'est au programme demandeur de traiter la liste de blocs qui lui est présentée (à lui de déterminer comment
traiter les propriétés en double, par exemple).

### Cas avec dépendances

#### Cas 1

Ajoutons des blocs `commons` et une dépendance dans l'exemple :

```jsonc
// file.arct
{
  "versions": {
    "_": 1,
    "programme_1": 1
  },
  "general": {
    "programme_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "nom_du_cas_1": {
      "_":{
        "depend_a": ["nom_config_1"]
      },
      "programme_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "nom_config_1": {
      "programme_1": {
        // <reserved_block>
      }
    }
  }
}
```

Le nombre de blocs lu va changer si nous demandons le cas `nom_du_cas_1`. En effet, dans ce bloc, nous retrouvons la
propriété `depend_a` qui permet de demander la lecture de blocs après `nom_du_cas_1` :

`./programme_1 -j file.arct -c nom_du_cas_1`

```txt
{ [general] } -> { [nom_du_cas_1] -> nom_config_1 }
```

Si l'on remplace la propriété `depend_a` par `depend_b`, on va retrouver une liste comme celle-ci :

```txt
{ [general] } -> { nom_config_1 -> [nom_du_cas_1] }
```

#### Cas 2

Compliquons un peu l'exemple :

```jsonc
// file.arct
{
  // Blocs "versions" et "general" omis.
  "cases": {
    "nom_du_cas_1": {
      "_":{
        "depend_b": ["nom_config_1"],
        "depend_a": ["nom_config_2", "nom_config_3"]
      },
      "programme_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "nom_config_1": {
      "programme_1": {
        // <reserved_block>
      }
    },
    "nom_config_2": {
      "programme_1": {
        // <reserved_block>
      }
    },
    "nom_config_3": {
      "programme_1": {
        // <reserved_block>
      }
    }
  }
}
```

On mélange `depend_b` et `depend_a` :

`./programme_1 -j file.arct -c nom_du_cas_1`

```txt
{ [general] } -> { nom_config_1 -> [nom_du_cas_1] -> nom_config_2 -> nom_config_3 }
```

#### Cas 3

Dernier exemple de ce type :

```jsonc
// file.arct
{
  // Bloc "versions" omis.
  "general": {
    "_":{
      "depend_b": ["nom_config_1"]
    },
    "programme_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "nom_du_cas_1": {
      "_":{
        "depend_b": ["nom_config_2"],
        "depend_a": ["nom_config_3"]
      },
      "programme_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "nom_config_1": {
      "programme_1": {
        // <reserved_block>
      }
    },
    "nom_config_2": {
      "_":{
        "depend_a": ["nom_config_4"]
      },
      "programme_1": {
        // <reserved_block>
      }
    },
    "nom_config_3": {
      "programme_1": {
        // <reserved_block>
      }
    },
    "nom_config_4": {
      "programme_1": {
        // <reserved_block>
      }
    }
  }
}
```

Tous les blocs de configurations peuvent avoir des dépendances :

`./programme_1 -j file.arct -c nom_du_cas_1`

```txt
{ nom_config_1 -> [general] } -> { { [nom_config_2] -> nom_config_4 } -> [nom_du_cas_1] -> nom_config_3 }
```

#### Cas avec variations

L'ordre des blocs dans la liste avec des variations ne change pas par rapport à l'ordre avec des blocs `commons` :

```jsonc
// file.arct
{
  // Bloc "versions" omis.
  "general": {
    "_":{
      "depend_b": ["nom_config_1"]
    },
    "programme_1": {
      // <reserved_block>
    }
  },
  "cases": {
    "nom_du_cas_1": {
      "_":{
        "depend_b": ["variation_1"],
        "depend_a": ["variation_2=variant_1"]
      },
      "programme_1": {
        // <reserved_block>
      }
    }
  },
  "commons": {
    "nom_config_1": {
      "programme_1": {
        // <reserved_block>
      }
    }
  },
  "variations": {
    "variation_1": {
      "variant_1": {
        "programme_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "programme_1": {
          // <reserved_block>
        }
      }
    },
    "variation_2": {
      "variant_1": {
        "programme_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "programme_1": {
          // <reserved_block>
        }
      }
    }
  }
}
```

La syntaxe pour résoudre des variations est expliquée dans la prochaine partie. Pour faire simple, si on lance ce cas
avec la même ligne de commande que précédemment, on obtient plusieurs cas à lire :

`./programme_1 -j file.arct -c nom_du_cas_1`

```txt
{ nom_config_1 -> [general] } -> { variation_1=variant_1 -> [nom_du_cas_1] -> variation_2=variant_1 }
{ nom_config_1 -> [general] } -> { variation_1=variant_2 -> [nom_du_cas_1] -> variation_2=variant_1 }
```

L'ordre de lecture reste cependant le même qu'avec des dépendances classiques.

### Limitations des dépendances

Certaines règles ont été établies pour les dépendances :

- Pas de cycles (`{ nom_config_1 -> nom_config_2 -> nom_config_1 -> ... }`) : les cycles doivent être détectés par le
  lecteur de fichier ARCT et déclencher une erreur.
- Un bloc de configurations ne peut être lu qu'une seule fois par cas d'utilisation (sinon, le lecteur de fichier ARCT
  émet une erreur).
- Une variation non résolue ne peut être demandée qu'une seule fois par chemin de lancement (sinon, le lecteur de
  fichier ARCT émet une erreur).
- Toutes les dépendances inscrites dans les propriétés `depend_b` et `depend_a` doivent exister.
- Les blocs de configurations `cases` ne peuvent pas être des dépendances, tout comme le bloc `general`.

### Dépendances partagées (diamants)

(Sous-partie rédigée par IA)

Les dépendances d'un chemin de lancement forment un arbre : un même bloc de configurations ne peut pas être demandé
par deux blocs d'un même chemin. C'est une conséquence directe de la règle "un bloc ne peut être lu qu'une seule
fois".

Si le partage était autorisé, le lecteur devrait décider à quelle position insérer la dépendance commune dans la
liste de blocs de configurations : avant le premier demandeur, après le dernier, entre les deux demandeurs... Aucune
de ces positions n'est plus naturelle qu'une autre, et chaque programme demandeur devrait de plus décider comment
traiter les deux lectures du même bloc. Cette complexité justifie l'interdiction.

Le contournement est de porter la dépendance partagée par l'ancêtre commun des blocs qui la demandent.

Exemple de fichier invalide (`socle` serait lu deux fois) :

```jsonc
// file.arct
{
  "cases": {
    "cas_1": {
      "_": {
        "depend_a": ["fonctionnalite_1", "fonctionnalite_2"]
      }
    }
  },
  "commons": {
    "socle": {
    },
    "fonctionnalite_1": {
      "_": {
        "depend_b": ["socle"]
      }
    },
    "fonctionnalite_2": {
      "_": {
        "depend_b": ["socle"]
      }
    }
  }
}
```

Fichier équivalent valide :

```jsonc
// file.arct
{
  "cases": {
    "cas_1": {
      "_": {
        "depend_b": ["socle"],
        "depend_a": ["fonctionnalite_1", "fonctionnalite_2"]
      }
    }
  },
  "commons": {
    "fonctionnalite_1": {
    },
    "fonctionnalite_2": {
    }
  }
}
```

À noter : deux variants frères d'une même variation peuvent dépendre du même bloc : un seul variant étant lu par cas
d'utilisation, il n'y a jamais de lecture multiple.

La même règle s'applique aux variations non résolues : une variation demandée par deux blocs d'un même chemin (par
exemple par le cas de base et par un bloc commun qui en dépend) déclenche une erreur. Sinon, le lecteur devrait
décider quelle demande (et donc quel filtre) l'emporte, puis à quelle position insérer le bloc du variant choisi dans
la liste de blocs de configurations. La dépendance doit, là aussi, être portée par l'ancêtre commun des demandeurs.

## Variations

Prenons cet exemple pour illustrer :

```jsonc
// file.arct
{
  "versions": {
    "_": 1,
    "programme_1": 1
  },
  "general": {
    "programme_1": {
      // <reserved_block>
    }
  },
  // Voir après pour le bloc `cases`.
  "commons": {
    "config_1": {
      "programme_1": {
        // <reserved_block>
      }
    }
  },
  "variations": {
    "variation_1": {
      "variant_1": {
        "programme_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "programme_1": {
          // <reserved_block>
        }
      },
      "variant_3": {
        "programme_1": {
          // <reserved_block>
        }
      }
    },
    "variation_2": {
      "variant_1": {
        "programme_1": {
          // <reserved_block>
        }
      },
      "variant_2": {
        "programme_1": {
          // <reserved_block>
        }
      }
    }
  }
}
```

### Variations résolues

Une variation résolue est une variation dont on connait le variant qu'il est nécessaire de mettre dans la liste des
blocs de configurations.

Prenons le bloc `cases` suivant :

```jsonc
"cases": {
  "cas_1": {
    "_":{
      "depend_a": ["variation_1=variant_1"]
    },
    "programme_1": {
      // <reserved_block>
    }
  }
}
```

Si l'on met une variation résolue dans la propriété `depend_a` (ou `depend_b`, peu importe ici), celle-ci se comporte
comme une dépendance de base `commons` :

`./programme_1 -j file.arct -c cas_1`

```txt
{ [general] } -> { [cas_1] -> variation_1=variant_1 }
```

### Variations non résolues

Une variation non résolue est une variation dont on ne connait pas le variant à choisir à partir des informations du
fichier ARCT.

Prenons un bloc `cases` un peu différent :
```jsonc
"cases": {
  "cas_1": {
    "_":{
      "depend_a": ["variation_1", "variation_2"]
    },
    "programme_1": {
      // <reserved_block>
    }
  }
}
```

Le but d'une variation non résolue est de définir facilement plusieurs combinaisons de cas pour, par exemple, tester un
programme.

Cet exemple définit en réalité six cas :

- `cas_1:variation_1=variant_1:variation_2=variant_1`,
- `cas_1:variation_1=variant_2:variation_2=variant_1`,
- `cas_1:variation_1=variant_3:variation_2=variant_1`,
- `cas_1:variation_1=variant_1:variation_2=variant_2`,
- `cas_1:variation_1=variant_2:variation_2=variant_2`,
- `cas_1:variation_1=variant_3:variation_2=variant_2`.

L'ordre de la liste des listes de blocs de configurations issue de variations non résolues n'est pas défini dans ce
schéma étant donné que tous les cas d'utilisation doivent être indépendants les uns des autres. Au lecteur de choisir
(l'ordre doit néanmoins rester le même entre deux lancements, pour un même fichier).

Note : les exemples présentent toujours des variations non résolues dans les cas `cases`. Nous pouvons mettre des
variations non résolues dans n'importe quel bloc de configurations. Même dans des variations non résolues.

### Définition des variations

Plusieurs symboles sont réservés afin d'affiner la sélection des variations dans un bloc de configurations. Nous avons
déjà vu le symbole `=` qui permet de résoudre d'office une variation.

Deux autres symboles sont disponibles pour restreindre le nombre de variations : `!` et `+`.

Ces trois symboles ne peuvent pas être mélangés pour une même variation.

Le symbole `!` permet d'exclure des variants de l'ensemble des variants disponibles à lancer.

Le symbole `+` permet d'inclure des variants à lancer, parmi tous les variants disponibles.

Si, pour une variation, tous ses variants sont exclus, une erreur doit être émise par le lecteur de fichier ARCT.

#### Exemple

Dans l'exemple précédent, nous avions six cas.

Admettons que nous souhaitons exclure `variation_2=variant_2`. La propriété `depend_a` (ou `depend_b`) devra contenir :

```jsonc
"depend_a": ["variation_1", "variation_2!variant_2"]
```

Dans ce cas, nous n'obtenons plus que trois cas :

- `cas_1:variation_1=variant_1:variation_2=variant_1`,
- `cas_1:variation_1=variant_2:variation_2=variant_1`,
- `cas_1:variation_1=variant_3:variation_2=variant_1`.

On peut aussi vouloir exclure `variation_1=variant_1` et `variation_1=variant_2` :

```jsonc
"depend_a": ["variation_1!variant_1!variant_2", "variation_2!variant_2"]
```

Dans ce cas, il ne nous reste qu'un seul cas :

- `cas_1:variation_1=variant_3:variation_2=variant_1`.

Nous pouvons obtenir ce résultat simplement avec des `=` ou des `+`:

```jsonc
"depend_a": ["variation_1=variant_1", "variation_2=variant_1"]
```

```jsonc
"depend_a": ["variation_1+variant_1", "variation_2+variant_1"]
```

Remarque : une variation est résolue si et seulement si un variant unique est désigné par `=`. `!` et `+` ne font que
filtrer l'ensemble des variants candidats : même réduite à un seul, la variation reste non résolue et le nom reste
incomplet ; pour un programme mono-cas, l'utilisateur doit alors la résoudre par `=` dans le nom de cas.

| Symbole | Rôle       | Sémantique                                                                    | La variation est résolue ? | Exemple (variants : 10, 20, 30) | Noms de cas obtenus                            |
|---------|------------|-------------------------------------------------------------------------------|----------------------------|---------------------------------|------------------------------------------------|
| =       | Résolution | Désigne exactement un variant ; la variation devient une dépendance classique | Oui                        | nb_iterations=10                | cas_1:nb_iterations=10 (un seul cas)           |
| !       | Exclusion  | Retire des variants de l'ensemble des candidats                               | Non                        | nb_iterations!10                | cas_1:nb_iterations=20, cas_1:nb_iterations=30 |
| +       | Inclusion  | Sélectionne des variants parmi les disponibles                                | Non                        | nb_iterations+20+30             | cas_1:nb_iterations=20, cas_1:nb_iterations=30 |

## Schéma d'un nom de cas d'utilisation

Le nom d'un cas d'utilisation peut être complet ou incomplet.

Un nom complet permet de désigner une seule et unique liste de blocs de configurations. C'est nécessaire pour les
programmes demandeurs qui ne peuvent lancer qu'un seul cas d'utilisation en une fois. Un nom complet implique que toutes
les variations du chemin de lancement sont résolues totalement (donc un seul cas d'utilisation), que ce soit via le nom
du cas et/ou via le fichier ARCT directement.

Un nom incomplet permet de désigner plusieurs listes de blocs de configurations. C'est utile pour les programmes
demandeurs qui peuvent traiter plusieurs cas d'utilisation en une fois. Un nom incomplet implique qu'il y a au moins une
variation du chemin de lancement qui n'est pas résolue totalement, qui implique donc plusieurs cas d'utilisation
(un par variant).

Pour construire ces noms, nous utilisons les mêmes symboles que dans le fichier ARCT : `:`, `~`, `=`, `!` et `+`.

Le symbole `:` permet de séparer les différents éléments du nom de cas. Il doit être utilisé uniquement pour séparer les
différents éléments constituant le nom de cas. Il ne doit pas être mis au début ou à la fin du nom de cas (s'il n'y a
pas de cas de base à exécuter, seulement des variations à résoudre, pas de `:` au début).

Le symbole `=` permet de résoudre une variation.

Les symboles `!` et `+` permettent de restreindre le nom de variants à traiter. Avec ces symboles, il est impossible de
résoudre totalement une variation. Ces symboles peuvent donc être utilisés uniquement dans le cas de noms incomplets.

Les variations résolues dans le fichier ARCT sont considérées comme des dépendances classiques. Donc, si une variation
résolue réapparait dans le nom de cas d'utilisation, une erreur sera émise par le lecteur de fichier ARCT.

Si une variation est résolue, totalement ou partiellement, dans le nom du cas d'utilisation alors que cette variation
n'apparait pas dans la liste de bloc finale, une erreur sera émise par le lecteur de fichier ARCT.

Il ne peut pas y avoir deux résolutions (partielles ou totales) pour une même variation dans un nom de cas.

### Ordre des éléments dans le nom

Un nom de cas doit toujours commencer par le nom du cas de base (du bloc de configurations `cases`) sauf s'il n'y a pas
de cas de base à exécuter (le bloc `general` peut avoir des dépendances variations).

Ensuite, il peut y avoir des résolutions de variations (complètes ou partielles). L'ordre de ces résolutions n'est pas
défini.

Une résolution doit avoir la forme : `<variation><symbole><variant>[<symbole><variant>...]`.

Un nom de cas doit ressembler à ça :
`[[<cas_de_base>][~<nom_alternative>][:<variation><symbole><variant>[<symbole><variant> ...]][: ...]]`.

Exemples de nom valides :

- `cas_1:variation_1=variant_1`.
- `cas_1:variation_1=variant_1:variation_2+variant_2+variant_3`.
- `cas_1~init:variation_1=variant_1:variation_2+variant_2+variant_3`.
- `~init:variation_1=variant_1:variation_2+variant_2+variant_3`.
- `~init`.
- `variation_1=variant_1`.

Exemples de nom invalides :

- `cas_1:variation_1=variant_1:`.
- `:cas_1:variation_1=variant_1:`.
- `variation_1=variant_1:cas_1`.
- `variation_1=variant_1:cas_1~init`.
- `variation_1=variant_1:~init`.

Note : les espaces dans le nom ne sont pas supportés.

### Exemple

Pour les programmes demandeurs nécessitant un nom complet, il est possible de spécifier dans la ligne de commande le nom
du ou des cas résolus avec le symbole `=` (les autres symboles n'étant pas supportés ici) :

`./programme_1 -j file.arct -c cas_1:variation_1=variant_1:variation_2=variant_2`

```txt
{ [general] } -> { [cas_1] -> variation_1=variant_1 -> variation_2=variant_2 }
```

Pour les programmes demandeurs pouvant supporter un nom incomplet, il est possible d'utiliser les autres symboles dans
la ligne de commande :

`./programme_1 -j file.arct -c cas_1:variation_2!variant_2`

```txt
{ [general] } -> { [cas_1] -> variation_1=variant_1 -> variation_2=variant_1 }
{ [general] } -> { [cas_1] -> variation_1=variant_2 -> variation_2=variant_1 }
{ [general] } -> { [cas_1] -> variation_1=variant_3 -> variation_2=variant_1 }
```

Note : c'est au programme demandeur de dire au lecteur de fichier ARCT s'il supporte les noms incomplets ou non. Le
lecteur analysera le fichier et le nom du cas et renverra une erreur s'il y a une incompatibilité.

