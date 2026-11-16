# ARCT ArcCher Block v1 {#arct_arccher_specs}

ArcCher block for ArcaNet File. Version 1.

## Objectifs

Ce fichier a pour objectif de décrire le format de bloc réservé (**reserved_block**) ArcCher qui peut être utilisé dans
les fichiers ARCT.

Ce fichier ne décrira pas le fonctionnement interne de ArcCher. D'autres programmes que ArcCher pourront utiliser ce
bloc, mais ils devront suivre ces spécifications.

Ce bloc a pour but de décrire comment lancer un programme, puis de décrire les options de lancement.

ArcCher est un programme mono-cas prenant en argument au moins un fichier/chemin de fichier ARCT et un nom de cas
d'utilisation complet.

## Spécifications

- Ce bloc est un objet JSON standard suivant les spécifications ARCT,
- Les noms des propriétés suivront la convention snake_case.

## Fonctionnement général

Le bloc ArcCher peut être découpé en deux parties :

- Description du lancement,
- Configuration du lancement.

### Description du lancement

Cette partie est composée de deux objets JSON :

- `codes`,
- `executors`.

Ces deux objets sont identiques. La seule différence est la possibilité d'utiliser l'option réservée `_CODE`.

Dans ces objets, il est possible de décrire la commande de lancement d'un programme et de déclarer des options à
utiliser dans la seconde partie *Configuration du lancement*.

#### Schéma racine -> programme

L'objet `codes` (et `executors`) contient un objet par programme :

```json5
{
  "arccher": {
    "codes": {
      "<program_name>": {
        // ...
      }
    }
  }
}
```

Ces descriptions ne seront pas forcément toutes utiles pour le lancement final ; cela donne la possibilité de décrire
toutes les combinaisons possibles.

Il n'est pas possible d'utiliser le même nom de programme pour un `codes` et pour un `executors`.

Le format du nom de programme `<program_name>` est libre (le caractère `~` a néanmoins une fonction spéciale qui
sera décrite plus bas TODO). D'usage, on utilise soit le `camelCase`, soit le `snake_case`, sans espaces.
Le nom de programme est libre et ne sera pas utilisé dans la ligne de commande.

##### Fusion de blocs

La fusion de deux objets `codes` (ou `executors`) doit suivre ces règles :

- Règle générale : concaténation des objets des blocs `codes` (ou `executors`).
- Si deux objets `<program_name>` ont le même nom : le nouveau écrase l'ancien.

#### Schéma programme

L'objet `<program_name>` peut contenir deux propriétés JSON : un **String** et un objet :

```json5
{
  "<program_name>": {
    "cmdline": "prog {<string_option_1>}",
    "cmdopt": {
      "<string_option_1>": {
        // ...
      }
    }
  }
}
```

La chaîne de caractère `cmdline` permet de formuler la ligne de commande qui devra être construite puis exécutée. Deux
caractères sont réservés afin de pouvoir ajouter des options libres : `{` et `}`. À noter qu'il est possible
d'échapper ces caractères pour les faire apparaitre dans la commande finale.

Pour définir des options libres et/ou configurer des options réservées, il est possible d'utiliser l'objet
`cmdopt`.

#### Schéma propriétés

L'objet `cmdopt` continent un objet par option :

```json5
{
  "cmdopt": {
    "<string_option_1>": {
      "format": "-n {}",
      "optional": true,
      "needed": true,
      "default": "1"
    },
    "<bool_option_1>": {
      "format": "-v",
      "optional": true,
      "needed": false,
      "default": true
    }
  }
}
```

Ces options peuvent avoir deux types différents : `booléen` ou `string`.

Pour détecter quel type est utilisé, il faut vérifier la présence du symbole `{}` dans la propriété `format`. Si ce
symbole est présent, il s'agit d'une option de type `string`, sinon `booléen`.

Lors de l'exécution, le symbole `{}` sera remplacé par la valeur de l'option choisie dans la seconde partie
*Configuration du lancement*, ou par la valeur par défaut.

Deux autres propriétés sont disponibles : `optional` et `needed` :

- `optional` permet de savoir si l'utilisateur est obligé de définir une valeur dans la seconde partie du bloc.
  - Si `optional` est à `false` et que l'utilisateur ne définit pas de valeur, alors une erreur doit être émise.
  - Si `optional` est à `true`, le comportement dépend de la propriété `needed`.

- `needed` permet de savoir si l'option peut être vide ou non.
  - Si `needed` est à `false` et que l'utilisateur n'a pas défini de valeur, alors l'option finale sera vide (dans
    l'exemple, `{<string_option_1>}` sera remplacé par une chaîne vide).
  - Si `needed` est à `true` et que l'utilisateur n'a pas défini de valeur, alors la valeur par défaut sera utilisée
    (dans l'exemple, `{<string_option_1>}` sera remplacé par `-n 1`).

L'utilisation de ces propriétés dépend de ces règles :

- Si `optional` est à `false` : `needed` et `default` sont inutiles.
- Si `needed` est à `false` : `default` est inutile.

Les valeurs par défaut de ces propriétés doivent être les suivantes :

```json5
{
  "format": "{}",
  "optional": false,
  "needed": false,
  "default": ""
}
```

#### Option commune

Si un `codes` et un `executors` possèdent une option commune, les propriétés ne sont pas fusionnées. Chacun suivra les
règles qu'il a définies. 



### Configuration du lancement

Cette partie peut être composée de plusieurs éléments. En effet, c'est dans cette partie que les options libres peuvent
être définies.


#### Options par défaut et options de programme

Il y a deux niveaux d'options dans cette partie : les options par défaut et les options de programme.

Les options par défaut sont des options qui sont définies pour tous les programmes à lancer (s'ils en ont besoin).

Il peut y avoir des options inutile pour le lancement.

Ces options sont à définir à la racine du bloc `arccher` :

```json5
{
  "arccher": {
    "<string_option_1>": "4",
    "<bool_option_1>": true,
    // ...
  }
}
```

Il est possible de définir des options de programme, spécifique pour ces programmes (comme l'emplacement du binaire à
lancer), dans un objet JSON au nom du programme :

```json5
{
  "arccher": {
    "<program_name>": {
      "<string_option_1>": "2"
      // ...
    },
    // ...
  }
}
```

Lorsque le programme `<program_name>` sera lancé, les options par défaut seront écrasées par les options du programme :

```json5
{
  "<string_option_1>": "2",
  "<bool_option_1>": true
}
```

##### Fusion d'option (côté ARCT)

La fusion de deux options doit suivre ces règles :

- Règle générale : la nouvelle option écrase l'ancienne.


#### Programmes à lancer

La propriété `run` est un tableau qui contient l'ordre de lancement des programmes :

```json5
{
  "arccher": {
    "run": ["<program_name>", "<program_name_2>"]
  }
}
```

Cette propriété doit être forcément positionnée à la racine du bloc `arccher`.

Chaque programme de cette liste doit être présent dans l'objet `codes`, sinon une erreur doit être émise (avant le
lancement des programmes).


##### Fusion de tableau

La fusion de deux tableaux `run` doit suivre ces règles :

- Règle générale : concaténation des tableaux `run` dans le même ordre de lecture de la liste de bloc de configuration
  ARCT.

Exemple :

```jsonc
{
  // "versions" block omitted.
  "cases": {
    "case_1": {
      "_": {
        "depend_b": ["feature_1"],
        "depend_a": ["feature_2"]
      },
      "arccher": {
        "run": ["<program_name_case_1>"]
      }
    }
  },
  "commons": {
    "feature_1": {
      "arccher": {
        "run": ["<program_name_feature_1>"]
      }
    },
    "feature_2": {
      "arccher": {
        "run": ["<program_name_feature_2>"]
      }
    }
  }
}
```

Cet exemple donnera le tableau :

```json5
{
  "arccher": {
    "run": ["<program_name_feature_1>", "<program_name_case_1>", "<program_name_feature_2>"]
  }
}
```


#### Variables d'environnement

La propriété `env_var` est un objet JSON qui contient les variables d'environnement à ajouter avant le lancement d'un
programme :

```json5
{
  "arccher": {
    "env_var": {
      "<VARIABLE>": "<VALUE>",
      //...
    },
    "<program_name>": {
      "env_var": {
        "<VARIABLE>": "<NEW_VALUE>",
        //...
      }
    }
  }
}
```

Cette propriété peut être positionnée à la racine du bloc `arccher` et/ou dans un objet JSON de programme.

##### Option de programme / option par défaut

La fusion de deux objets `env_var` doit suivre ces règles :

- Règle générale : concaténation des objets des blocs `env_var`.
- Si deux objets `<VARIABLE>` ont le même nom : l'option de programme écrase l'option par défaut.

##### Fusion de blocs (côté ARCT)

La fusion de deux objets `env_var` doit suivre ces règles :

- Règle générale : concaténation des objets des blocs `env_var`.
- Si deux objets `<VARIABLE>` ont le même nom : le nouveau écrase l'ancien.


#### Options utilisées par ArcCher

Deux options sont utilisées par ArcCher :

- `executor_name` (`string`),
- `working_dir` (`string`),

```json5
{
  "arccher": {
    "executor_name": "mpirun",
    "working_dir": "/home/aaa/build"
  }
}
```
Elles se comportent comme une option classique et peuvent être utilisées dans la partie `cmdline`
(**Description du lancement**).
Ces propriétés peuvent être positionnées à la racine du bloc `arccher` et/ou dans un objet JSON de programme.

L'option `executor_name` est une chaîne de caractères qui continent le lanceur de programme (`mpirun` par exemple).
L'option `working_dir` est une chaîne de caractères qui continent le répertoire de travail où se déplacer avant
l'exécution.



### Options réservées

Trois options sont définies par ArcCher :

- `_ARCT_PATH`,
- `_CASE_NAME`,
- `_CODE` (réservé au bloc `executors`).

Une erreur sera émise si l'utilisateur tente de redéfinir ces options dans la partie **Configuration du lancement**.

Ces options peuvent être utilisées dans la partie `cmdline` et peuvent être reconfigurées dans la partie `cmdopt`.

#### _ARCT_PATH

Cette option contient le chemin vers le fichier ARCT passé en argument de ArcCher.

Cette option est forcément non-vide.

#### _CASE_NAME

Cette option contient le nom de cas d'utilisation passé en argument de ArcCher.

Si le nom de programme contient le caractère `~` avec un nom d'alternative, cette alternative sera ajoutée au nom de
cas.

#### _CODE

Cette option contient la commande pour exécuter le programme. Il s'agit de la résolution de la commande
`codes/<program_name>/cmdline`.

### Options recommandées

Afin d'uniformiser les blocs ArcCher, il est recommandé d'utiliser ces noms pour certaines options libres :

- `executor_binary` : Le chemin vers l'exécutable du lanceur de programme,
- `code_binary` : Le chemin vers l'exécutable du code,
- `nb_proc` : Nombre de processus à lancer,
- `nb_node` : Nombre de noeuds à réserver,
- `nb_thread` : Nombre de threads nécessaire pour le lancement,
- `nb_core_per_proc` : Nombre de coeurs par processus.
