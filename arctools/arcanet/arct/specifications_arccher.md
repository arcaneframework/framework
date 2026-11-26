# ARCT ArcCher Block v1 {#arct_arccher_specs}

ArcCher block for ArcaNet File. Version 1.

## Objectifs

Ce fichier a pour objectif de décrire le format de bloc réservé (**reserved_block**) ArcCher qui peut être utilisé dans
les fichiers ARCT.

Ce fichier ne décrira pas le fonctionnement interne d'ArcCher. D'autres programmes qu'ArcCher pourront utiliser ce
bloc, mais ils devront suivre ces spécifications.

Ce bloc a pour but de décrire comment lancer un programme, puis de décrire les options de lancement.

ArcCher est un programme mono-cas prenant en argument au moins un fichier/chemin de fichier ARCT.
Le nom de cas d'utilisation est facultatif mais, s'il est présent, doit être complet.

## Spécifications

- Ce bloc est un objet JSON standard suivant les spécifications ARCT,
- Les noms des propriétés suivront la convention snake_case,
- Le nom du bloc devant apparaitre dans le bloc ARCT `versions` est `arccher` et un simple entier désigne le numéro de
  version.

Un bloc répondant aux spécifications de ce fichier doit avoir la version 1 :

```json5
{
  "versions": {
    "arccher": 1
  }
}
```

Toute version non supportée par ArcCher doit déclencher une erreur.

## Fonctionnement général

Pour exécuter un code avec ArcCher, nous avons besoin de définir comment lancer ce code en détaillant tous les
paramètres utiles et comment les communiquer au code (via les options de la ligne de commande ou via des variables
d'environnement). Dans cette partie, nous allons déclarer des paramètres, comment les utiliser (en option et/ou en
variable d'environnement) et les configurer (obligatoire ou facultatif).

Puis, nous avons besoin de décrire le lancement que l'on veut effectuer en définissant des valeurs pour ces paramètres.
Dans cette partie, il n'y aura pas de configuration de paramètre.

Le bloc ArcCher peut donc être découpé en deux parties :

- Description du lancement,
- Configuration du lancement.

Lorsqu'une erreur est trouvée dans un des blocs réservés ArcCher, elle doit être émise avant le lancement du premier
programme par ArcCher.

### Description du lancement

Cette partie est composée de deux objets JSON (blocs de description de lancement) :

- `codes`,
- `executors`.

Le format de ces deux objets est identique. La seule différence est la possibilité d'utiliser certains paramètres
prédéfinis dans l'un et non dans l'autre (voir partie **Ordre de résolution**).

Dans ces objets, il est possible de décrire la commande de lancement d'un programme et de déclarer des paramètres à
utiliser dans la seconde partie *Configuration du lancement*.

#### Schéma racine -> programme

Les blocs de description de lancement (`codes` et `executors`) contiennent un objet par programme :

```json5
{
  "arccher": {
    "codes": { // ou "executors"
      "<program_name>": {
        // ...
      }
    }
  }
}
```

Ces descriptions ne seront pas forcément toutes utiles pour le lancement final ; cela donne la possibilité de décrire
toutes les combinaisons possibles.

Il n'est pas possible d'utiliser le même nom de programme pour un `codes` et pour un `executors` (y compris l'executor
implicite `_NONE`), sinon une erreur sera émise.

Le format du nom de programme `<program_name>` est libre. D'usage, on utilise soit le `camelCase`, soit le `snake_case`,
sans espaces.


Le nom de programme ne doit pas contenir ces caractères (sinon erreur) :

- `{`,
- `}`,
- ` ` (espace),
- `~`,
- `\`.

Note : on ne peut pas définir d'alternative dans les blocs de description de lancement, donc le symbole `~` est exclu.

Le nom de programme ne doit pas être un nom réservé par ArcCher (sinon erreur) :

- `codes`,
- `executors`,
- `run`,
- `executor_name`,
- `working_dir`,
- `cmdline`,
- `cmdopt`,
- `cmdenv`,
- `env_var`,
- `_NONE`,
- `_ARCT_PATH`,
- `_CASE_NAME`,
- `_CODE`.

Si un nom de programme et un paramètre ont le même nom, une erreur doit être émise.

Le nom de programme ne sera pas utilisé dans la ligne de commande `cmdline`, il servira uniquement pour la seconde
partie.

##### Fusion de blocs

La fusion de deux objets `codes` (ou `executors`) doit suivre ces règles :

- Règle générale : concaténation des objets des blocs `codes` (ou `executors`).
- Si deux objets `<program_name>` ont le même nom : le nouveau écrase l'ancien.

#### Schéma programme

L'objet `<program_name>` peut contenir trois propriétés JSON : un **String** et deux objets :

```json5
{
  "<program_name>": {
    "cmdline": "prog {<string_param_1>} {<bool_param_1>}",
    "cmdopt": {
      "<string_param_1>": {
        // ...
      }
    },
    "cmdenv": {
      "<string_param_2>": {
        // ...
      }
    }
  }
}
```

La chaîne de caractères `cmdline` permet de formuler la ligne de commande qui devra être construite puis exécutée. Cette
propriété est obligatoirement présente, sinon une erreur sera émise.

La chaîne `cmdline` peut contenir des paramètres qui seront remplacés avant exécution. La configuration de ces
paramètres se fera dans la partie `cmdopt`.
Deux caractères sont réservés afin de pouvoir ajouter des paramètres : `{` et `}`.

À noter qu'il est possible d'échapper le caractère `{` pour le faire apparaitre dans la commande finale avec un
backslash `\`. Le backslash sera supprimé dans la commande finale. Naturellement, il est aussi possible d'échapper le
backslash avec un second backslash :

- Soit `var` = `VALUE` :
  - `cmdline` = `{var}` donnera `VALUE`,
  - `cmdline` = `\{var}` donnera `{var}`,
  - `cmdline` = `\\{var}` donnera `\VALUE`,
  - `cmdline` = `\\\{var}` donnera `\{var}`.

Si un caractère `{` non échappé est trouvé sans `}`, une erreur doit être émise.

Les caractères `}` n'ont pas besoin d'être échappés.

La commande `cmdline` finale, une fois le remplacement des paramètres effectué, ne doit pas être vide (et ne doit pas
contenir que les espaces).

Pour définir et/ou configurer des options de lancement, il est possible d'utiliser l'objet `cmdopt`.

Pour définir et/ou configurer des variables d'environnement, il est possible d'utiliser l'objet `cmdenv`.

Si un paramètre est présent dans la `cmdline` mais n'est pas présent dans `cmdopt` (hors paramètres réservés et
prédéfinis, voir parties **Paramètres prédéfinis** et **Paramètres réservés**), une erreur doit être émise. Il
doit au moins y avoir un objet vide (les propriétés par défaut s'appliqueront donc), exemple :

```json5
{
  "cmdline": "prog {<string_param_1>}",
  "cmdopt": {
    "<string_param_1>": {
    }
  }
}
```

Note : si une autre propriété JSON que `cmdline`, `cmdopt` ou `cmdenv` est présente, il n'y aura pas d'erreurs.

#### Schéma paramètres et environnement

Les objets `cmdopt` et `cmdenv` sont assez semblables mais n'ont pas la même fonction.

L'objet `cmdopt` permet de décrire des options qui pourront être ajoutées dans le morceau de ligne de commande
`cmdline` avec les valeurs des paramètres de la seconde partie *Configuration du lancement*.

L'objet `cmdenv` permet de décrire des variables d'environnement qui pourront être ajoutées avant le lancement du code.
Les valeurs des paramètres seront aussi utilisées dans cette partie.

Si une autre propriété JSON que celles présentées après est présente dans l'objet d'une option de `cmdopt` ou d'une
variable de `cmdvar`, il n'y aura pas d'erreurs.

##### Schéma paramètres

L'objet `cmdopt` contient un objet par option :

```json5
{
  "cmdopt": {
    "<string_param_1>": {
      "format": "-n {}",
      "optional": true,
      "needed": true,
      "default": 1
    },
    "<bool_param_1>": {
      "format": "-v",
      "optional": true,
      "default": true
    }
  }
}
```

L'objet `cmdopt` n'est pas obligatoire.


Ces options peuvent avoir deux types différents : `booléen` ou `string`/`number`.

À noter que le type `number` est converti en `string` et se comporte donc comme celui-ci.

Pour détecter quel type est utilisé, il faut vérifier la présence du symbole `{}` dans la propriété `format`. Si ce
symbole est présent, il s'agit d'une option de type `string`, sinon `booléen`.

##### Type string

Lors de l'exécution, le symbole `{}` sera remplacé par la valeur du paramètre choisie dans la seconde partie
*Configuration du lancement*, ou par la valeur par défaut.

Deux autres propriétés sont disponibles : `optional` et `needed` :

- `optional` permet de savoir si l'utilisateur est obligé de définir une valeur pour le paramètre dans la seconde partie
  du bloc.
  - Si `optional` est à `false` et que l'utilisateur ne définit pas de valeur pour le paramètre, alors une erreur doit
    être émise.
  - Si `optional` est à `true`, le comportement dépend de la propriété `needed`.

- `needed` permet de savoir si l'option doit être obligatoirement présente ou si elle peut être remplacé par un vide.
  - Si `needed` est à `false` et que l'utilisateur n'a pas défini de valeur, alors l'option finale sera vide (dans
    l'exemple, `{<string_param_1>}` sera remplacé par une chaîne vide).
  - Si `needed` est à `true` et que l'utilisateur n'a pas défini de valeur, alors la valeur par défaut sera utilisée
    (dans l'exemple, `{<string_param_1>}` sera remplacé par `-n 1`).

Donc :

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

À noter que l'utilisation de chaînes vides peut conduire à l'introduction de doubles espaces. Il n'est pas nécessaire de
les retirer.

À noter aussi que si un booléen est passé comme valeur à un paramètre de type `string`, il sera converti en `string` :

- `true` = `"true"`
- `false` = `"false"`


##### Type booléen

Par rapport au type `string` où un remplacement de symbole est effectué, ce type permet simplement de placer une option,
spécifiée dans le champ `format`, dans la ligne de commande.

Une propriété est disponible : `optional` :

- `optional` permet de savoir si l'utilisateur est obligé de définir une valeur dans la seconde partie du bloc.
  - Si `optional` est à `false` et que l'utilisateur ne définit pas de valeur, alors une erreur doit être émise.
  - Si `optional` est à `true`, le comportement dépend de la propriété `default`.
    - Si `default` est à `true`, alors l'option est ajoutée.
    - Si `default` est à `false`, alors l'option ne sera pas ajoutée (le symbole de la ligne de commande sera remplacé
      par une chaîne vide).

Les valeurs par défaut de ces propriétés doivent être les suivantes :

```json5
{
  "optional": false,
  "default": false
}
```

`default` peut être de type JSON `bool` ou `string` (`number` étant converti en `string`). S'il est de type `string` :

- `"1"` = `true`
- `"true"` = `true`
- `"True"` = `true`
- `""` = `false`, `"<toutes autres chaînes>"` = `false`.

Ces règles de conversion s'appliquent aussi à la valeur de la propriété (pas uniquement à la propriété `default`) dans
la partie **Configuration du lancement**.


##### Schéma environnement

L'objet `cmdenv` contient un objet par variable :

```json5
{
  "cmdenv": {
    "<string_param_2>": {
      "variable": "VERBOSITY",
      "optional": true,
      "needed": true,
      "default": 1
    }
  }
}
```
L'objet `cmdenv` n'est pas obligatoire.

Deux propriétés sont disponibles : `optional` et `needed` :

- `optional` permet de savoir si l'utilisateur est obligé de définir une valeur pour le paramètre dans la seconde partie
  du bloc.
  - Si `optional` est à `false` et que l'utilisateur ne définit pas de valeur, alors une erreur doit être émise.
  - Si `optional` est à `true`, le comportement dépend de la propriété `needed`.

- `needed` permet de savoir si la variable d'environnement doit être obligatoirement présente ou non.
  - Si `needed` est à `false` et que l'utilisateur n'a pas défini de valeur, alors la variable d'environnement ne sera
    pas définie.
  - Si `needed` est à `true` et que l'utilisateur n'a pas défini de valeur, alors la valeur par défaut sera utilisée
    pour la variable.

Donc :

- Si `optional` est à `false` : `needed` et `default` sont inutiles.
- Si `needed` est à `false` : `default` est inutile.

Les valeurs par défaut de ces propriétés doivent être les suivantes :

```json5
{
  "variable": "",
  "optional": false,
  "needed": false,
  "default": ""
}
```

À noter que si un booléen est passé comme valeur à un paramètre de type `string`, il sera converti en `string` :

- `true` = `"true"`
- `false` = `"false"`


#### Option commune

Si un `codes` et un `executors` possèdent une option ou une variable d'environnement commune, les propriétés ne sont pas
fusionnées. Chacun suivra les règles qu'il a définies. 

#### Executor par défaut

Il est possible qu'un executor ne soit pas spécifié dans la seconde partie. Il est donc nécessaire d'avoir un executor
par défaut qui ne fait que lancer le code.

Un executor spécial doit donc être défini par défaut :

```json5
{
  "arccher": {
    "executors": {
      "_NONE": {
        "cmdline": "{_CODE}"
      }
    }
  }
}
```

Cet executor ne doit pas être présent explicitement dans le fichier ARCT mais défini implicitement par ArcCher
directement. Une erreur sera émise si c'est le cas.

Note : comme `{_CODE}` est toujours définie et non vide, les propriétés n'ont pas d'importance (voir la partie
**Paramètres prédéfinis**).

#### Résolution

Une résolution est le fait de construire un bout de la ligne de commande, en remplaçant les options de `cmdline` par
les valeurs des paramètres du même nom.

Cette résolution doit être faite une seule fois pour un bloc de description de lancement.
Exemple :

```json
{
  "cmdline": "ls {list} $\\{SRCDIR}"
}
```

La résolution de cet exemple, si le paramètre `list` est égal à `"-l"`, donnera forcément le bout de ligne de commande
suivant :

```shell
ls -l ${SRCDIR}
```

`{SRCDIR}` ne devra jamais être remplacé par un hypothétique paramètre `SRCDIR` (ce qui serait le cas si l'on avait deux
résolutions pour cette `cmdline`).


##### Ordre de résolution

L'ordre de résolution est toujours le même :

1. `codes`
2. `executors`

Les blocs de description de lancement, une fois résolus, peuvent générer un ou plusieurs paramètres prédéfinis
utilisables dans les blocs de description de lancement suivants, dans l'ordre de résolution.

Par exemple, les blocs `codes` génèrent le paramètre prédéfini `{_CODE}` qui contient le morceau de ligne de commande
`cmdline` résolu et qui peut être utilisée dans les blocs `executors`.

Les blocs `executors` ne génèrent pas de paramètres.




### Paramètres

Les paramètres, de tous types, sont communs pour les `codes` et les `executors` et permettent de résoudre leurs
options/variables d'environnement.

Il y a trois types de paramètres :

- Paramètres libres,
- Paramètres réservés,
- Paramètres prédéfinis.

Exemple :

```json5
{
  "arccher": {
    // Partie configuration du lancement :
    "executors": {
      "openmpi": {
        "cmdline": "{executor_binary} {nb_proc} {_CODE}",
        "cmdopt": {
          "executor_binary": {
            "format": "{}",
            "optional": true,
            "needed": true,
            "default": "/usr/bin/mpiexec"
          },
          "nb_proc": {
            "format": "-n {}",
            "optional": true,
            "needed": true,
            "default": "1"
          }
        }
      }
    },
    "codes": {
      "arcane": {
        "cmdline": "{code_binary} {_ARCT_PATH} {_CASE_NAME}",
        "cmdopt": {
          "code_binary": {
            "format": "{}"
          },
          "_ARCT_PATH": {
            "format": "-A,ParamFile={}",
            "optional": false
          },
          "_CASE_NAME": {
            "format": "-A,Case={}",
            "optional": true,
            "needed": false
          }
        },
        "cmdenv": {
          "use_backwardcpp": {
            "variable": "ARCANE_USE_BACKWARDCPP",
            "optional": true,
            "needed": false
          }
        }
      }
    },
    
    // Partie description du lancement :
    "run": ["arcane"],
    "working_dir": "/home/user",
    "executor_name": "openmpi",
    "executor_binary": "/usr/bin/mpich/mpirun",
    "use_backwardcpp": 1,
    "arcane": {
      "code_binary": "/home/user/template/HelloWorld",
      "nb_proc": 4
    }
  }
}
```

Il est possible de configurer et d'utiliser ces paramètres dans la partie *Description du lancement*.

Dans cet exemple, nous avons sept paramètres utilisés :

- `executor_binary` (paramètre libre),
- `nb_proc` (paramètre libre),
- `_CODE` (paramètre prédéfini),
- `code_binary` (paramètre libre),
- `_ARCT_PATH` (paramètre prédéfini),
- `_CASE_NAME` (paramètre prédéfini),
- `use_backwardcpp` (paramètre libre).

Il est possible de définir des valeurs pour les paramètres libres et réservés dans la partie
*Configuration du lancement*.

Dans cet exemple, nous avons six paramètres définis :

- `working_dir` (paramètre réservé),
- `executor_name` (paramètre réservé),
- `executor_binary` (paramètre libre),
- `use_backwardcpp` (paramètre libre),
- `code_binary` (paramètre libre),
- `nb_proc` (paramètre libre).


#### Généralités

Si un nom de programme et un nom de paramètre est identique, une erreur doit être émise.

Le nom d'un paramètre ne doit pas contenir ces caractères (sinon erreur) :

- `{`,
- `}`,
- ` ` (espace),
- `~`,
- `\`.

Le nom d'un paramètre ne doit pas être un nom réservé par ArcCher (sinon erreur) :

- `codes`,
- `executors`,
- `run`,
- `env_var`,
- `cmdline`,
- `cmdopt`.

Deux paramètres sont réservés par ArcCher :

- `executor_name`,
- `working_dir`.

Et trois paramètres sont prédéfinis par ArcCher :

- `_ARCT_PATH`,
- `_CASE_NAME`,
- `_CODE` (réservé au bloc `executors`).


#### Paramètres libres

Comme le nom le suggère, un paramètre libre est un paramètre que l'utilisateur peut ajouter/configurer/utiliser comme il
le souhaite. Il peut être utilisé dans la partie `cmdline` et peut être configuré dans la partie `cmdopt`.

Afin d'uniformiser les blocs ArcCher, il est néanmoins recommandé d'utiliser ces noms pour certains paramètres libres :

- `executor_binary` : Le chemin vers l'exécutable du lanceur de programme,
- `code_binary` : Le chemin vers l'exécutable du code,
- `nb_proc` : Nombre de processus à lancer,
- `nb_node` : Nombre de noeuds à réserver,
- `nb_thread` : Nombre de threads nécessaire pour le lancement,
- `nb_core_per_proc` : Nombre de coeurs par processus.


#### Paramètres réservés

Deux paramètres sont utilisés par ArcCher :

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

Ces propriétés peuvent être positionnées à la racine du bloc `arccher` et/ou dans un objet JSON de programme. Leur
fusion se fera comme un paramètre de type string classique.

Ces paramètres peuvent être utilisés dans la partie `cmdline` (**Description du lancement**) mais ne peuvent pas être
configurées dans la partie `cmdopt` (sinon erreur).

Le paramètre `executor_name` est une chaîne de caractères qui contient le lanceur de programme (`mpirun` par exemple).
Si ce paramètre est vide ou non présent, ArcCher doit le remplacer par `_NONE`. Si le lanceur demandé n'existe pas, une
erreur doit être émise.

Le paramètre `working_dir` est une chaîne de caractères qui contient le répertoire de travail où se déplacer avant
l'exécution. Si vide, alors le répertoire courant d'où est lancé ArcCher sera utilisé.


#### Paramètres prédéfinis

Trois paramètres sont définis par ArcCher :

- `_ARCT_PATH`,
- `_CASE_NAME`,
- `_CODE` (uniquement pour le bloc `executors`).

Une erreur sera émise si l'utilisateur tente de redéfinir ces paramètres dans la partie **Configuration du lancement**.

Ces paramètres peuvent être utilisées dans la partie `cmdline` et peuvent être reconfigurées dans la partie `cmdopt`.

#### _ARCT_PATH

Cet paramètre contient le chemin vers le fichier ARCT passé en argument d'ArcCher.

Ce paramètre est forcément non-vide (donc les propriétés `optional`/`needed`/`default` sont inutiles).

#### _CASE_NAME

Ce paramètre contient le nom de cas d'utilisation passé en argument d'ArcCher. Si cet argument n'est pas présent,
ce paramètre suivra les propriétés `optional`/`needed`/`default` comme un paramètre classique.

Si le nom de programme contient le caractère `~` avec un nom d'alternative, cette alternative sera ajoutée au nom de
cas.

#### _CODE

Ce paramètre contient la commande pour exécuter le programme. Il s'agit de la résolution de la commande
`codes/<program_name>/cmdline`.

Ce paramètre est forcément non-vide (donc les propriétés `optional`/`needed`/`default` sont inutiles).





### Configuration du lancement

#### Paramètres par défaut et paramètres de programme

Il y a deux niveaux de paramètres dans cette partie : les paramètres par défaut et les paramètres de programme.

Les paramètres par défaut sont des paramètres qui sont définis pour tous les programmes/lanceurs à lancer (s'ils en ont
besoin).

Les paramètres de programme sont des paramètres spécifiques pour ces programmes (comme l'emplacement du binaire à
lancer). Il est important de noter que les `executors` ne peuvent pas avoir de paramètres de programme dédiés : ils
utilisent les paramètres de programme du `code` lancé.

Si un `executors` possède des paramètres de programme, une erreur doit être émise.

Il peut y avoir des paramètres inutiles pour le lancement.

Ces paramètres sont à définir à la racine du bloc `arccher` :

```json5
{
  "arccher": {
    "<string_param_1>": "4",
    "<bool_param_1>": true,
    // ...
  }
}
```

Les paramètres de programme utilisent un objet JSON au nom du programme :

```json5
{
  "arccher": {
    "<program_name>": {
      "<string_param_1>": "2"
      // ...
    },
    // ...
  }
}
```

Lorsque le programme `<program_name>` sera lancé, les paramètres par défaut seront écrasées par les paramètres du
programme :

```json5
{
  "<string_param_1>": "2",
  "<bool_param_1>": true
}
```

Il est autorisé d'avoir des paramètres de programme "inutiles" (des paramètres pour un programme qui est référencé dans
le bloc `codes` mais qui n'est pas dans le tableau `run`).

En revanche, il n'est pas possible d'avoir des paramètres de programmes inconnus (des paramètres pour un programme qui
n'est pas référencé dans le bloc `codes`). Si tel est le cas, une erreur sera émise.

##### Fusion de paramètres (côté ARCT)

La fusion de deux paramètres doit suivre cette règle :

- le nouveau paramètre écrase l'ancien.

##### Ordre des fusions

D'abord, on fusionne tous les paramètres par défaut de tous les blocs de configuration, dans l'ordre donné par le
lecteur ARCT.

Ensuite, on fusionne tous les paramètres de programme de tous les blocs de configuration, dans l'ordre donné par le
lecteur ARCT.

Enfin, on fusionne les paramètres de programme et les paramètres par défaut (comme décrit au-dessus).

Il est possible d'optimiser ces étapes si le résultat reste le même.


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

Si le nom de programme contient une alternative (`~alt`), il faut que le nom du programme sans alternative soit dans
l'objet `codes` (si l'on a `arcane~alt` dans le tableau `run`, alors `arcane` doit se trouver dans l'objet `codes`).
C'est celui-ci qui sera utilisé.

L'exécution est faite dans l'ordre du tableau ; après les différentes fusions faites ; séquentiellement.

Il peut y avoir plusieurs fois le même nom de programme dedans.

Si un programme émet une erreur pendant son exécution (si son code de retour est différent de 0), ArcCher émet une
erreur, même s'il reste des programmes à lancer.

Si le tableau est vide ou non présent, ArcCher s'exécute et s'arrête sans erreurs, en précisant qu'il n'a rien fait car
la liste de programmes à exécuter est vide.

##### Fusion de tableau

La fusion de deux tableaux `run` doit suivre cette règle :

- concaténation des tableaux `run` dans le même ordre de lecture de la liste de blocs de configuration ARCT.

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

Si la valeur d'une des propriétés au sein de `env_var` n'est pas un `string`, elle sera convertie en `string`. Les
valeurs de type objet JSON ne sont pas autorisées, une erreur devra être émise si une telle valeur est trouvée.

Ces variables d'environnement s'appliquent aussi bien au code `codes` qu'au lanceur `executor` utilisé.

Ces variables d'environnement sont prioritaires par rapport aux variables d'environnement définies dans la partie
`cmdenv` selon cet ordre (du plus prioritaire au moins prioritaire), pour un code de nom `<program_name>` :

1. `<program_name>/env_var`,
2. `<program_name>/<parameter>`,
3. `env_var`,
4. `<parameter>`,

##### Paramètre de programme / paramètre par défaut

La fusion de deux objets `env_var` doit suivre ces règles :

- Règle générale : concaténation des objets des blocs `env_var`.
- Si deux objets `<VARIABLE>` ont le même nom : le paramètre de programme écrase le paramètre par défaut.

##### Fusion de blocs (côté ARCT)

La fusion de deux objets `env_var` doit suivre ces règles :

- Règle générale : concaténation des objets des blocs `env_var`.
- Si deux objets `<VARIABLE>` ont le même nom : le nouveau écrase l'ancien.


##### Ordre des fusions

D'abord, on fusionne tous les blocs de configuration, dans l'ordre donné par le lecteur ARCT.

Puis, on fusionne les paramètres de programme et les paramètres par défaut (comme décrit au-dessus).






### Gestion des blocs réservés alternatifs et des noms de programme alternatifs

Il n'est pas possible de mélanger les deux types d'alternatives. Une erreur sera émise si ce cas se présente.

Exemple de cas interdit :

```json5
{
  "arccher~alt": {
    "run": ["arcane", "arcane~init"],
    "arcane": {},
    "arcane~init": {},
    // ...
  }
}
```

Note : ArcCher devra vérifier le nom de cas d'utilisation et les programmes de `run` pour éviter le mélange via
fallback. Exemple :

Lancement du cas `case_1~alt1` :
`./program_1 -j file.arct -c case_1~alt1`

Dans un des blocs de configuration, `arccher~alt1` n'est pas présent, donc le lecteur ARCT donne le bloc `arccher` qui
présente un nom de programme alternatif :

```json5
{
  "arccher": {
    "run": ["arcane", "arcane~init"],
    "arcane": {},
    "arcane~init": {},
    // ...
  }
}
```

Ce cas doit conduire à une erreur.


#### Blocs réservés alternatifs

Il est possible (mais déconseillé, car cela complexifie la lecture) d'utiliser les blocs réservés alternatifs avec
ArcCher :

```json5
{
  "arccher": {
    "run": ["arcane"],
    "executor_name": "mpirun",
    "working_dir": "/home/aaa/build"
  },
  "arccher~alt": {
    "run": ["arcane"],
    "executor_name": "",
    "working_dir": "/home/aaa/build"
  }
}
```

Dans ce cas, les règles ARCT s'appliquent : si le nom de cas d'utilisation contient `~alt`, alors le bloc `arccher~alt`
sera utilisé. Si le bloc `arccher~alt` n'est pas présent dans l'un des blocs de configuration à exécuter, alors le bloc
`arccher` sera utilisé.

Le nom de l'alternative sera transmis aux programmes exécutés : le nom de cas d'utilisation ne sera pas modifié.

#### Noms de programme alternatifs

Si un nom de programme alternatif est utilisé, alors le nom de cas d'utilisation sera modifié, pour l'exécution du
programme concerné : il devra y avoir l'ajout du nom de l'alternative au nom de cas d'utilisation passé à ArcCher.

La valeur du paramètre `_CASE_NAME` sera donc modifiée.

Le principe d'héritage n'existe pas avec ArcCher : si le tableau `run` contient `arcane~init` mais que l'objet
`arcane~init` n'existe pas, les paramètres par défaut seront utilisés, non l'objet `arcane`.

Il faut que le nom de cas d'utilisation généré soit valide, donc qu'il respecte les règles imposées par le format ARCT.
On hérite donc de ces règles.

De plus, une alternative sans nom de programme est invalide, une erreur devra être émise (`~alt` sans nom de programme
avant, par exemple).
