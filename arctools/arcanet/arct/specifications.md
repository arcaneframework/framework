# ARCT v1

ArcaNet File. Version 1.

## Extension de nom de fichier

Les fichiers ArcaNet utilisent l'extension `.arct`.

## Objectifs

Ce fichier a pour objectif de décrire comment lancer un ou plusieurs cas d'utilisations pour un projet donné.

Tous les programmes de la chaine de calcul de ce projet peuvent prendre ce fichier en paramètre, ainsi que le nom du cas
à exécuter et charge aux programmes de se configurer selon les paramètres donnés.

La construction du nom de cas sera aussi spécifiée dans ce fichier.

Ce fichier ne décrit pas l'ordre de lancement des programmes, uniquement les paramètres de lancement/de configuration.

Dans ce fichier, chaque programme peut avoir un ou plusieurs objets JSON qui lui est totalement réservé, à sa charge (à
lui de les mettre à jour, par exemple).

En pratique, tous les programmes de la chaine de calcul ne supporteront pas ce type de fichier. Un lanceur de programme,
supportant ce type de fichier, pourra prendre à sa charge la configuration de ces programmes, afin de rendre cette
incompatibilité invisible pour l'utilisateur final. Ce lanceur de programme ne sera pas spécifié ici.

## Fonctionnement géneral

Un bloc de base de ce schéma de fichier est un objet JSON appelé **config_block** ("bloc de configuration") contenant
des objets JSON destinés aux programmes demandeurs appelés **reserved_block** ("bloc réservé \[pour un programme
demandeur\]").

Le principe de base de ce fichier est de construire une liste de **config_block**, liste qui dépendra du nom de cas
donné par l'utilisateur.

L'ordre de cette liste dépend des différentes dépendances entres les blocs de configuration.

Le programme demandeur, via un lecteur de fichier ARCT, obtiendra cette liste et devra la traiter, bloc par bloc. Ce
traitement est dépendant du programme demandeur (exemple : si un tableau est présent dans ces blocs, il pourra les
fusionner ou les écraser ou...).

Le programme demandeur aura une liste de blocs de configuration, avec les blocs réservés de tous les programmes. Il peut
lire les blocs réservés d'autres programmes, même les mettre à jour s'il le souhaite, mais il devra tenir compte des
spécifications imposées par le programme propriétaire du bloc.

## Spécifications

- ARCT est un schéma de fichier JSON avec prise en charge des commentaires.

## Schéma général

Le schéma ARCT est composé de quatre objets JSON à la racine :

- `general` : configuration générale,
- `cases` : configurations des cas d'utilisations,
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

Cet objet contient uniquement un bloc de configuration :

```jsonc
{
  "general": {
    // <config_block>
  }
}
```

Cet objet est facultatif.

### Cases

Cet objet doit contenir tous les cas d'utilisations pour le projet.

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

Le format du nom des cas est libre (sauf certains caractères qui sont réservés par le schéma). D'usage, on utilise soit
le `camelCase`, soit le `snake_case`, sans espaces.

Il ne peut pas y avoir deux cas avec le même nom.

Aussi, le nom des cas doit être différent du nom des configurations communes et du nom des variations de configuration.
Si ce n'est pas le cas, et comme le format JSON ne l'interdit pas, le comportement sera indéfini.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom des cas :

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

Le nom des configurations doit être différent du nom des cas d'utilisations et du nom des variations de configuration.
Si ce n'est pas le cas, et comme le format JSON ne l'interdit pas, le comportement sera indéfini.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom de la configuration :

- `:`,
- `=`,
- `~`,
- `!`,
- `+`.

### Variations/Variants

Cet objet doit contenir toutes les variations de configuration, contenant eux-mêmes toutes les variantes de la
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

Le nom des variations doit être différent du nom des cas d'utilisations et du nom des configurations communes. Si ce
n'est pas le cas, et comme le format JSON ne l'interdit pas, le comportement sera indéfini. Le nom des variants peut
être identique aux noms des variations, aux noms des cas d'utilisations et aux noms des configurations communes.

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

La clef `"_"` est réservé au numéro de version du fichier lui-même, décrit dans ce fichier.

Le format de la valeur de version est imposé. Ce doit être soit un entier positif, soit un tableau d'entiers positifs.

Si un programme demandeur possède un bloc réservé dans un bloc de configuration, il doit le versionner, donc mettre le
numéro de version dans l'objet `versions`.

Le programme demandeur doit définir une spécification pour chaque version, qu'il peut rendre public s'il souhaite que
d'autres programmes puisse lire correctement son bloc réservé.

Ce programme ne peut pas mélanger plusieurs versions de ces blocs.

## Schéma du bloc de configuration

Le bloc de configuration est un objet JSON contenant des blocs réservés (eux-même des objets JSON).

Un bloc de configuration peut être un objet vide.

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

Chaque objet du bloc de configuration doit être facultatif.

Si un objet est présent, même vide, un numéro de version doit être présent dans la partie prévue à cet effet.

Le format du nom des programmes est libre (sauf certains caractères qui sont réservés par le schéma). Le seul nom
réservé est `"_"`.

#### Blocs réservés alternatifs

Le format de fichier JSON impose que les noms soient uniques. Néanmoins, il peut être nécessaire de faire apparaitre
plusieurs blocs réservés pour un même programme, suivant le mode de lancement par exemple. Ce sont les blocs réservés
alternatifs.

Un des caractères réservés est justement prévu pour ce cas de figure : `~`.

Exemple :

```jsonc
{
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

Pour voir comment passer par ces blocs réservés alternatif, voir la partie TODO.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom des blocs réservés :

- `:`,
- `=`,
- `~` (sauf pour définir un bloc réservé alternatif),
- `!`,
- `+`.

## Ordre de lecture des blocs de configuration

## Format du nom de cas d'utilisation
