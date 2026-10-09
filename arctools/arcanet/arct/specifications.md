# ARCT v1

ArcaNet File.
Version 1.

## Extension de nom de fichier

Les fichiers ArcaNet utilisent l'extension `.arct`.

## Objectifs

Ce fichier a pour objectif de décrire comment lancer un ou plusieurs cas
d'utilisations pour un projet donné.

Tous les programmes de la chaine de calcul de ce projet peuvent prendre ce
fichier en paramètre, ainsi que le nom du cas à exécuter et charge aux
programmes de se configurer selon les paramètres donnés.

La construction du nom de cas sera aussi spécifiée dans ce fichier.

Ce fichier ne décrit pas l'ordre de lancement des programmes, uniquement les
paramètres de lancement/de configuration.

Dans ce fichier, chaque programme peut avoir un ou plusieurs objets JSON qui lui
est totalement réservé, à sa charge (à lui de les mettre à jour, par exemple).

En pratique, tous les programmes de la chaine de calcul ne supporteront pas
ce type de fichier. Un lanceur de programme, supportant ce type de fichier,
pourra prendre à sa charge la configuration de ces programmes, afin de rendre
cette incompatibilité invisible pour l'utilisateur final. Ce lanceur de
programme ne sera pas spécifié ici.

## Fonctionnement

Un bloc de base de ce schéma de fichier est un objet JSON appelé
**config_block** ("bloc de configuration") contenant des objets JSON
destinés aux programmes demandeurs appelés **reserved_block** ("bloc réservé
\[pour un programme demandeur\]").

Le principe de base de ce fichier est de construire une liste de
**config_block**, liste qui dépendra du nom de cas donné par l'utilisateur.

L'ordre de cette liste dépend des différentes dépendances entres les blocs de
configuration.

Le programme demandeur, via un lecteur de fichier ARCT, obtiendra cette liste
et devra la traiter, bloc par bloc.
Ce traitement est dépendant du programme demandeur (exemple : si un tableau est
présent dans ces blocs, il pourra les fusionner ou les écraser ou...).

Le programme demandeur aura une liste de blocs de configuration, avec les blocs
réservés de tous les programmes. Il peut lire les blocs réservés d'autres
programmes, même les mettre à jour s'il le souhaite, mais il devra tenir compte
des spécifications imposées par le programme propriétaire du bloc.

## Spécifications

- ARCT est un schéma de fichier JSON avec prise en charge des commentaires.

## Schéma général

Le schéma ARCT est composé de quatre objets JSON à la racine :
- `general`,
- `cases`,
- `commons`,
- `variations`,
- `versions`.

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

### Cases

Cet objet doit contenir tous les cas d'utilisations pour le projet.

Exemple de cet objet :

```jsonc
{
  "cases": {
    "nom_du_cas": {
      // <config_block>
    }
  }
}
```

Comme spécifié dans le format de fichier JSON, il ne peut pas y avoir deux
cas avec le même nom.

#### Caractères réservés

Les caractères suivants ne peuvent pas être utilisés dans le nom du cas :

- `:`, 
- `=`, 
- `~`, 
- `!`, 
- `+`.



### Versions

Cet objet doit contenir la version de chaque **reserved_block**.

La clef `"_"` est réservé au numéro de version du fichier lui-même, décrit dans
ce fichier.

Si un programme demandeur possède un bloc réservé dans un bloc de configuration,
il doit le versionner, donc mettre le numéro de version dans l'objet `versions`.

Le programme demandeur doit définir une spécification pour chaque version, qu'il
peut rendre public s'il souhaite que d'autres programmes puisse lire
correctement son bloc réservé.

Ce programme ne peut pas mélanger plusieurs versions de ces blocs.

## Schéma du bloc réservé

## Ordre de lecture des blocs de configuration

