# Module {#arcanedoc_core_types_module}

[TOC]

Un module est un ensemble de **points d'entrée** et 
de **variables**. Il peut posséder des options de configuration 
qui permettent à l'utilisateur de paramètrer le module via le jeu de 
données de la simulation.

Un module est représenté par une classe et un fichier XML appelé 
**descripteur de module**.

## Descripteur de module {#arcanedoc_core_types_module_desc}

Le descripteur de module est un fichier XML ayant 
l'extension ".axl".Il présente les caractéristiques du module :
- ses variables,
- ses points d'entrée,
- ses options de configuration.

```xml
<?xml version="1.0"?>
<module name="Test" version="1.0">
	<name lang="fr">Test</name>

	<description>Descripteur du module Test</description>

	<variables/>

	<entry-points/>

	<options/>
</module>
```
  
Par exemple, le fichier \c Test.axl ci-dessus présente le module 
nommé **Test** dont la classe de base, est \c BasicModule (cas général).
<strong>TODO: ajouter doc référence sur les autres attributs du module.</strong>

Ce module \c Test ne possède ni variable, ni point d'entrée, 
ni option de configuration. 

Les variables et les points d'entrée seront décrits dans les 
chapitres suivants.
Les options de configuration sont présentées dans le document 
\ref arcanedoc_core_types_caseoptions.

## Classe représentant le module {#arcanedoc_core_types_module_class}

Grâce à l'utilitaire \c axl2cc, le fichier \c Test.axl 
génère un fichier Test_axl.h. Ce fichier contient 
la classe \c ArcaneTestObject, classe de base du module de Test.

```cpp
class TestModule
: public ArcaneTestObject
{
 public:
  // Construit un module avec les paramètres spécifiés dans \a mb
  ModuleTest(const ModuleBuildInfo & mbi)
  : ArcaneTestObject(mbi) {}

  // Retourne le numéro de version du module
  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }
};
```

L'exemple précédent montre que %Arcane impose que le constructeur du module 
prenne un object de type \c ModuleBuildInfo en paramètre pour le transmettre
à sa classe de base. %Arcane impose également la définition d'une méthode
\c versionInfo() qui retourne le numéro de version de votre module.

\note
Le fait de dériver de la classe ArcaneTestObject donne accés, entre autre,
aux traces %Arcane (cf \ref arcanedoc_general_traces) et aux méthodes suivantes :  
<table>
<tr><th>Méthode</th><th>Action</th></tr>
<tr><td>\c allCells() </td><td> retourne le groupe de toutes les mailles </td></tr>
<tr><td>\c allNodes() </td><td> retourne le groupe de tous les noeuds </td></tr>
<tr><td>\c allFaces() </td><td> retourne le groupe de toutes les faces </td></tr>
<tr><td>\c ownCells() </td><td> retourne le groupe des mailles propres au sous-domaine </td></tr>
<tr><td>\c ownNodes() </td><td> retourne le groupe des noeuds propres au sous-domaine </td></tr>
<tr><td>\c ownFaces() </td><td> retourne le groupe de toutes les faces propres au sous-domaine </td></tr>
</table>

## Connexion du module à Arcane {#arcanedoc_core_types_module_connectarcane}

Une instance du module est contruite par l'architecture lors de l'exécution. 

L'utilisateur doit donc fournir une fonction pour créer une instance 
de la classe du module. %Arcane fournit une macro permettant de définir
une fonction générique de création. Cette macro doit être écrite dans 
le fichier source où est défini le module. Elle possède le prototype suivant :

```cpp
ARCANE_REGISTER_MODULE_TEST(TestModule);
```

\c *TestModule* correspond au nom de la classe et *TEST* qui suit 
**ARCANE_REGISTER_MODULE_** permet de définir la fonction de création.

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_variable
</span>
</div>