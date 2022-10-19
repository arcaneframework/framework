# Service {#arcanedoc_core_types_service}

[TOC]

Un service présente les mêmes caractéristiques qu'un module,
sauf qu'il ne possède pas de point d'entrée. Tout comme le module, 
il peut donc posséder des variables et des options de configuration.

Le service est représenté par une classe et 
un fichier XML appelé *descripteur de service*.

Les services sont généralement utilisés :
- pour capitaliser du code entre plusieurs modules. Par exemple, 
  on peut créer un service de schéma numérique utilisé par plusieurs 
  modules numériques comme la thermique et l'hydrodynamique.
- pour paramétrer un module avec plusieurs algorithmes. Par exemple,
  on peut faire un service chargé d'appliquer une équation d'état
  pour le module d'hydrodynamique. On définit alors 2 implémentations
  `gaz parfait` et `stiffened gaz` et on paramètre le module, via
  son jeu de données, avec l'une ou l'autre des implémentations.

Du point de vue de la conception, cela implique :
- de déclarer une interface qui va être le contrat du service.
  Par exemple, voici l'interface d'un service pour la résolution d'une équation
  d'état sur un groupe de mailles passé en argument :

```cpp
class IEquationOfState
{
 public:
  virtual void applyEOS(const Arcane::CellGroup& group) =0;
};
```

- de créer une ou plusieurs implémentations de cette interface.

Une instance de service est créé lors de la lecture du jeu de données
du module qui utilise ce service. Les méthodes du service peuvent alors être 
appelées directement depuis le module.

\note
Les options de configuration du module, notamment celle permettant de
référencer un service, sont présentées dans le document 
\ref arcanedoc_core_types_axl_caseoptions.
  
## Descripteur de service {#arcanedoc_core_types_service_desc}

Comme pour le module, le descripteur de service est un fichier XML ayant 
l'extension ".axl". Il présente les caractéristiques du service:
- la où les interfaces qu'il implémente,
- ses options de configuration,
- ses variables.

Contrairement au module, un service n'a pas de points d'entrée.

L'exemple suivant définit un descripteur de service pour la résolution
d'une équation d'état de type "gaz parfait" :

```xml
<?xml version="1.0"?>
<service name="PerfectGasEOS" version="1.0">
  <description>Jeu de données du service PerfectGasEOS</description>
  <interface name="IEquationOfState" />

  <options> ... Liste des options ... </options>
  <variables> ... Liste des variables ... </variables>
</service>
```
  
L'exemple suivant définit un descripteur de service utilisé pour la résolution
d'une équation d'état de type "stiffened gaz" :

```xml
<?xml version="1.0"?>
<service name="StiffenedGasEOS" version="1.0" type="caseoption">
  <description>Jeu de données du service StiffenedGasEOS</description>
  <interface name="IEquationOfState" />

	<options> ... Liste des options ... </options>
  <variables> ... Liste des variables ... </variables>
</service>
```

`type="caseoption"`  
L'attribut *type* valant *caseoption* indique qu'il s'agit d'un
service pouvant être référencé dans un jeu de données.

`singleton="true"`  
Il est aussi possible de spécifier un attribut *singleton* ayant
comme valeur un booléen indiquant si le service peut être singleton.

\note Un service utilisé en tant que singleton n'a pas d'accès direct
aux données du jeu de données.

\remark Un service utilisé en tant que singleton n'a pas à être déclaré
dans le descripteur de module (fichier .axl) mais dans la configuration
du code (fichier .config) étant donné que le principe du singleton est 
d'avoir une seule instance pour tout le code.

## Classe représentant le service {#arcanedoc_core_types_service_class}

Comme pour le module, la compilation des fichiers \c PerfectGasEOS.axl 
et \c StiffenedGasEOS.axl avec l'utilitaire \c axl2cc génère respectiment
les fichiers PerfectGasEOS_axl.h et StiffenedGasEOS_axl.h contenant les classes 
\c ArcanePerfectGasEOSObject et \c ArcaneStiffenedGasEOSObject, classes de 
base des services.

Voici les classes pour les services définis précédemment dans les
descripteurs :

```cpp
class PerfectGasEOSService 
: public ArcanePerfectGasEOSObject
{
 public:
  explicit PerfectGasEOSService(const Arcane::ServiceBuildInfo& sbi)
	: ArcanePerfectGasEOSObject(sbi) {}

 public:
  void applyEOS(const Arcane::CellGroup& group) override
  {
    // ... corps de la méthode 
  }
};
```

```cpp
class StiffenedGasEOSService 
: public ArcaneStiffenedGasEOSObject
{
 public:
  explicit StiffenedGasEOSService(const Arcane::ServiceBuildInfo& sbi)
	: ArcaneStiffenedGasEOSObject(sbi) {}
	
 public:
  void applyEOS(const Arcane::CellGroup& group) override
  { 
    // ... corps de la méthode 
  }
};
```

L'exemple précédent montre que %Arcane impose que le constructeur d'un service 
prenne un object de type \c ServiceBuildInfo en paramètre pour le transmettre
à sa classe de base. On peut également constater que le service hérite
de l'interface définissant le contrat du service.

## Connexion du service à Arcane {#arcanedoc_core_types_service_connectarcane}

Une instance du service est construite par l'architecture lorsque un module
référence le service dans son jeu de données. 

L'utilisateur doit donc fournir une fonction pour créer une instance 
de la classe du service. %Arcane fournit une macro permettant de définir
une fonction générique de création. Cette macro doit être écrite dans 
le fichier source où est défini le service.

Voici cette macro pour les exemples précédents :

```cpp
ARCANE_REGISTER_SERVICE_PERFECTGASEOS(PerfectGasEOS, PerfectGasEOSService);
ARCANE_REGISTER_SERVICE_STIFFENEDGASEOS(StiffenedGasEOS, StiffenedGasEOSService);
```

*PerfectGasEOS* et *StiffenedGasEOS* correspondent aux noms 
d'enregistrement dans %Arcane et donc aux noms par 
lesquels les services seront référencés dans le jeu de données des modules.
*PerfectGasEOSService* et *StiffenedGasEOSService* correspondent aux noms 
des classes C++ et les noms qui suivent **ARCANE_REGISTER_SERVICE_**
permettent de définir la fonction de création.

Il est toutefois possible d'enregistrer un service même si celui-ci ne possède pas
de fichier axl. Cela se fait par la macro ARCANE_REGISTER_SERVICE(). Par exemple,
pour enregistrer la class *MyClass* comme un service de sous-domaine de nom 'Toto', qui
implémente l'interface 'IToto', on écrira:

```cpp
ARCANE_REGISTER_SERVICE(MyClass,
                        ServiceProperty("Toto",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IToto));
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_module
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl
</span>
</div>