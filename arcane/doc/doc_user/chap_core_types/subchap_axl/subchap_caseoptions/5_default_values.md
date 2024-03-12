# Gestion des valeurs par défaut {#arcanedoc_core_types_axl_caseoptions_default_values}

[TOC]

Il est possible de spécifier une valeur par défaut dans le fichier
**axl** pour les options simples, énumérées, étendues et les
services. Depuis la version 2.10.0 (septembre 2018), il est aussi possible
de définir ces valeurs par catégorie et de spécifier la catégorie
voulue lors de l'exécution. Le choix de la catégorie doit se faire
avant la lecture des options du jeu de donnée, par exemple dans la
classe gérant la session du code :
```cpp
#include "arcane/ICaseDocument.h"
using namespace Arcane;
void f()
{
  ISubDomain* sd = ...;
  ICaseDocument* doc = sd->caseDocument();
  doc->setDefaultCategory("MyCategory");
}
```

Enfin, depuis la version 2.9.1 (juin 2018) de %Arcane, il est aussi possible
de changer dynamiquement lors de l'exécution ces valeurs par
défaut. Cela peut être utile par exemple si on veut des valeurs par
défaut par type de boucle en temps, par dimension du maillage, ...

Pour changer les valeurs par défaut, il existe une méthode
**setDefaultValue()** suivant le type de l'option :

| Classe %Arcane                                  |  Description
|-------------------------------------------------|---------------------------------------
| \arcane{CaseOptionSimpleT::setDefaultValue()}   | options simples
| \arcane{CaseOptionEnumT::setDefaultValue()}     | options énumérées
| \arcane{CaseOptionExtendedT::setDefaultValue()} | options étendues
| \arcane{CaseOptionService::setDefaultValue()}   | services

\note Il n'est pas possible de changer les valeurs par défaut des options
possédant des occurrences multiples.

Pour bien comprendre comment utiliser cette méthode, il est nécessaire
de connaitre les mécanismes de lecture des options du jeu de
données. La lecture du jeu de données se fait en plusieurs phases :
1. Phase 1. Lors de cette phase, on lit toutes les options sauf les
   options étendues car elles peuvent reposer sur le maillage et dans
   cette phase le maillage n'est pas encore lu. C'est lors de cette
   phase que sont aussi créés les différentes instances des services
   qui apparaissent dans le jeu de données.
2. Appel des points d'entrée **Build** du code.
3. Phase 2. Lors de cette phase, le maillage a été lu et on lit les
   options étendues du jeu de données. Après cette phase, toutes les
   options ont été lues.
4. Affichage dans le listing des valeurs des options du jeu de
   données.
5. Appel des points d'entrée **Init** du code.

Pour exécuter du code lors des parties (*2*) et (*5*), il faut utiliser des
points d'entrées déclarées dans la boucle en temps. Pour exécuter du code lors des parties 1 ou 3, il
est possible de s'enregistrer auprès du Arcane::ICaseMng::observable()
pour être notifié du début des phases 1 et 2. Le code sera exécuté
avant que %Arcane n'effectue la phase correspondante. Par exemple :

```cpp
#include "arcane/ObserverPool.h"
using namespace Arcane;
class MyService
{
 public:
  MyService(const ServiceBuildInfo& sbi)
  {
    ICaseMng* cm = sbi.subDomain()->caseMng();
    m_observers.addObserver(this,&MyService::onBeforePhase1,
                            cm->observable(eCaseMngEventType::BeginReadOptionsPhase1));
    m_observers.addObserver(this,&MyService::onBeforePhase2,
                            cm->observable(eCaseMngEventType::BeginReadOptionsPhase2));
  }
  void onBeforePhase1() { ... }
  void onBeforePhase2() { ... }
 private:
  ObserverPool m_observers;
};
```

Les points suivants sont à noter :

- si on souhaite changer la valeur par défaut d'un service, il faut le
  faire lors de la partie (*1*) car ensuite les services ont déjà été
  créés.
- si une valeur par défaut est présente dans le fichier **axl**, ce
  sera cette valeur qui sera utilisée tant qu'il n'y a pas eu d'appel
  à setDefaultValue(). Si on change une valeur d'une option simple
  lors de la partie (*3*) par exemple, elle ne sera pas encore prise
  en compte dans lors de l'appel des points d'entrée **Build** (qui
  sont dans la partie (*2*)).
- il est possible de mettre une valeur par défaut même s'il n'y en a
  pas dans le fichier **axl**. Dans ce cas, il faut la positionner
  dans la partie (*1*) sinon %Arcane considérera le jeu de donnée
  comme invalide après lecture de la phase 1.
- si on souhaite changer une valeur par défaut en fonction des
  informations du maillage, il faut le faire lors de la partie (*3*).


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_usage
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_others
</span>
</div>
