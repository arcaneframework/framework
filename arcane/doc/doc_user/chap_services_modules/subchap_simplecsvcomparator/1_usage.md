# Utilisation {#arcanedoc_services_modules_simplecsvcomparator_usage}

[TOC]

## Singleton

Pour une utilisation en tant que singleton (même objet pour tous les modules) :

Placer ces lignes dans le .config de votre projet :

```xml
<singleton-services>
  <service name="SimpleCsvComparator" need="required" />
</singleton-services>
```

Et dans votre/vos module(s) :

```cpp
#include <arcane/ISimpleTableComparator.h>

using namespace Arcane;

ISimpleTableComparator* comparator = ServiceBuilder<ISimpleTableComparator>(subDomain()).getSingleton();
comparator->init(/* Pointeur vers un ISimpleTableOutput */); // Ne doit être fait que par un seul module.
comparator->compareWithReference(); // Ne doit être fait que par un seul module.
```

## Service

Pour une utilisation en tant que service (objet different pour chaque module) : 

Placer ces lignes dans le .axl de votre module :

```xml
<!-- <options> -->
  <service-instance name="simple-table-comparator" type="Arcane::ISimpleTableComparator">
    <description>Service implémentant ISimpleTableComparator</description>
  </service-instance>
<!-- </options> -->
```

Dans le .arc, vous pouvez déclarer l'implémentation à utiliser. Par exemple :

```xml
<!-- <mon-module> -->
  <simple-table-comparator name="SimpleCsvComparator">
  </simple-table-comparator>
<!-- </mon-module> -->
```

Et dans votre module :

```cpp
#include <arcane/ISimpleTableComparator.h>

using namespace Arcane;

options()->simpleCsvComparator()->init(/* Pointeur vers un ISimpleTableOutput */);
options()->simpleCsvComparator()->compareWithReference();
```

(Pour un exemple plus concret, voir les pages suivantes)


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_examples
</span>
</div>
