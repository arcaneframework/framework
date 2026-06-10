# Usage {#arcanedoc_services_modules_simplecsvcomparator_usage}

[TOC]

## Singleton

For use as a singleton (same object for all modules):

Place these lines in your project's .config:

```xml
<singleton-services>
  <service name="SimpleCsvComparator" need="required" />
</singleton-services>
```

And in your module(s):

```cpp
#include <arcane/core/ISimpleTableComparator.h>

using namespace Arcane;

ISimpleTableComparator* comparator = ServiceBuilder<ISimpleTableComparator>(subDomain()).getSingleton();
comparator->init(/* Pointer to an ISimpleTableOutput */); // Must only be done by a single module.
comparator->compareWithReference(); // Must only be done by a single module.
```

## Service

For use as a service (different object for each module): 

Place these lines in your module's .axl:

```xml
<!-- <options> -->
  <service-instance name="simple-table-comparator" type="Arcane::ISimpleTableComparator">
    <description>Service implementing ISimpleTableComparator</description>
  </service-instance>
<!-- </options> -->
```

In the .arc, you can declare the implementation to use. For example:

```xml
<!-- <mon-module> -->
  <simple-table-comparator name="SimpleCsvComparator">
  </simple-table-comparator>
<!-- </mon-module> -->
```

And in your module:

```cpp
#include <arcane/core/ISimpleTableComparator.h>

using namespace Arcane;

options()->simpleCsvComparator()->init(/* Pointer to an ISimpleTableOutput */);
options()->simpleCsvComparator()->compareWithReference();
```

(For a more concrete example, see the following pages)


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator
</span>
<span class="next_section_button">
\ref arcanedoc_services_modules_simplecsvcomparator_examples
</span>
</div>
