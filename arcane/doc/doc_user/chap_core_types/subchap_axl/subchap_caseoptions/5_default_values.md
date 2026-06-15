# Default Value Management {#arcanedoc_core_types_axl_caseoptions_default_values}

[TOC]

It is possible to specify a default value in the **axl** file for simple,
enumerated, extended options, and services. Since version 2.10.0 (September
2018), it is also possible to define these values by category and specify the
desired category during execution. The category choice must be made before
reading the dataset options, for example in the session managing class:
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

Finally, since version 2.9.1 (June 2018) of %Arcane, it is also possible to
dynamically change these default values during execution. This can be useful,
for example, if you want default values based on the time loop type, mesh
dimension, etc.

To change the default values, there is a **setDefaultValue()** method depending
on the option type:

| %Arcane Class                                   | Description        |
|-------------------------------------------------|--------------------|
| \arcane{CaseOptionSimpleT::setDefaultValue()}   | simple options     |
| \arcane{CaseOptionEnumT::setDefaultValue()}     | enumerated options |
| \arcane{CaseOptionExtendedT::setDefaultValue()} | extended options   |
| \arcane{CaseOptionService::setDefaultValue()}   | services           |

\note It is not possible to change the default values of options that have
multiple occurrences.

To fully understand how to use this method, it is necessary to know the
mechanisms for reading dataset options. The dataset reading occurs in several
phases:
1. Phase 1. During this phase, all options are read except for extended options
   because they may depend on the mesh, and in this phase, the mesh has not yet
   been read. It is also during this phase that the different instances of
   services appearing in the dataset are created.
2. Calling the code's **Build** entry points.
3. Phase 2. During this phase, the mesh has been read and the extended dataset
   options are read. After this phase, all options have been read.
4. Displaying the dataset option values in the listing.
5. Calling the code's **Init** entry points.

To execute code during steps (*2*) and (*5*), you must use entry points declared
in the time loop. To execute code during steps 1 or 3, it is possible to
register with Arcane::ICaseMng::observable() to be notified of the start of
phases 1 and 2. The code will be executed before %Arcane performs the
corresponding phase. For example:

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

The following points should be noted:

- if you want to change the default value of a service, you must do it during
  step (*1*) because the services have already been created afterward.
- if a default value is present in the **axl** file, that value will be used
  until setDefaultValue() is called. If you change the value of a simple option
  during step (*3*), for example, it will not yet be taken into account when
  calling the **Build** entry points (which are in step (*2*)).
- it is possible to set a default value even if there is none in the **axl**
  file. In this case, you must place it in step (*1*), otherwise %Arcane will
  consider the dataset invalid after reading phase 1.
- if you want to change a default value based on mesh information, you must do
  it during step (*3*).


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_usage
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_others
</span>
</div>
