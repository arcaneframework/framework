# Service {#arcanedoc_core_types_service}

[TOC]

A service has the same characteristics as a module, except that it does not have
an entry point. Like the module, it can therefore have variables and
configuration options.

The service is represented by a class and an XML file called a
*service descriptor*.

Services are generally used:
- to capitalize code across multiple modules. For example, a numerical scheme
  service can be created and used by several numerical modules such as thermal
  and hydrodynamics.
- to parameterize a module with multiple algorithms. For example, a service can
  be created to apply an equation of state for the hydrodynamics module. Two
  implementations, `perfect gas` and `stiffened gas`, are then defined, and the
  module is parameterized, via its dataset, with one or the other
  implementation.

From a design perspective, this implies:
- declaring an interface that will be the service contract. For example, here is
  the interface of a service for solving an equation of state on a group of
  meshes passed as an argument:

```cpp
class IEquationOfState
{
 public:
  virtual void applyEOS(const Arcane::CellGroup& group) =0;
};
```

- creating one or more implementations of this interface.

A service instance is created when the dataset of the module that uses this
service is read. The service methods can then be called directly from the
module.

\note
The module's configuration options, particularly those allowing a service
to be referenced, are presented in the document
\ref arcanedoc_core_types_axl_caseoptions.

## Service Descriptor {#arcanedoc_core_types_service_desc}

Like the module, the service descriptor is an XML file with the ".axl"
extension. It presents the characteristics of the service:
- the interfaces it implements,
- its configuration options,
- its variables.

Unlike the module, a service does not have entry points.

The following example defines a service descriptor for solving a "perfect gas"
equation of state:

```xml
<?xml version="1.0"?>
<service name="PerfectGasEOS" version="1.0">
  <description>Dataset PerfectGasEOS service</description>
  <interface name="IEquationOfState"/>

  <options>... Options ...</options>
  <variables>... Variables ...</variables>
</service>
```

The following example defines a service descriptor used for solving a "stiffened
gas" equation of state:

```xml
<?xml version="1.0"?>
<service name="StiffenedGasEOS" version="1.0" type="caseoption">
  <description>Dataset StiffenedGasEOS service</description>
  <interface name="IEquationOfState"/>

  <options>... Options ...</options>
  <variables>... Variables ...</variables>
</service>
```

`type="caseoption"`  
The *type* attribute set to *caseoption* indicates that it is a service that can
be referenced in a dataset.

`singleton="true"`  
It is also possible to specify a *singleton* attribute with a boolean value
indicating whether the service can be a singleton.

\note A service used as a singleton does not have direct access to the dataset
data.

\remark A service used as a singleton does not need to be declared in the module
descriptor (file .axl) but in the code configuration (file .config), given that
the principle of the singleton is to have only one instance for all the code.

## Class Representing the Service {#arcanedoc_core_types_service_class}

Like the module, compiling the files \c PerfectGasEOS.axl and
\c StiffenedGasEOS.axl with the \c axl2cc utility generates the files
PerfectGasEOS_axl.h and StiffenedGasEOS_axl.h, containing the classes
\c ArcanePerfectGasEOSObject and \c ArcaneStiffenedGasEOSObject, which are base
classes for services.

Here are the classes for the services defined previously in the descriptors:

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

The previous example shows that %Arcane requires the service constructor to take
an object of type \c ServiceBuildInfo as a parameter to pass to its base class.
It can also be seen that the service inherits the interface defining the service
contract.

## Connecting the Service to Arcane {#arcanedoc_core_types_service_connectarcane}

A service instance is constructed by the architecture when a module references
the service in its dataset.

The user must therefore provide a function to create an instance of the service
class. %Arcane provides a macro to define a generic creation function. This
macro must be written in the source file where the service is defined.

Here is this macro for the previous examples:

```cpp
ARCANE_REGISTER_SERVICE_PERFECTGASEOS(PerfectGasEOS, PerfectGasEOSService);
ARCANE_REGISTER_SERVICE_STIFFENEDGASEOS(StiffenedGasEOS, StiffenedGasEOSService);
```

*PerfectGasEOS* and *StiffenedGasEOS* correspond to the registration names in
%Arcane and thus to the names by which the services will be referenced in the
modules' dataset. *PerfectGasEOSService* and *StiffenedGasEOSService* correspond
to the C++ class names, and the names following **ARCANE_REGISTER_SERVICE_**
allow the creation function to be defined.

However, it is possible to register a service even if it does not have an axl
file. This is done using the ARCANE_REGISTER_SERVICE() macro. For example, to
register the class *MyClass* as a service in the 'Toto' sub-domain, which
implements the 'IToto' interface, you would write:

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
