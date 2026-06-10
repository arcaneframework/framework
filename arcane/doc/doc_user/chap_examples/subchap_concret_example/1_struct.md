# General Structure {#arcanedoc_examples_concret_example_struct}

[TOC]

## Quicksilver Arcane Mini-App (QAMA) {#arcanedoc_examples_concret_example_struct_qama}

QAMA is a Monte-Carlo particle transport mini-application. This application was
written using Quicksilver, a mini-app written by LLNL but utilizing resources
provided by the %Arcane framework.

Here is a diagram representing the structure of Quicksilver (available
here: https://github.com/arcaneframework/arcane-benchs):

\image html QAMA_schema.jpg

The following elements can be found (among others):
- 3 modules named "QS", "SamplingMC", and "TrackingMC".
- 1 "singleton" service named "RNG".

In each module, we have the three usual files:
- a header (.h)
- a source file (.cc)
- a file containing the dataset options (.axl)

The interface for the RNG service is a service interface included in the %Arcane
framework. In Quicksilver, we use our own implementation.

\warning
Certain services, such as `BasicParticleExchanger` implementing the interface
`Arcane::IParticleExchanger` or `SimpleCsvOutput` implementing the interface
`Arcane::ISimpleTableOutput`, are services included directly in the %Arcane
framework (this is why they are dashed).

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_concret_example
</span>
<span class="next_section_button">
\ref arcanedoc_examples_concret_example_config
</span>
</div>
