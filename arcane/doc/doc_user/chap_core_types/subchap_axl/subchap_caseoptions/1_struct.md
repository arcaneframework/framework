# File Structure {#arcanedoc_core_types_axl_caseoptions_struct}

[TOC]

The module descriptor is in XML format. We will focus on the options
configuration part contained in the \c options element of this file. Here is an
example:

```xml
<options>
  <simple name="simple-real" type="real">
    <name lang='fr'>reel-simple</name>
    <description>Réel simple</description>
  </simple>
</options>
```

This example defines a configuration option called *simple-real*. This option is
a simple variable of type `real` without a default value.

The structure of any options configuration element of a module is similar to
this one. All possible options must appear in child elements of \c options.

The different possibilities are as follows:
- simple options, of type `real`, `bool`, `integer`, or `string`.
- enumerated options, which must correspond to a C++ `enum` type.
- extended type options. These are user-created types (classes, structures...).
  This includes, for example, mesh entity groups.
- complex options, which are themselves composed of options. Complex options can
  be nested.
- service options, which allow referencing a service (see document \ref
  arcanedoc_core_types_service).


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_common_struct
</span>
</div>
