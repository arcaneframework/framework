# Attributes and properties common to all options {#arcanedoc_core_types_axl_caseoptions_common_struct}

[TOC]

Regardless of the option, the element defining it must contain the following two
attributes:

<table>
<tr>
<th>attribute</th>
<th>occurrence</th>
<th>description</th>
<tr>
<td>`name`</td>
<td>required</td>
<td>name of the option. It must consist only of lowercase alphabetic characters
with '-' as a separator. This name is used to generate a field of the same name
in the C++ class where '-' are replaced by uppercase letters. For example, an
option named `max-epsilon` will be retrieved in the code by the `maxEpsilon()`
method,
</td>
</tr>
<tr>
<td>`type`</td>
<td>required</td>
<td>type of the option. The value and meaning of this attribute depend on the
option type.</td>
</tr>
<tr>
<td>`default`</td>
<td>optional</td>
<td>default value of the option. If the option is not present in the dataset,
%Arcane will act as if the value of this attribute was entered by the user. The
option will then take the value provided in the \c default attribute. If this
attribute is not present, the option has no default value and must always be
present in the dataset.</td>
</tr>
<tr>
<td>`minOccurs`</td>
<td>optional</td>
<td>integer that specifies the minimum possible number of occurrences for the
element. If this value is zero, the option can be omitted even if the `default`
attribute is absent. If this attribute is absent, the minimum number of
occurrences is 1.</td>
</tr>
<tr>
<td>`maxOccurs`</td>
<td>optional</td>
<td>integer that specifies the maximum possible number of occurrences for the
element. This value must be greater than or equal to `minOccurs`. The special
value `unbounded` means that the maximum number of occurrences is not limited.
If this attribute is absent, the maximum number of occurrences is 1.</td>
</tr>
</table>

For each option, the following child elements can be added:

<table>
<tr>
<th>element</th>
<th>occurrence</th>
<th>description</th>
</tr>
<tr>
<td>`description`</td>
<td>optional</td>
<td>which is used to describe the use of the option. This description can use
HTML elements. The content of this element is used by %Arcane for generating the
dataset documentation.
</td>
</tr>
<tr>
<td>`userclass`</td>
<td>optional</td>
<td>indicates the option's belonging class. This class specifies a user
category, which allows, for example, restricting certain options to a specific
category. By default, if this element is absent, the option is only usable for
the user class. It is possible to specify this element multiple times with a
different category each time. In this case, the option belongs to all specified
categories.
</td>
</tr>
<tr>
<td>`defaultvalue`</td>
<td>0..infinity</td>
<td>allows indicating a default value for a given category. For example:

```xml
<simple name="simple-real" type="real">
  <defaultvalue category="Code1">2.0</defaultvalue>
  <defaultvalue category="Code2">3.0</defaultvalue>
</simple>
```

In the previous example, if the category is 'Code1', the default value will be '
2.0'. It is possible to specify as many categories as desired. The category used
during execution is set via the Arcane::ICaseDocument::setDefaultCategory()
method.
</td>
</tr>
<tr>
<td>`name`</td>
<td>0..infinity</td>
<td>
allows indicating a translation for the option's name. The value of this element
is the translated name for the option corresponding to the language specified by
the <tt>lang</tt> attribute. For example:

```xml
<simple name="simple-real" type="real">
  <name lang='fr'>reel-simple</name>
</simple>
```

indicates that the option 'simple-real' is called 'reel-simple' in French.
Multiple <tt>name</tt> elements are possible, each specifying a translation. The
dataset must be provided in the default language, French in our case. If no
translation is provided, the value of the \c name attribute is used.
</td>
</tr>
</table>


____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions_struct
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_axl_caseoptions_options
</span>
</div>
