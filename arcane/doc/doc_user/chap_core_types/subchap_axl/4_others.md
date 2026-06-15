# Other elements {#arcanedoc_core_types_axl_others}

[TOC]

## The documentation {#arcanedoc_core_types_axl_others_doc}

Let's take this example:
```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <userclass>User</userclass>
  <description>
    Descripteur du module Test
  </description>

  <variables>
    <variable field-name="pressure" name="Pressure" data-type="real"
              item-kind="cell" dim="0" dump="true" need-sync="true">
      <userclass>User</userclass>
      <description>Descripteur de la variable pressure</description>
    </variable>
  </variables>

  <entry-points>
    <entry-point method-name="testPressureSync" name="TestPressureSync"
                 where="compute-loop" property="none">
      <userclass>User</userclass>
      <description>Descripteur du point d''entrée TestPressureSync</description>
    </entry-point>
  </entry-points>

  <options>
    <simple name="simple-real" type="real">
      <name lang='fr'>reel-simple</name>
      <userclass>User</userclass>
      <description>Réel simple</description>
    </simple>
  </options>
</module>
```
This is valid for modules and services.

### <userclass> Tags

The <userclass> tags are available and allow you to choose which documentation
version each element appears in, along with the corresponding description.

The possible values are `User` and `Dev`.

### <description> Tags

The <description> tags are available for every element in the `axl`. In addition
to describing the element for those who read the `axl` file, this field allows
generating documentation for this module/service. The generation will be
performed by the `axldoc` program.

This program will generate (among others) a page listing all available services
and modules (`axldoc_casemainpage.md`, available here: \ref axldoc_casemainpage)
and a page for the service/module.

\todo Maybe create a page to explain how to generate this documentation for a
project using %Arcane.

### Specifics of <userclass> and <description> tags between <module> (or <service>) tags <em>(lines 3-6 of the example)</em>

Let's assume we are generating the `User` documentation.
For our module to appear in the list of modules, it is necessary to add
`<userclass>User</userclass>` between the `<module>` (or `<service>`) tags.

On the page \ref axldoc_casemainpage "axldoc_casemainpage.md", it is possible to
see that there are descriptions for each module and service. It is now possible
to control the generation of this description. Three attributes are available
for the `<description>` tag:


<details>
  <summary>Attribute doc-brief-max-nb-of-char</summary>
  ```xml
  <description doc-brief-max-nb-of-char="120"></description>
  ```
    Attribute allowing you to limit the maximum number of characters in the
short description.
  By default, the limit is set to `120`. Setting the value to `-1` allows you to
disable this limit.
</details>

<details>
  <summary>Attribute doc-brief-force-close-cmds</summary>
  ```xml
  <description doc-brief-force-close-cmds="true"></description>
  ```
  Attribute allowing you to force `axldoc` not to cut the description when the
  character limit is reached in a Doxygen command before the end of that command.
  This is a very useful attribute, for example, to prevent cutting a LaTeX formula
  in the middle.

  However, `axldoc` guarantees that compatible tags are properly closed and closes
  them if they are not.

  The compatible tags are:
  - `\verbatim \endverbatim`
  - `\code \endcode`
  - `\f$ \f$`
  - `\f( \f)`
  - `\f[ \f]`
  - `\f{ \f}`

  By default, this attribute is set to `false`.

  Example:
  ```xml
  <description doc-brief-force-close-cmds="false" doc-brief-max-nb-of-char="120">
   Ma description
     \f[
     |I_2|=\left| \int_{0}^T \psi(t)
     \left\{
     u(a,t)-
     \int_{\gamma(t)}^a
     \frac{d\theta}{k(\theta,t)}
     \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
     \right\} dt
     \right|
     \f]
  </description>
  ```
  yields:

  Ma description
  \f[
  |I_2|=\left| \int_{0}^T \psi(t)
  \left\{
  u(a,t)-
  \int_{\gamma(t)}^a
  \frac{d\theta}{k(\theta,t)}
  \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi 
  \f]

  whereas:
  ```xml
  <description doc-brief-force-close-cmds="true" doc-brief-max-nb-of-char="120">
    Ma description
    \f[
    |I_2|=\left| \int_{0}^T \psi(t)
    \left\{
    u(a,t)-
    \int_{\gamma(t)}^a
    \frac{d\theta}{k(\theta,t)}
    \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
    \right\} dt
    \right|
    \f]
  </description>
  ```
  yields:

  Ma description
  \f[
  |I_2|=\left| \int_{0}^T \psi(t)
  \left\{
  u(a,t)-
  \int_{\gamma(t)}^a
  \frac{d\theta}{k(\theta,t)}
  \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi
  \right\} dt
  \right|
  \f]

</details>


<details>
  <summary>Attribute doc-brief-stop-at-dot</summary>
  ```xml
  <description doc-brief-stop-at-dot="true"></description>
  ```
  Attribute allowing you to limit the number of characters by cutting the
  short description at the first found period.
  By default, this attribute is set to `true`.
</details>

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_casefile
</span>
</div>
