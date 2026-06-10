# Configuration File {#arcanedoc_examples_simple_example_config}

[TOC]

In addition to the `.arc`, our application needs a `.config` file.
This file allows %Arcane to "see a summary" of our application.
\note
It can also serve code readers by providing a global overview before reading.
\warning
The name of this file must correspond to the project name (with `.config` at the
end).

## HelloWorld.config {#arcanedoc_examples_simple_example_config_helloworldconfig}
```xml
<?xml version="1.0" ?>
<arcane-config code-name="HelloWorld">
  <time-loops>
    <time-loop name="HelloWorldLoop">

      <title>SayHello</title>
      <description>Default timeloop for code HelloWorld</description>

      <modules>
        <module name="SayHello" need="required" />
      </modules>

      <entry-points where="init">
        <entry-point name="SayHello.StartInit" />
      </entry-points>
      <entry-points where="compute-loop">
        <entry-point name="SayHello.Compute" />
      </entry-points>
      <entry-points where="exit">
        <entry-point name="SayHello.EndModule" />
      </entry-points>

    </time-loop>
  </time-loops>
</arcane-config>
```
Once again, there are several things to look at here.
We start with the classic application name on the second line.

Then:
```xml
<time-loop name="HelloWorldLoop">
```
We find the name of the time loop, which must be present in the `.arc` files.

____

```xml
<modules>
  <module name="SayHello" need="required" />
</modules>
```
Here we find our `SayHello` module and specify that it must be present.

____

```xml
<entry-points where="init">
  <entry-point name="SayHello.StartInit" />
</entry-points>
<entry-points where="compute-loop">
  <entry-point name="SayHello.Compute" />
</entry-points>
<entry-points where="exit">
  <entry-point name="SayHello.EndModule" />
</entry-points>
```
If we look back at the `.axl`, we can see that we find our entry points.
Here, we find the entry points for all modules (but since we only have one
module here, we only have the entry points for `SayHello`).
In addition, we use the name "Arcane" and not the method names (capitalized
names).

\remarks
Here is the `.axl` part of the `SayHello` module that we are talking about:

```xml
<entry-points>
  <entry-point method-name="startInit" name="StartInit" where="start-init"
               property="none"/>
  <entry-point method-name="compute" name="Compute" where="compute-loop"
               property="none"/>
  <entry-point method-name="endModule" name="EndModule" where="exit"
               property="none"/>
</entry-points>
```

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_examples_simple_example_arc
</span>
<span class="next_section_button">
\ref arcanedoc_examples_simple_example_main
</span>
</div>
