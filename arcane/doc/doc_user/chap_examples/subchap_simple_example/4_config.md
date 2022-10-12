# Fichier de configuration {#arcanedoc_examples_simple_example_config}

[TOC]

En plus du `.arc`, Notre application a besoin d'un fichier `.config`.
Ce fichier va permettre à %Arcane de "voir un résumé" de notre application.
\note
Ça peut aussi servir aux lecteurs de code pour avoir un aperçu global avant lecture.
\warning
Le nom de ce fichier doit correspondre au nom du projet (avec `.config` à la fin).

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
Encore une fois, il y a plusieurs choses à voir ici.
On commence par le classique nom de l'application à la seconde ligne.

Puis :
```xml
<time-loop name="HelloWorldLoop">
```
On retrouve le nom de la boucle en temps qui doit être présent dans les fichiers `.arc`.

____

```xml
<modules>
  <module name="SayHello" need="required" />
</modules>
```
On retrouve ici notre module `SayHello` et on précise qu'il doit être obligatoirement présent.

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
Si on fait un retour dans le `.axl`, on peut voir qu'on retrouve nos points d'entrées.
Ici, on retrouve les points d'entrées de tous les modules (mais vu que l'on a qu'un seul
module ici, on a que les points d'entrées de `SayHello`).
De plus, on utilise le nom "Arcane" et non le nom des méthodes (la majuscule devant les noms).

\remarks
Voici la partie du `.axl` du module `SayHello` dont on parle ici :
```xml
<entry-points>
  <entry-point method-name="startInit" name="StartInit" where="start-init" property="none" />
  <entry-point method-name="compute" name="Compute" where="compute-loop" property="none" />
  <entry-point method-name="endModule" name="EndModule" where="exit" property="none" />
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