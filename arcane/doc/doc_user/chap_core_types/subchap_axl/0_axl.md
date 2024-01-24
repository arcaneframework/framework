# Descripteur de module/service (.AXL) {#arcanedoc_core_types_axl}

Comme expliqué précédemment, le descripteur de module/service est un fichier
ayant l'extension `.axl` qui accompagne les modules et les services et qui
décrit les éléments suivants :
- là où les interfaces qu'il implémente (**service uniquement**),
- ses variables,
- ses points d'entrée (**module uniquement**),
- ses options de configuration.

Ce sous-chapitre va donc rentrer dans les détails et expliquer ces trois
principes essentiels dans %Arcane.

Pour rappel, un descripteur de module ressemble à ça :

```xml
<?xml version="1.0"?>
<module name="Hydro" version="1.0">
	<description>Descripteur du module Hydro</description>

	<variables>
    <!-- Voir partie "Variable". -->
	</variables>
	<entry-points>
    <!-- Voir partie "Point d'entrée". -->
	</entry-points>
	<options>
    <!-- Service de type IEquationOfState. -->
    <!-- Voir partie "Options". -->
	</options>
</module>
```

Et un descripteur de service ressemble à ça :

```xml
<?xml version="1.0"?>
<service name="PerfectGasEOS" version="1.0">
  <description>Descripteur du service PerfectGasEOSService</description>

  <interface name="IEquationOfState" />

	<variables>
    <!-- Voir partie "Variable". -->
	</variables>
	<options>
    <!-- Voir partie "Options". -->
	</options>
</service>
```

<br>

Sommaire de ce sous-chapitre :

1. \subpage arcanedoc_core_types_axl_variable <br>
  Présente la notion de variable dans %Arcane.

2. \subpage arcanedoc_core_types_axl_entrypoint <br>
  Présente la notion de point d'entrée dans %Arcane.

3. \subpage arcanedoc_core_types_axl_caseoptions <br>
  Explique comment paramétrer les modules avec des options utilisateurs fournies 
  dans le jeu de données.

4. \subpage arcanedoc_core_types_axl_others <br>
  Présente les détails qui n'apparaissent pas dans les
  autres sous-chapitres.

____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_core_types_axl_variable
</span>
</div>
