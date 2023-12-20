# Autres élements {#arcanedoc_core_types_axl_others}

[TOC]

## La documentation

Comme vous avez pu le constater précédemment, il y a des champs `<description>`
qui sont disponibles.

Prenons cet exemple :
```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <userclass>User</userclass>
  <description>Descripteur du module Test</description>

  <variables>
    <variable field-name="pressure" name="Pressure" data-type="real" item-kind="cell" dim="0" dump="true" need-sync="true">
      <userclass>User</userclass>
      <description>Descripteur de la variable pressure</description>
    </variable>
  </variables>

  <entry-points>
		<entry-point method-name="testPressureSync" name="TestPressureSync" where="compute-loop" property="none">
      <userclass>User</userclass>
      <description>Descripteur du point d''entrée TestPressureSync</description>
    </entry-point>
  </entry-points>

  <options>
    <simple name = "simple-real" type = "real">
      <name lang='fr'>reel-simple</name>
      <userclass>User</userclass>
      <description>Réel simple</description>
    </simple>
  </options>
</module>
```
TODO `<description doc-brief-force-close-cmds="true" doc-brief-max-nb-of-char="-1" doc-brief-stop-at-dot="false">`

____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_core_types_axl_caseoptions
</span>
<span class="next_section_button">
\ref arcanedoc_core_types_casefile
</span>
</div>
