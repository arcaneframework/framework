# Documentation Customization {#arcanedoc_doc_config}

[TOC]

This page contains several options for customizing the documentation theme.

This new theme is quite different from the old theme (the default Doxygen
theme), so this page allows you to adjust certain elements to have a more
pleasant documentation experience.

Each section of this page is dedicated to one option. Generally, there are three
elements per section: text indicating whether an option is active or not, an
"Enable"/"Disable" button allowing you to activate or deactivate an option, and
a "Test Option" button allowing you to see the option's effect directly.

This page has another particularity: it does not call the already activated
options. This allows you to revert an option to default if there is a problem.
The only way to see the effect of an option on this page is to click the "Test
Option" buttons.

\note
Some of these options are experimental and may have undesirable effects
(effects which are nevertheless specified when they exist).

## Expand Selected Item {#arcanedoc_doc_config_expand_current}

This option allows you to expand the sub-pages of the visited item. If you press
the "Test Option" button, you will see the option's effect in the left
navigation bar.

Once activated, this option will expand the visited item every time, allowing
you to view the content of a chapter directly from the navigation bar.

\htmlonly
<br>
<center>
<span id="span_expand_current_item"></span>
<br>
<button id="button_apply_expand_current_item">Pas de JS</button>
<button id="button_test_expand_current_item">Tester l'option</button>
</center>
\endhtmlonly

## Table of Contents Always Above (Doxygen 1.13.0 and earlier) {#arcanedoc_doc_config_toc_above_all}

This option allows you to display the table of contents above the text. This
saves space on the right, which is reserved by default for the table of
contents.

\warning
If the table of contents is very large, part of the text will be
permanently hidden.

\note If the options \ref arcanedoc_doc_config_toc_above_all and
\ref arcanedoc_doc_config_apply_old_toc are activated together, the option
\ref arcanedoc_doc_config_apply_old_toc will be taken into account.

\htmlonly
<br>
<center>
<span id="span_toc_above_all"></span>
<br>
<button id="button_apply_toc_above_all"></button>
<button id="button_test_toc_above_all">Tester l'option</button>
</center>
\endhtmlonly

## Old Table of Contents Location (Doxygen 1.13.0 and earlier) {#arcanedoc_doc_config_apply_old_toc}

Before the theme update, the table of contents was fixed at the top of the page.
This option allows you to restore the original location of the table of
contents.

\note
If the options \ref arcanedoc_doc_config_toc_above_all and
\ref arcanedoc_doc_config_apply_old_toc are activated together, the option
\ref arcanedoc_doc_config_apply_old_toc will be taken into account.

\htmlonly
<br>
<center>
<span id="span_apply_old_toc"></span>
<br>
<button id="button_apply_apply_old_toc"></button>
<button id="button_test_apply_old_toc">Tester l'option</button>
</center>
\endhtmlonly

## Page Text Width {#arcanedoc_doc_config_edit_max_width}

This option allows you to modify the width dedicated to displaying the text.
This new theme adopts the principles of modern web pages and therefore sets the
maximum width a page can take on a large screen. Since this new display mode
does not suit everyone, this option allows you to change that.

The theme authors set the width to 1050px (pixels). The slider below allows you
to modify this value. After modifying the width, the "Test Option" button allows
you to see what it looks like. The "Save Width" button allows you to save the
modification in the browser's memory. The "100% Screen Width (old theme)" button
allows you to set the width to "100%", which allows you to recover the format of
the original theme.

\htmlonly
<br>
<center>
<span id="span_edit_max_width"></span>
<br>
<input type="range" id="range_edit_max_width" min="500" max="2000" step="100">
<br>
<button id="button_max_edit_max_width">Largeur 100% de l'écran (ancien thème)</button>
<br>
<button id="button_test_edit_max_width">Tester l'option</button>
<button id="button_apply_edit_max_width">Mémoriser la largeur définie</button>
<button id="button_default_edit_max_width">Redéfinir la largeur par défaut</button>
</center>
\endhtmlonly




## Button to Expand Menu {#arcanedoc_doc_config_expand_level_two}

This option allows you to add another button next to the round button at the
bottom of the left menu:

\htmlonly
<center>
<img src="../../sync_on.png" style="transform:rotate(90deg);">
</center>
\endhtmlonly

This button, after activating the option, allows you to expand the menu. This
allows you to get an overview of all the chapters.

\note
This button will not appear on this page; you must go to another page for
it to appear (if the option is activated).

\htmlonly
<br>
<center>
<span id="span_expand_level_two"></span>
<br>
<button id="button_apply_expand_level_two"></button>
<button id="button_test_expand_level_two">Tester l'option</button>
</center>
\endhtmlonly




\htmlonly
<script type="text/javascript">
  updateConfigWithCookies();
  // On this page, customization is disabled.
  no_custom_theme = true;
</script>
\endhtmlonly
