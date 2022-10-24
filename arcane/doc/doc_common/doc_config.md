# Personnalisation de la doc {#arcanedoc_doc_config}

[TOC]

Cette page contient plusieurs options permettant de personnaliser
le thème de la documentation.

Ce nouveau thème est assez différent de l'ancien thème
(le thème par défaut de Doxygen) donc cette page permet
de régler certains éléments de sorte d'avoir une documentation
plus agréable à lire.

Chaque partie de cette page est dédiée à une option.
Globalement, il y a trois éléments par partie :
un texte permettant de savoir si une option est activé
ou non, un bouton "Activer"/"Désactiver" permettant
d'activer ou de désactiver une option et un bouton "Tester l'option"
permettant de voir directement l'effet de l'option.

Cette page possède une autre particularité : elle n'appelle
pas les options déjà activées. Cela permet de pouvoir remettre
une option par défaut s'il y a un problème. Le seul moyen de
voir l'effet d'une option sur cette page est de cliquer sur les
boutons "Tester l'option".

\note
Certaines de ces options sont expérimentales et peuvent
avoir des effets indésirables (effets qui sont néanmoins
précisés lorsqu'il y en a).


## Étendre l'élément séléctionné {#arcanedoc_doc_config_expand_current}

Cette option permet d'étendre les sous-pages de l'élément
visité.
Si vous appuyez sur le bouton "Tester l'option", vous verrez l'effet
de l'option dans la barre de navigation à gauche.

Une fois activée, cette option étendra à chaque fois l'élément
visité, permettant d'avoir une visibilité sur le contenu d'un
chapitre directement depuis la barre de navigation.

\htmlonly
<br>
<center>
<span id="span_expand_current_item"></span>
<br>
<button id="button_apply_expand_current_item">Pas de JS</button>
<button id="button_test_expand_current_item">Tester l'option</button>
</center>
\endhtmlonly


## Table des matières toujours devant {#arcanedoc_doc_config_toc_above_all}

Cette option permet d'afficher la table des matières
par dessus le texte.
Ça permet de gagner de la place à droite, place qui est réservée
par défaut pour la table des matières.

\warning
Si la table des matières est très grande, une partie du texte sera
caché en permanance.

\note
Si les options \ref arcanedoc_doc_config_toc_above_all et
\ref arcanedoc_doc_config_apply_old_toc sont activées ensembles,
c'est l'option \ref arcanedoc_doc_config_apply_old_toc qui
sera prise en compte.

\htmlonly
<br>
<center>
<span id="span_toc_above_all"></span>
<br>
<button id="button_apply_toc_above_all"></button>
<button id="button_test_toc_above_all">Tester l'option</button>
</center>
\endhtmlonly


## Ancien emplacement de la table des matières {#arcanedoc_doc_config_apply_old_toc}

Avant la mise à jour du thème, la table des matières était fixée
en haut de la page. Cette option permet de restaurer l'emplacement
d'origine de la table des matières.

\note
Si les options \ref arcanedoc_doc_config_toc_above_all et
\ref arcanedoc_doc_config_apply_old_toc sont activées ensembles,
c'est l'option \ref arcanedoc_doc_config_apply_old_toc qui
sera prise en compte.

\htmlonly
<br>
<center>
<span id="span_apply_old_toc"></span>
<br>
<button id="button_apply_apply_old_toc"></button>
<button id="button_test_apply_old_toc">Tester l'option</button>
</center>
\endhtmlonly


## Largeur du texte des pages {#arcanedoc_doc_config_edit_max_width}

Cette option permet de modifier la largeur dédiée à l'affichage du texte.
Ce nouveau thème reprend les principes des pages web d'aujourd'hui et
donc fixe la largeur maximale que peut prendre une page sur grand écran.  
Comme ce nouveau mode d'affichage ne convient pas à tout le monde,
cette option permet de modifier cela.

Les auteurs du thèmes ont fixé la largeur à 1050px (pixels).
Le curseur ci-dessous permet de modifier cette valeur.
Après modification de la largeur, le bouton "Tester l'option" permet de voir
ce que ça donne.  
Le bouton "Mémoriser la largeur" permet d'enregistrer
la modification dans la mémoire du navigateur.  
Le bouton "Largeur 100% de l'écran (ancien thème)" permet
de définir la largeur à "100%", ce qui permet de retrouver le format
du thème d'origine.

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





## Bouton pour étendre le menu {#arcanedoc_doc_config_expand_level_two}

Cette option permet d'attribuer une autre fonctionnalité au bouton rond
en bas du menu à gauche :
\image html sync_on.png

Ce bouton, après activation de l'option, permet d'étendre le menu.
Cela permet d'avoir une vision de l'ensemble des chapitres.

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
  // Dans cette page, la personnalisation est désactivée.
  no_custom_theme = true;
</script>
\endhtmlonly
