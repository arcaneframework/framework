# Autres élements {#arcanedoc_core_types_axl_others}

[TOC]

## La documentation {#arcanedoc_core_types_axl_others_doc}

Prenons cet exemple :
```xml
<module name="Test" version="1.0">
  <name lang="fr">Test</name>
  <userclass>User</userclass>
  <description>
    Descripteur du module Test
  </description>

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
Ceci est valable pour les modules et les services.

### Balises <userclass>

Les balises `<userclass>` sont disponibles et permettent de choisir dans quelle version
de la documentation faire apparaitre chaque élément, avec la description correspondante.

Les valeurs possibles sont `User` et `Dev`.

### Balises <description>

Les balises `<description>` sont disponibles pour chaque élément de l'`axl`.
En plus de décrire l'élément pour ceux qui lisent le fichier `axl`, ce champ permet
de générer une documentation pour ce module/service. La génération sera effectuée
par le programme `axldoc`.

Ce programme va générer (entre autres) une page listant tous les services et modules
disponibles (`axldoc_casemainpage.md`, disponible ici : \ref axldoc_casemainpage)
et une page pour le service/module.

\todo Peut-être faire une page pour expliquer comment générer cette documentation
pour un projet utilisant %Arcane.

### Spécificités des balises <userclass> et <description> entre les balises <module> (ou <service>) 
<em>(lignes 3-6 de l'exemple)</em>

Admettons que nous générons la documentation `User`.
Pour que notre module apparaisse dans la liste des modules, il est nécessaire d'ajouter
`<userclass>User</userclass>` entre les balises `<module>` (ou `<service>`).

Sur la page \ref axldoc_casemainpage "axldoc_casemainpage.md", il est possible de voir qu'il y a
des descriptions pour chaque module et service. Il est désormais possible de contrôler la génération
de cette description. Trois attributs sont disponibles pour la balise `<description>` :


<details>
  <summary>Attribut doc-brief-max-nb-of-char</summary>
  ```xml
  <description doc-brief-max-nb-of-char="120"></description>
  ```
  Attribut permettant de limiter le nombre de caractères max de la courte description.
  Par défaut, la limite est définie à `120`. Mettre la valeur `-1` permet
  de désactiver cette limite.
</details>

<details>
  <summary>Attribut doc-brief-force-close-cmds</summary>
  ```xml
  <description doc-brief-force-close-cmds="true"></description>
  ```
  Attribut permettant de forcer `axldoc` à ne pas couper la description lorsque la limite de caractères
  est atteinte dans une commande Doxygen avant la fin de cette commande. C'est un attribut assez utile
  pour, par exemple, ne pas découper une formule LaTeX en plein milieu.

  En revanche, `axldoc` garantie que les balises compatibles sont bien refermées et les referment si 
  ce n'est pas fait.

  Les balises compatibles sont :
  - `\verbatim \endverbatim`
  - `\code \endcode`
  - `\f$ \f$`
  - `\f( \f)`
  - `\f[ \f]`
  - `\f{ \f}`

  Par défaut, cet attribut est défini à `false`.

  Exemple :
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
  donne :

  Ma description
  \f[
  |I_2|=\left| \int_{0}^T \psi(t)
  \left\{
  u(a,t)-
  \int_{\gamma(t)}^a
  \frac{d\theta}{k(\theta,t)}
  \int_{a}^\theta c(\xi)u_t(\xi,t)\,d\xi 
  \f]

  alors que :
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
  donne :

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
  <summary>Attribut doc-brief-stop-at-dot</summary>
  ```xml
  <description doc-brief-stop-at-dot="true"></description>
  ```
  Attribut permettant de limiter le nombre de caractères en coupant la courte description
  au premier point trouvé.
  Par défaut, cet attribut est défini à `true`.
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
