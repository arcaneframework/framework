# Paralléliser un code {#arcanedoc_parallel}

Si vous souhaiter accélérer votre code, ce chapitre devrait vous intéresser.  
Diveres méthodes sont disponibles dans %Arcane pour permettre d'accélérer un
code : utilisation de tous les coeurs CPU disponibles, utilisation
des unités vectoriels du CPU et utilisation d'accélérateurs (GPU).  
Dans le cas d'un code déséquilibré, il est aussi possible d'utiliser
de l'équilibre de charge, afin de répartir équitablement la charge de calcul
sur tous les sous-domaines.

<br>

Sommaire de ce chapitre :

1. \subpage arcanedoc_parallel_intro <br>
  Introduction au parallélisme introduit dans %Arcane.

2. \subpage arcanedoc_parallel_concurrency <br>
  Présente l'utilisation du multi-threading dans %Arcane (en plus de la
  décomposition de domaine).

3. \subpage arcanedoc_parallel_simd <br>
  Présente les mécanismes disponibles dans %Arcane pour pouvoir utiliser
  les unités vectoriels des CPU d'aujourd'hui.

4. \subpage arcanedoc_parallel_loadbalance <br>
  Décrit l'utilisation du mécanisme d'équilibrage de charge sur le maillage.

5. \subpage arcanedoc_parallel_shmem <br>
   Décrit l'utilisation des fenêtres mémoires en mémoire partagée.


____

<div class="section_buttons">
<span class="next_section_button">
\ref arcanedoc_parallel_intro
</span>
</div>