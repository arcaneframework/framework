# Utilisation pour les matériaux {#arcanedoc_acceleratorapi_materials}

[TOC]

Il est possible d'utiliser l'API accélérateur de %Arcane pour gérer
les constituants. Les instances du gestionnaire de matériaux
\arcanemat{IMeshMaterialMng} utilisent le gestionnaire d'exécution
(\arcaneacc{Runner}) associé au sous-domaine dont est issu le
maillage. La gestion des constituants se fera donc automatiquement
avec le même environnement d'exécution que les opérations sur les
entités classiques du maillage.

L'utilisation de l'API accélérateur pour les matériaux est similaire à
l'utilisation sur les entités du maillage. La macro
RUNCOMMAND_MAT_ENUMERATE() permet d'itérer sur un milieu
\arcanemat{IMeshEnvironment} ou un matériau \arcanemat{IMeshMaterial}.

Les valeurs possibles pour cette macro sont:

<table>
<tr>
<th>Type d'itération</th><th>Valeur de l'itérateur</th>
<th>Type du conteneur</th><th>Description</th>
</tr>

<tr>
<td>EnvAndGlobalCell</td>
<td>\arcanemat{EnvAndGlobalCellIteratorValue}</td>
<td>\arcanemat{IMeshEnvironment} <br></br>
\arcanemat{EnvCellVectorView}</td>
<td>Itération sur un milieu permettant de récupérer pour chaque itération
le numéro local de la maille milieu, l'index de l'itération et le
numéro local de la maille globale associée</td>
</tr>

<tr>
<td>MatAndGlobalCell</td>
<td>\arcanemat{MatAndGlobalCellIteratorValue}</td>
<td>\arcanemat{IMeshMaterial} <br></br> \arcanemat{MatCellVectorView}</td>
<td>Itération sur un matériau permettant de récupérer pour chaque itération
le numéro local de la maille matériau, l'index de l'itération et le
numéro local de la maille globale associée</td>
</tr>

<tr>
<td>AllEnvCell</td>
<td>\arcanemat{AllEnvCell}</td>
<td>\arcanemat{AllEnvCellVectorView}</td>
<td>Itération sur les AllEnvCell</td>
</tr>

<tr>
<td>EnvCell</td>
<td>\arcanemat{EnvCellLocalId}</td>
<td>\arcanemat{IMeshEnvironment} <br></br> \arcanemat{EnvCellVectorView}</td>
<td>Itération sur un milieu permettant de récupérer pour chaque itération
uniquement le numéro local de la maille milieu</td>
</tr>

<tr>
<td>MatCell</td>
<td>\arcanemat{MatCellLocalId}</td>
<td>\arcanemat{IMeshMaterial} <br></br> \arcanemat{MatCellVectorView}</td>
<td>Itération sur un matériau permettant de récupérer pour chaque itération
uniquement le numéro local de la maille matériau</td>
</tr>

</table>

Si on souhaite uniquement accéder aux numéros locaux des mailles
matériaux ou milieux il est préférable pour des raisons de performance
d'utiliser la version avec `EnvCell` ou `MatCell` comme type
d'itérateur.

Voici un exemple de code pour itérer sur une maille milieu avec
l'information de l'index de l'itération et de la maille globale associée

\snippet MeshMaterialAcceleratorUnitTest.cc SampleEnvAndGlobalCell

Voici un autre exemple pour itérer sur les \arcanemat{AllEnvCell} et
récupérer les informations sur les milieux et matériaux présents dans
chaque \arcanemat{AllEnvCell}

\snippet MeshMaterialAcceleratorUnitTest.cc SampleAllEnvCell

La classe \arcanemat{ComponentCellVector} et les classes qui en
dérivent (\arcanemat{MatCellVector} et \arcanemat{EnvCellVector}) sont
supportées par l'API accélérateur et leur création ou modification
utilise l'environnement d'exécution par défaut.

A partir de la version 3.16 de %Arcane, si l'environnement d'exécution
par défaut est un accélérateur, il est possible de spécifier
une politique d'exécution sur l'hôte via la méthode
\arcanemat{IMeshComponent::setSpecificExecutionPolicy()}. Cette
politique sera utilisée pour créer et modifier les instances de
\arcanemat{ComponentCellVector}, \arcanemat{MatCellVector} et
\arcanemat{EnvCellVector}.
____

<div class="section_buttons">
<span class="back_section_button">
\ref arcanedoc_parallel_accelerator
</span>
<span class="next_section_button">
\ref arcanedoc_acceleratorapi_reduction
</span>
</div>
