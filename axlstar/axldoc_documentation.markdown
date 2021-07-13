Axldoc evolution   {#mainpage}
================

Description des évolutions pour axldoc (SDC)
-------------------------------------

Test au passage de l'utilisation du format markdown pour intégration dans doxygen. 

Les nouvelles fonctionalités sont les suivantes:

- Ajout d'un outillage pour la génération d'un schéma xsd de l'application :
    - génère un xsd avec une collectino de type pour les services : l'association entre une balise de service et  son type n'est pas automatique à la validation et doit être précisé dans le .arc, balise xsi:type  dans balise service), 
(semble difficile à faire automatiquement, cf. types de services non discernables). Ex :  
      <pre>&ltformat name="IXM4Writer" xsi:type="Arcane-IPostProcessorWriter_IXM4Writer-type"&gt</pre>
      @IFPEN nous avons scripté l'ajout de l'info dans le xsd pour permettre la validation, ie ajout du type xsi:type dans les balises services d'un .arc existant.
    
    - Comme le xsd ne permet pas de définir un ordre aléatoire dans les options (en 1.0) , par défaut il impose l'ordre alphabétique dans les balises. P
Pour pouvoir imposer un ordre différent on peut utiliser une balise <option-index> dans la balise <description> des axl (todo pouvoir conserver l'ordre dans lequel sont définies les options)
    - Au niveau code, ajout des classes XmlSchemaFile et XmlSchemaVisitor.
    - Le branchement s'effectue par l'option --generate-xsd (l'ajout d'info dans le xsd, cf les options possibles est utilisé @IFPEN pour ajouter dans des .arc existant l'info de type dans les balises services et pouvoir ainsi faire de la validation).
		
- Ajout de lien retour d'un service vers les modules qui l'utilisent (qui déclarent son interface en option)

- Ces différentes modifs ont nécessité l'enrichissement de l'axl\_db_file (todo : faire un schéma xsd ?)
	  - ajout attribut application-name dans la balise root (pour le branchement applicatif)
	  - balise has-service-instance : pour lien retour (depuis une page de service, on voit toutes les occurrence de l'utilisation de son interface)
	  - balise alias : pour la génération du xsd (parfois les services ont plusieurs noms et ça pose problème pour valider) (=> donner un ex)


- Ajout de la possibilité de générer des pages propres aux applications via un branchement plugin applicatif (interface IPrivateApplicationPages)

- Modification cosmétique : séparateur entre option (pour mieux séparer les options complexes et les autres) et une intro avant la table des services
