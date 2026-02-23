// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariable.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Interface de la classe Variable.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLE_H
#define ARCANE_CORE_IVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/SerializeGlobal.h"

#include "arcane/utils/Ref.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une variable.
 *
 * L'implémentation de cette interface est la classe Variable.
 *
 * En général cette interface n'est pas utilisée directement. Les variables
 * sont gérées par la classe VariableRef et les classes qui en dérive.
 */
class ARCANE_CORE_EXPORT IVariable
{
 public:

  //! Type des dépendances
  enum eDependType
  {
    DPT_PreviousTime,
    DPT_CurrentTime
  };

 public:

  /*!
   * \brief Propriétés d'une variable.
   */
  enum
  {
    //! Indique que la variable ne doit pas être sauvegardée.
    PNoDump = (1 << 0),

    /*!
     * \brief Indique que la variable n'est pas nécessairement synchronisée.
     *
     * Cela signifie qu'il est normal que les valeurs de la variable soient
     * différentes d'un processeur à l'autre sur les mailles fantômes
     */
    PNoNeedSync = (1 << 1),

    //! Indique que la variable est tracée (uniquement en mode trace)
    PHasTrace = (1 << 2),

    /*! \brief Indique que la valeur de la variable est dépendante du sous-domaine.
     *
     * Cela signifie entre autre que la valeur de la variable est différente
     * dès que le nombre de sous-domaines varie. C'est par exemple le cas
     * de la variable contenant le numéro du sous-domaine propriétaire d'une entité.
     */
    PSubDomainDepend = (1 << 3),

    /*! \brief Indique que la variable est privée au sous-domaine.
     *
     * Cela signifie que la variable est dépendante du sous-domaine et notamment
     * qu'elle n'existe pas forcément sur tout les sous-domaines. Cette
     * propriété ne peut pas être positionnée pour les variables du maillage.
     */
    PSubDomainPrivate = (1 << 4),

    /*! \brief Indique que la valeur de la variable est dépendante de l'exécution
     *
     * Les valeurs de ces variables changent entre deux exécutions. C'est par
     * exemple le cas d'une variable contenant le temps CPU utilisé.
     */
    PExecutionDepend = (1 << 5),

    /*! \brief Indique que la variable est privée
     *
     * Une variable privée ne peut pas posséder plus d'une référence.
     * Cette propriété ne peut être positionner que lors de la création de la
     * variable
     */
    PPrivate = (1 << 6),

    /*! \brief Indique que la variable est temporaire
     *
     * Une variable temporaire est comme son nom l'indique temporaire. Elle
     * ne peut pas être sauvée, n'est pas transférée en cas d'équilibrage
     * du maillage (mais peut être synchronisée) et n'est pas sauvée en
     * cas de retour arrière.
     *
     * Une variable temporaire qui n'est plus utilisée (aucune référence dessus)
     * peut être désallouée.
     */
    PTemporary = (1 << 7),

    /*! \brief Indique que la variable ne doit pas être restaurée.
     *
     * Une variable de ce type n'est pas sauvegardée ni restorée en cas
     * de retour-arrière.
     */
    PNoRestore = (1 << 8),

    /*! \brief Indique que la variable ne doit pas être échangée.
     *
     * Une variable de ce type n'est pas échangée lors d'un repartitionnement
     * de maillage par exemple. Cela permet d'éviter l'envoie de données
     * inutiles si cette variable n'est utilisée que temporairement ou
     * qu'elle est recalculée dans un des points d'entrée appelé
     * suite à un repartitionnement.
     */
    PNoExchange = (1 << 9),

    /*!
     * \brief Indique que la variable est persistante.
     *
     * Une variable persistante n'est pas détruite s'il n'existe plus de référence dessus.
     */
    PPersistant = (1 << 10),

    /*!
     * \brief Indique que la variable n'a pas forcément la même valeur
     * entre les réplicas.
     *
     * Cela signifie qu'il est normal que les valeurs de la variable soient
     * différentes sur les mêmes sous-domaines des autres réplicas.
     */
    PNoReplicaSync = (1 << 11),

    /*!
     * \brief Indique que la variable doit être alloué en mémoire partagée.
     *
     * L'allocateur DynamicMachineMemoryWindowMemoryAllocator sera utilisé.
     * La classe DynamicMachineMemoryWindowVariable pourra être utilisé avec
     * cette variable.
     */
    PInShMem = (1 << 12)
  };

 public:

  //! Tag utilisé pour indiquer si une variable sera post-traitée
  static const char* TAG_POST_PROCESSING;

  //! Tag utilisé pour indiquer si une variable sera post-traitée à cette itération
  static const char* TAG_POST_PROCESSING_AT_THIS_ITERATION;

 public:

  friend class VariableMng;

 public:

  virtual ~IVariable() = default; //!< Libère les ressources

 public:

  //! Sous-domaine associé à la variable (TODO rendre obsolète fin 2023)
  virtual ISubDomain* subDomain() = 0;

 public:

  //! Gestionnaire de variable associé à la variable
  virtual IVariableMng* variableMng() const = 0;

  //! Taille mémoire (en Koctet) utilisée par la variable
  virtual Real allocatedMemory() const = 0;

  //! Nom de la variable
  virtual String name() const = 0;

  //! Nom complet de la variable (avec le préfixe de la famille)
  virtual String fullName() const = 0;

  //! Type de la donnée gérée par la variable (Real, Integer, ...)
  virtual eDataType dataType() const = 0;

  /*!
   * \brief Genre des entités du maillage sur lequel repose la variable.
   *
   * Pour les variables scalaire ou tableau, il n'y a pas de genre et la
   * méthode retourne #IK_Unknown.
   * Pour les autres variables, retourne le genre de l'élément de
   * maillage (Node, Cell, ...), à savoir:
   * - #IK_Node pour les noeuds
   * - #IK_Edge pour les arêtes
   * - #IK_Face pour les faces
   * - #IK_Cell pour les mailles
   * - #IK_Particle pour les particules
   * - #IK_DoF pour les degrés de liberté
   */
  virtual eItemKind itemKind() const = 0;

  /*!
   * \brief Dimension de la variable.
   *
   * Les valeurs possibles sont les suivantes:
   * - 0 pour une variable scalaire,.
   * - 1 pour une variable tableau mono-dim ou variable scalaire du maillage.
   * - 2 pour une variable tableau bi-dim ou variable tableau du maillage.
   */
  virtual Integer dimension() const = 0;

  /*!
   * \brief Indique si la variable est un tableau à taille multiple.
   *
   * Cette valeur n'est utile que pour les tableaux 2D ou plus.
   * - 0 pour une variable scalaire ou tableau 2D standard.
   * - 1 pour une variable tableau 2D à taille multiple.
   * - 2 pour une variable tableau 2D ancient format (obsolète).
   */
  virtual Integer multiTag() const = 0;

  /*!
   * \brief Nombre d'éléments de la variable.
   *
   * Les valeurs retournées dépendent de la dimension de la variable:
   * - pour une dimension 0, retourne 1,
   * - pour une dimension 1, retourne le nombre d'éléments du tableau
   * - pour une dimension 2, retourne le nombre total d'éléments en sommant
   * le nombre d'éléments par dimension.
   */
  virtual Integer nbElement() const = 0;

  //! Retourne les propriétés de la variable
  virtual int property() const = 0;

  //! Indique que les propriétés d'une des références à cette variable ont changé (interne)
  virtual void notifyReferencePropertyChanged() = 0;

  /*!
   * \brief Ajoute une référence à cette variable
   *
   * \pre \a var_ref ne doit pas déjà référencer une variable.
   */
  virtual void addVariableRef(VariableRef* var_ref) = 0;

  /*!
   * \brief Supprime une référence à cette variable
   *
   * \pre \a var_ref doit référencer cette variable (un appel à addVariableRef()
   * doit avoir été effectué sur cette variable).
   */
  virtual void removeVariableRef(VariableRef* var_ref) = 0;

  //! Première réference (ou null) sur cette variable
  virtual VariableRef* firstReference() const = 0;

  //! Nombre de références sur cette variable
  virtual Integer nbReference() const = 0;

 public:

  ARCANE_DEPRECATED_REASON("Y2021: This method is a noop")
  virtual void setTraceInfo(Integer id, eTraceType tt) = 0;

 public:

  /*!
   * \brief Positionne le nombre d'éléments pour une variable tableau.
   *
   * Lorsque la variable est du type tableau 1D ou 2D, positionne le nombre
   * d'éléments du tableau à \a new_size. Pour un tableau 2D, c'est le
   * nombre d'éléments de la première dimension qui est modifié.
   *
   * Cette opération ne doit pas être appelée pour les variables du maillage
   * car le nombre d'éléments est déterminé automatiquement en fonction du nombre
   * d'entités du groupe sur lequel elle s'appuie. Pour ce type de variable,
   * il faut appeler resizeFromGroup().
   *
   * Cette opération synchronise les références (syncReferences()).
   */
  virtual void resize(Integer new_size) = 0;

  /*!
   * \brief Positionne le nombre d'éléments pour une variable du maillage.
   *
   * Réalloue la taille de la variable du maillage à partir du groupe
   * sur laquelle elle s'appuie.
   *
   * Cette opération n'a d'effet que pour les variables du maillage.
   * Pour les autres, aucun action n'est effectuée.
   *
   * Cette opération synchronise les références (syncReferences()).
   */
  virtual void resizeFromGroup() = 0;

  /*!
   * \brief Libère l'éventuelle mémoire supplémentaire allouée pour
   * les données.
   *
   * Cette méthode n'est utilie que pour les variables non scalaires
   */
  virtual void shrinkMemory() = 0;

  //! Positionne les informations sur l'allocation
  virtual void setAllocationInfo(const DataAllocationInfo& v) = 0;

  //! Informations sur l'allocation
  virtual DataAllocationInfo allocationInfo() const = 0;

 public:

  /*!
   * \brief Initialise la variable sur un groupe.
   *
   * Initialise la variable avec la valeur \a value pour tous les éléments du
   * groupe \a group.
	 *
   * Cette opération n'est utilisable qu'avec les variables de maillage.
	 *
   * \param group_name groupe. Il doit correspondre à un groupe existant
   * du type de la variable (par exemple CellGroup pour une variable au maille).
   * \param value valeur d'initialisation. La chaîne doit pouvoir être convertie
   * en le type de la variable.
   *
   * \retval true en cas d'erreur ou si la variable n'est pas une variable du
   * maillage.
   * \retval false si l'initialisation est un succès.
  */
  virtual bool initialize(const ItemGroup& group, const String& value) = 0;

  //! @name Opérations de vérification
  //@{
  /*! \brief Vérifie si la variable est bien synchronisée.
   *
   * Cette opération ne fonctionne que pour les variables de maillage.
   *
   * Un variable est synchronisée lorsque ses valeurs sont les mêmes
   * sur tous les sous-domaines à la fois sur les éléments propres et
   * les éléments fantômes.
   *
   * Pour chaque élément non synchronisé, un message est affiché.
   * 
   * \param max_print nombre maximum de messages à afficher.
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a max_print élément. Si négatif, tous les éléments sont affichés.
   *
   * \return le nombre de valeurs différentes de la référence
   */
  virtual Int32 checkIfSync(Integer max_print = 0) = 0;

  /*! \brief Vérifie que la variable est identique à une valeur de référence
   *
   * Cette opération vérifie que les valeurs de la variable sont identique
   * à une valeur de référence qui est lu à partir du lecteur \a reader.
   *
   * Pour chaque valeur différente de la référence, un message est affiché.
   *
   * \param max_print nombre maximum de messages à afficher.
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a max_print élément. Si négatif, tous les éléments sont affichés.
   * \param compare_ghost si vrai, compare les valeurs à la fois sur les entités
   * propres et les entités fantômes. Sinon, ne fait la comparaison que sur les
   * entités propres.
   *
   * \return le nombre de valeurs différentes de la référence
   */
  virtual Int32 checkIfSame(IDataReader* reader, Integer max_print, bool compare_ghost) = 0;

  /*!
   * \brief Vérifie si la variable a les mêmes valeurs sur tous les réplicas.
   *
   * Compare les valeurs de la variable avec celle du même sous-domaine
   * des autres réplicas. Pour chaque élément différent,
   * un message est affiché.
   *
   * Cette méthode est collective sur le même sous-domaine des autres réplica.
   * Il ne faut donc l'appeler que si la variable existe sur tous les sous-domaines
   * sinon cela provoque un blocage.
   *
   * Cette méthode ne fonctionne que pour les variables sur les types numériques.
   * Dans ce cas, elle renvoie une exception de type NotSupportedException.
   *
   * \param max_print nombre maximum de messages à afficher.
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a max_print élément. Si négatif, tous les éléments sont affichés.
   * Pour chaque élément différent est affiché la valeur minimale et
   * maximale.
   *
   * \return le nombre de valeurs différentes de la référence.
   */
  virtual Int32 checkIfSameOnAllReplica(Integer max_print = 0) = 0;
  //@}

  /*!
   * \brief Synchronise la variable.
   *
   La synchronisation ne peut se faire que sur les variables du maillage.
   */
  virtual void synchronize() = 0;

  // TODO: à rendre virtuelle pure (décembre 2024)
  /*!
   * \brief Synchronise la variable sur une liste d'entités.
   *
   * La synchronisation ne peut se faire que sur les variables du maillage.
   * Seules les entités listées dans \a local_ids seront synchronisées. Attention :
   * une entité présente dans cette liste sur un sous-domaine doit être présente
   * dans cette liste pour tout autre sous-domaine qui possède cette entité.
   */
  virtual void synchronize(Int32ConstArrayView local_ids);

  /*!
   * \brief Maillage auquel est associé la variable.
   *
   * Cette opération n'est significative que pour les variables sur des
   * entités du maillage.
   */
  ARCCORE_DEPRECATED_2020("Use meshHandle() instead")
  virtual IMesh* mesh() const = 0;

  /*!
   * \brief Maillage auquel est associé la variable.
   *
   * Cette opération n'est significative que pour les variables sur des
   * entités du maillage.
   */
  virtual MeshHandle meshHandle() const = 0;

  /*!
   * \brief Groupe du maillage associé.
   *
   * \return le groupe du maillage associé si pour une variable du maillage
   * ou le groupe nul si la variable n'est pas une variable du maillage.
   *
   * Si une variable n'est pas utilisée ou pas encore allouée,
   * la valeur retournée est le group nul.
   * Cependant, la variable peut quand même être associée à un groupe.
   * Dans ce cas, il faut utiliser la fonction itemGroupName() pour
   * récupérer le nom de ce groupe.
   */
  virtual ItemGroup itemGroup() const = 0;

  //! Nom du groupe d'entité associée.
  virtual String itemGroupName() const = 0;

  /*!
   * \brief Famille d'entité associée.
   *
   * \return la famille associée à la variable ou 0
   * si la variable n'a pas de famille.
   *
   * Si une variable n'est pas utilisée ou pas encore allouée,
   * la valeur retournée est nulle.
   * Cependant, la variable peut quand même être associée à une famille.
   * Dans ce cas, il faut utiliser la fonction itemFamilyName() pour
   * récupérer le nom de cette famille.
   */
  virtual IItemFamily* itemFamily() const = 0;

  //! Nom de la famille associée (nul si aucune).
  virtual String itemFamilyName() const = 0;

  //! Nom du maillage associé (nul si aucun).
  virtual String meshName() const = 0;

  /*!
   * \brief Créé une instance contenant les meta-données de la variable.
   *
   * L'instance retournée doit être détruite par l'appel à l'opérateur delete.
   */
  ARCANE_DEPRECATED_REASON("Y2024: Use createMetaDataRef() instead")
  virtual VariableMetaData* createMetaData() const = 0;

  //! Créé une instance contenant les meta-données de la variable.
  virtual Ref<VariableMetaData> createMetaDataRef() const = 0;

  /*!
   * \brief Synchronise les références.
   *
   * Synchronise les valeurs des références (VariableRef) à cette variable
   * avec la valeur actuelle de la variable. Cette méthode est appelé
   * automatiquement lorsqu'une variable scalaire est modifiée ou
   * le nombre d'éléments d'une variable tableau change.
   */
  virtual void syncReferences() = 0;

 public:

  /*!
   * \brief Positionne l'état d'utilisation de la variable
   *
   * Si \v est faux, la variable devient inutilisable
   * et toutes les ressources associées sont libérées.
   *
   * Si \v est vrai, la variable est considérée comme utilisée et s'il s'agit
   * d'une variable du maillage et que setItemGroup() n'a pas été appelé, la
   * variable est allouée sur le groupe de toutes les entités.
   */
  virtual void setUsed(bool v) = 0;

  //! Etat d'utilisation de la variable
  virtual bool isUsed() const = 0;

  /*!
   * \brief Indique si la variable est partielle.
   *
   * Une variable est partielle lorsqu'elle n'est pas définie sur toutes les
   * entités d'une famille. Dans ce cas, group()!=itemFamily()->allItems().
   */
  virtual bool isPartial() const = 0;

 public:

  /*!
   * \brief Copie les valeurs des entités numéros @a source dans les entités
   * numéro @a destination
   * 
   * @note Cette opération est interne à Arcane et doit se faire en
   * conjonction avec la famille d'entité correspondant à cette
   * variable.
   * 
   * @param source liste des @b localId source
   * @param destination liste des @b localId destination
   */
  virtual void copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination) = 0;

  /*!
   * \brief Copie les moyennes des valeurs des entités numéros
   * @a first_source et @a second_source dans les entités numéros
   * @a destination
   * 
   * @param first_source liste des @b localId de la 1ère source
   * @param second_source  liste des @b localId de la 2ème source
   * @param destination  liste des @b localId destination
   */
  virtual void copyItemsMeanValues(Int32ConstArrayView first_source,
                                   Int32ConstArrayView second_source,
                                   Int32ConstArrayView destination) = 0;

  /*!
   * \brief Compacte les valeurs de la variable.
   *
   * Cette opération est interne à Arcane et doit se faire en
   * conjonction avec la famille d'entité correspondant à cette
   * variable.
   */
  virtual void compact(Int32ConstArrayView new_to_old_ids) = 0;

  //! pH : EXPERIMENTAL
  virtual void changeGroupIds(Int32ConstArrayView old_to_new_ids) = 0;

 public:

  //! Données associées à la variable
  virtual IData* data() = 0;

  //! Données associées à la variable
  virtual const IData* data() const = 0;

  //! Fabrique de données associées à la variable
  virtual IDataFactoryMng* dataFactoryMng() const = 0;

  //! @name Opérations de sérialisation
  //@{
  /*! Sérialize la variable.
   *
   * L'opération \a opération n'est significative qu'en lecture (ISerializer::ModeGet)
   */
  virtual void serialize(ISerializer* sbuffer, IDataOperation* operation = 0) = 0;

  /*!
   * \brief Sérialize la variable pour les identifiants \a ids.
   *
   * La sérialisation dépend de la dimension de la variable.
   * Pour les variables scalaires (dimension=0), rien n'est fait.
   * Pour les variables tableaux ou du maillage, \a ids correspond a un tableau
   * d'indirection de la première dimension.
   *
   * L'opération \a opération n'est significative qu'en lecture (ISerializer::ModeGet)
   */
  virtual void serialize(ISerializer* sbuffer, Int32ConstArrayView ids, IDataOperation* operation = 0) = 0;

  /*!
   * \brief Sauve la variable
   *
   * \deprecated A remplacer par le code suivant:
   * \code
   * IVariable* var;
   * var->notifyBeginWrite();
   * writer->write(var,var->data());
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void write(IDataWriter* writer) = 0;

  /*!
   * Relit la variable.
   *
   * \deprecated A remplacer par le code suivant:
   * \code
   * IVariable* var;
   * reader->read(var,var->data());
   * var->notifyEndRead();
   * \endcode
   */
  virtual ARCANE_DEPRECATED_2018 void read(IDataReader* reader) = 0;

  /*!
   * \brief Notifie de la modification externe de data().
   *
   * Signale à l'instance la fin d'une opération de lecture qui a modifié
   * data(). Cette méthode doit donc être appelée dès qu'on a effectué
   * une modication de data(). Cette méthode déclenche les observables enregistrés
   * dans readObservable().
   */
  virtual void notifyEndRead() = 0;

  /*!
   * \brief Notifie du début d'écriture de data().
   *
   * Cette méthode déclenche les observables enregistrés
   * dans writeObservable().
   */
  virtual void notifyBeginWrite() = 0;

  /*!
   * \brief Observable en écriture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * avant d'écrire la variable (opération write()).
   */
  virtual IObservable* writeObservable() = 0;

  /*! \brief Observable en lecture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * après avoir lu la variable (opération read).
   */
  virtual IObservable* readObservable() = 0;

  /*! \brief Observable en redimensionnement.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * lorsque le nombre d'éléments de la variable change.
   * C'est le cas par exemple après un remaillage pour une variable aux mailles
   */
  virtual IObservable* onSizeChangedObservable() = 0;
  //@}

  //@{ @name Gestion des tags
  //! Ajoute le tag \a tagname avev la valeur \a tagvalue
  virtual void addTag(const String& tagname, const String& tagvalue) = 0;
  /*! \brief Supprime le tag \a tagname
   *
   * Si le tag \a tagname n'est pas dans la liste, rien ne se passe.
   */
  virtual void removeTag(const String& tagname) = 0;
  //! \a true si la variable possède le tag \a tagname
  virtual bool hasTag(const String& tagname) = 0;
  //! Valeur du tag \a tagname. La chaîne est nulle si le tag n'existe pas.
  virtual String tagValue(const String& tagname) = 0;
  //@}

 public:

  //! Imprime les valeurs de la variable sur le flot \a o
  virtual void print(std::ostream& o) const = 0;

 public:

  //! @name Gestion des dépendances
  //@{
  /*!
   * \brief Recalcule la variable si nécessaire
   *
   * Par le mécanisme de dépendances, cette opération est appelée récursivement
   * sur toutes les variables dont dépend l'instance. La fonction de recalcul
   * computeFunction() est ensuite appelée s'il s'avère qu'une des variables
   * dont elle dépend a été modifiée plus récemment.
   *
   * \pre computeFunction() != 0
   */
  virtual void update() = 0;

  virtual void update(Real wanted_time) = 0;

  /*! \brief Indique que la variable vient d'être mise à jour.
   *
   * Pour une gestion correcte des dépendances, il faut que cette propriété
   * soit appelée toutes les fois où la mise à jour d'une variable a été
   * effectuée.
   */
  virtual void setUpToDate() = 0;

  //! Temps auquel la variable a été mise à jour
  virtual Int64 modifiedTime() = 0;

  //! Ajoute \a var à la liste des dépendances
  virtual void addDepend(IVariable* var, eDependType dt) = 0;

  //! Ajoute \a var à la liste des dépendances avec les infos de trace \a tinfo
  virtual void addDepend(IVariable* var, eDependType dt, const TraceInfo& tinfo) = 0;

  /*! \brief Supprime \a var de la liste des dépendances
   */
  virtual void removeDepend(IVariable* var) = 0;

  /*!
   * \brief Positionne la fonction de recalcul de la variable.
   *
   * La fonction spécifiée \a v doit être allouée via l'opérateur new.
   * Si une fonction de recalcule existait déjà, elle est détruite
   * (via l'opérateur delete) et remplacée par celle-ci.
   */
  virtual void setComputeFunction(IVariableComputeFunction* v) = 0;

  //! Fonction utilisée pour mettre à jour la variable
  virtual IVariableComputeFunction* computeFunction() = 0;

  /*!
   * \brief Infos de dépendances.
   *
   * Remplit le tableau \a infos avec les infos de dépendance.
   */
  virtual void dependInfos(Array<VariableDependInfo>& infos) = 0;
  //@}

 public:

  ARCANE_DEPRECATED_REASON("Y2021: This method is a noop")
  virtual IMemoryAccessTrace* memoryAccessTrace() const = 0;

  /*!
   * \brief Indique que la variable est synchronisée.
   *
   * Cette opération est collective.
   */
  virtual void setIsSynchronized() = 0;

  /*!
   * \brief Indique que la variable est synchronisée sur le group \a item_group
   *
   * Cette opération est collective.
   */
  virtual void setIsSynchronized(const ItemGroup& item_group) = 0;

 public:

  //! Incrémente le compteur de modification et retourne sa valeur avant modification
  static Int64 incrementModifiedTime();

 public:

  //! API interne à Arcane
  virtual IVariableInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
