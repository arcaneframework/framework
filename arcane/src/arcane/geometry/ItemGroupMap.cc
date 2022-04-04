﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "arcane/geometry/ItemGroupMap.h"

/* TODO LIST (?? et en vrac)
 * - Ajouter un methode isChanged 'intelligente' (ie si le groupe change effectivement).
 *   genre eInvalidateOnResize + méthodes invalidate et isInvalidated
 * - Optimiser resizeFromGroup quand la taille (le contenu du groupe ?) ne change pas
 * - Optimiser compact quand rien ne change
 * - Ajouter une proprité (à la création) pour ne pas l'utiliser comme une IVariable (no redimensionnement & co)
 * - Gestion des dépendances comme les IVariable classiques ?
 * - Tester en parallèle
 * - Déplacer les méthodes resizeFromGroup, compact voir même init dans le .cc (attn template)
 */

/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/MemoryAccessInfo.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ItemGroupObserver.h"

//%% ARCANE_BENCH_SUPPRESS_BEGIN
#include "arcane/expr/Expression.h"
#include "arcane/VariableExpressionImpl.h"
//%% ARCANE_BENCH_SUPPRESS_END
#include "arcane/Variable.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ISubDomain.h"
#include "arcane/VariableInfo.h"
#include "arcane/ISerializer.h"
#include "arcane/VariableBuildInfo.h"
#include "arcane/VariableComputeFunction.h"
#include "arcane/CommonVariables.h"
#include "arcane/IObservable.h"
#include "arcane/IVariableMng.h"
#include <arcane/ArcaneVersion.h>

#if 0

/*---------------------------------------------------------------------------*/

extern "C" ISubDomain* _arcaneGetDefaultSubDomain();

/*---------------------------------------------------------------------------*/

class ItemGroupMapAbstractVariable
: public IVariable
{
 private:
  ItemGroupMapAbstract* m_map;
  String m_name;
  IVariableMng* m_variable_mng;
  ITraceMng* m_trace_mng;
  IVariableComputeFunction* m_compute_function;
  bool m_used;

  static Int64 m_unique_id;

 public:

  ItemGroupMapAbstractVariable(ItemGroupMapAbstract * map);
 public:
  virtual ~ItemGroupMapAbstractVariable(); //!< Libère les ressources

 public:

  //! Sous-domaine associé à la variable
  virtual ISubDomain* subDomain();

  //! Taille mémoire (en Koctet) utilisée par la variable
  virtual Real allocatedMemory() const;

  //! Nom de la variable
  virtual const String& name() const;

  //! Nom complet de la variable (avec le préfixe de la famille)
  virtual const String& fullName() const;

  //! Type de la donnée gérée par la variable (Real, Integer, ...)
  virtual eDataType dataType() const;

  /*! \brief Type des entités du maillage sur lequel repose la variable.
   *
   Pour les variables scalaire ou tableau, il n'y a pas de type et la
   méthode retourne #IK_Unknown.
   Pour les autres variables, retourne le type de l'élément de
   maillage (Node, Cell, ...), à savoir:
   - #IK_Node pour les noeuds
   - #IK_Edge pour les arêtes
   - #IK_Face pour les faces
   - #IK_Cell pour les mailles
   - #IK_DualNode pour les noeuds duals
   - #IK_Link pour les liens du graphe
   - #IK_Particle pour les particules
  */
  virtual eItemKind itemKind() const;
 
  /*!
    \brief Dimension de la variable.
    
    Les valeurs possibles sont les suivantes:
    - 0 pour une variable scalaire,.
    - 1 pour une variable tableau mono-dim ou variable scalaire du maillage.
    - 2 pour une variable tableau bi-dim ou variable tableau du maillage.
  */
  virtual Integer dimension() const;

  /*!
    \brief Indique si la variable est un tableau à taille multiple.
    
    Cette valeur n'est utile que pour les tableaux 2D ou plus.
    - 0 pour une variable scalaire ou tableau 2D standard.
    - 1 pour une variable tableau 2D à taille multiple.
    - 2 pour une variable tableau 2D ancient format (obsolète).
  */
  virtual Integer multiTag() const;

  /*!
    \brief Nombre d'éléments de la variable.
    
    Les valeurs retournées dépendent de la dimension de la variable:
    - pour une dimension 0, retourne 1,
    - pour une dimension 1, retourne le nombre d'éléments du tableau
    - pour une dimension 2, retourne le nombre total d'éléments en sommant
    le nombre d'éléments par dimension.
  */
  virtual Integer nbElement() const;

  //! Retourne les propriétés de la variable
  virtual int property() const;

  //! Indique que les propriétés d'une des références à cette variable ont changé (interne)
  virtual void notifyReferencePropertyChanged();

  /*! \brief Ajoute une référence à cette variable
   *
   * \pre \a var_ref ne doit pas déjà référencer une variable.
   */
  virtual void addVariableRef(VariableRef* var_ref);

  /*! \brief Supprime une référence à cette variable
   *
   * \pre \a var_ref doit référencer cette variable (un appel à addVariableRef()
   * doit avoir été effectué sur cette variable).
   */
  virtual void removeVariableRef(VariableRef* var_ref);

  //! Nombre de références sur cette variable
  virtual Integer nbReference() const;

 public:

//%% ARCANE_BENCH_SUPPRESS_BEGIN
  virtual Expression expression();

 public:
//%% ARCANE_BENCH_SUPPRESS_END

 public:
 
  virtual void setTraceInfo(Integer id,eTraceType tt);

 public:

  virtual VariableRef* firstReference() const;
  virtual const String& meshName() const;

 public:

  /*!
    \brief Positionne le nombre d'éléments pour une variable tableau.
    
    Lorsque la variable est du type tableau 1D ou 2D, positionne le nombre
    d'éléments du tableau à \a new_size. Pour un tableau 2D, c'est le
    nombre d'éléments de la première dimension qui est modifié.

    Cette opération ne doit pas être appelée pour les variables du maillage
    car le nombre d'éléments est déterminé automatiquement en fonction du nombre
    d'entités du groupe sur lequel elle s'appuie. Pour ce type de variable,
    il faut appeler resizeFromGroup().
    
    Cette opération synchronise les références (syncReferences()).
  */
  virtual void resize(Integer new_size);

  /*!
    \brief Positionne le nombre d'éléments pour une variable du maillage.
    
    Réalloue la taille de la variable du maillage à partir du groupe
    sur laquelle elle s'appuie.

    Cette opération n'a d'effet que pour les variables du maillage.
    Pour les autres, aucun action n'est effectuée.

    Cette opération synchronise les références (syncReferences()).
  */
  virtual void resizeFromGroup();

 public:

  /*!
   * \brief Initialise la variable sur un groupe.
   *
   Initialise la variable avec la valeur \a value pour tous les éléments du
   groupe \a group.
	 
   Cette opération n'est utilisable qu'avec les variables de maillage.
	 
   \param group_name groupe. Il doit correspondre à un groupe existant
   du type de la variable (par exemple CellGroup pour une variable au maille).
   \param value valeur d'initialisation. La chaîne doit pouvoir être convertie
   en le type de la variable.

   \retval true en cas d'erreur ou si la variable n'est pas une variable du
   maillage.
   \retval false si l'initialisation est un succès.
  */
  virtual bool initialize(const ItemGroup& group,const String& value);


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
  virtual Integer checkIfSync(int max_print=0);

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
  virtual Integer checkIfSame(IDataReader* reader,int max_print,bool compare_ghost);
  //@}

  /*!
   * \brief Synchronise la variable.
   *
   La synchronisation ne peut se faire que sur les variables du maillage.
   */
  virtual void synchronize();

  /*! \brief Maillage auquel est associé la variable.
   *
   * Cette opération n'est significative que pour les variables sur des
   * entités du maillage.
   */
  virtual IMesh* mesh() const;
  
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
  virtual ItemGroup itemGroup() const;

  //! Nom du groupe d'entité associée.
  virtual const String& itemGroupName() const;

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
  virtual IItemFamily* itemFamily() const;

  //! Nom de la famille associée.
  virtual const String& itemFamilyName() const;

  /*! \brief Synchronise les références.
   *
   * Synchronise les valeurs des références (VariableRef) à cette variable
   * avec la valeur actuelle de la variable. Cette méthode est appelé
   * automatiquement lorsqu'une variable scalaire est modifiée ou
   * le nombre d'éléments d'une variable tableau change.
   */
  virtual void syncReferences();

 public:
	
  /*! \brief Positionne l'état d'utilisation de la variable
   *
   * Si \v est faux, la variable devient inutilisable
   * et toutes les ressources associées sont libérées.
   *
   * Si \v est vrai, la variable est considérée comme utilisée et s'il s'agit
   * d'une variable du maillage et que setItemGroup() n'a pas été appelé, la
   * variable est allouée sur le groupe de toutes les entités.
   */
  virtual void setUsed(bool v);

  //! Etat d'utilisation de la variable
  virtual bool isUsed() const;


  /*! \brief Indique si la variable est partielle.
   *
   * Une variable est partielle lorsqu'elle n'est pas définie sur toutes les
   * entités d'une famille. Dans ce cas, group()!=itemFamily()->allItems().
   */
  virtual bool isPartial() const;
  
 public:

  /** 
   * Copie les valeurs des entités numéros @a source dans les entités
   * numéro @a destination
   * 
   * @note Cette opération est interne à Arcane et doit se faire en
   * conjonction avec la famille d'entité correspondant à cette
   * variable.
   * 
   * @param source liste des @b localId source
   * @param destination liste des @b localId destination
   */
  virtual void copyItemsValues(Int32ConstArrayView source,Int32ConstArrayView destination);

  /** 
   * Copie les moyennes des valeurs des entités numéros
   * @a first_source et @a second_source dans les entités numéros
   * @a destination
   * 
   * @param first_source liste des @b localId de la 1ère source
   * @param second_source  liste des @b localId de la 2ème source
   * @param destination  liste des @b localId destination
   */
  virtual void copyItemsMeanValues(Int32ConstArrayView first_source,
                                   Int32ConstArrayView second_source,
                                   Int32ConstArrayView destination);

  /*! \brief Compacte les valeurs de la variable.
   *
   * Cette opération est interne à Arcane et doit se faire en
   * conjonction avec la famille d'entité correspondant à cette
   * variable.
   */
  virtual void compact(Int32ConstArrayView new_to_old_ids);

  virtual void changeGroupIds(Int32ConstArrayView old_to_new_ids);

 public:

  //! Données associées à la variable
  virtual IData* data();

  //! @name Opérations de sérialisation
  //@{
  /*! Sérialize la variable.
   *
   * L'opération \a opération n'est significative qu'en lecture (ISerializer::ModeGet)
   */
  virtual void serialize(ISerializer* sbuffer,IDataOperation* operation=0);

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
  virtual void serialize(ISerializer* sbuffer,Int32ConstArrayView ids,IDataOperation* operation=0);

  //! Sauve la variable
  virtual void write(IDataWriter*);

  //! Relit la variable
  virtual void read(IDataReader*);

  /*! \brief Observable en écriture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * avant d'écrire la variable (opération write()).
   */
  virtual IObservable* writeObservable();

  /*! \brief Observable en lecture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * après avoir lu la variable (opération read).
   */
  virtual IObservable* readObservable();

  /*! \brief Observable en redimensionnement.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * lorsque le nombre d'éléments de la variable change.
   * C'est le cas par exemple après un remaillage pour une variable aux mailles
   */
  virtual IObservable* onSizeChangedObservable();
  //@}
  
  //@{ @name Gestion des tags
  //! Ajoute le tag \a tagname avev la valeur \a tagvalue
  virtual void addTag(const String& tagname,const String& tagvalue);
  /*! \brief Supprime le tag \a tagname
   *
   * Si le tag \a tagname n'est pas dans la liste, rien ne se passe.
   */  
  virtual void removeTag(const String& tagname);
  //! \a true si la variable possède le tag \a tagname
  virtual bool hasTag(const String& tagname);
  //! Valeur du tag \a tagname. La chaîne est nulle si le tag n'existe pas.
  virtual String tagValue(const String& tagname);
  //@}

  virtual VariableMetaData* createMetaData() const;
  virtual void notifyEndRead();
  virtual void notifyBeginWrite();
  virtual Integer checkIfSameOnAllReplica(Integer max_print=0);

 public:
 
  //! Imprime les valeurs de la variable sur le flot \a o
  virtual void print(ostream& o) const;
 
 public:
  
  //! @name Gestion des dépendances
  //@{
  /*! \brief Recalcule la variable si nécessaire
   *
   * Par le mécanisme de dépendances, cette opération est appelée récursivement
   * sur toutes les variables dont dépend l'instance. La fonction de recalcul
   * computeFunction() est ensuite appelée s'il s'avère qu'une des variables
   * dont elle dépend a été modifiée plus récemment.
   *
   * \pre computeFunction() != 0
   */
  virtual void update();

  virtual void update(Real wanted_time);

  /*! \brief Indique que la variable vient d'être mise à jour.
   *
   * Pour une gestion correcte des dépendances, il faut que cette propriété
   * soit appelée toutes les fois où la mise à jour d'une variable a été
   * effectuée.
   */
  virtual void setUpToDate();

  //! Temps auquel la variable a été mise à jour
  virtual Int64 modifiedTime();

  //! Ajoute \a var à la liste des dépendances avec les infos de trace \a tinfo
  virtual void addDepend(IVariable* var,eDependType dt,const TraceInfo& tinfo);

  /*! \brief Ajoute \a var à la liste des dépendances
   *
   */
  virtual void addDepend(IVariable* var,eDependType dt);

  /*! \brief Supprime \a var de la liste des dépendances
   */
  virtual void removeDepend(IVariable* var);

  /*! \brief Positionne la fonction de recalcule de la variable.
   *
   * Si une fonction de recalcule existait déjà, elle est détruite
   * et remplacée par celle-ci.
   */
  virtual void setComputeFunction(IVariableComputeFunction* v);

  //! Fonction utilisée pour mettre à jour la variable
  virtual IVariableComputeFunction* computeFunction();

  /*!
   * \brief Infos de dépendances.
   *
   * Remplit le tableau \a infos avec les infos de dépendance.
   */
  virtual void dependInfos(Array<VariableDependInfo>& infos);
  //@}

 public:

  virtual IMemoryAccessTrace* memoryAccessTrace() const;

  /*!
   * \brief Indique que la variable est synchronisée.
   *
   * Cette opération est collective.
   */
  virtual void setIsSynchronized();

  /*!
   * \brief Indique que la variable est synchronisée sur le group \a item_group
   *
   * Cette opération est collective.
   */
  virtual void setIsSynchronized(const ItemGroup& item_group);

public:
  //! Accès au gestionnaire de trace
  ITraceMng * traceMng() const 
  { 
    return m_trace_mng;
    // return _arcaneGetDefaultSubDomain()->traceMng();
    // return itemGroup().mesh()->traceMng(); 
  }

  //! Accès au gestionnaire de variable
  IVariableMng * variableMng() const 
  { 
    return m_variable_mng;
    // return _arcaneGetDefaultSubDomain()->variableMng();
    // return itemGroup().mesh()->subDomain()->variableMng(); 
  }

  const char * _className() const { return "ItemGroupMapAbstractVariable"; }
};

/*---------------------------------------------------------------------------*/

Int64 ItemGroupMapAbstractVariable::m_unique_id = 1;

/*---------------------------------------------------------------------------*/

ItemGroupMapAbstractVariable::
ItemGroupMapAbstractVariable(ItemGroupMapAbstract * map)
{
  m_map = map;
  StringBuilder _name("ItemGroupMap_");
  _name += itemGroup().name();
  _name += "_";
  _name += m_unique_id++;
  m_name = _name.toString();
  ARCANE_ASSERT((not itemGroup().null()),("Cannot create ItemGroupMap on null group")); // Possible problem : check group item type
  m_variable_mng = itemGroup().mesh()->subDomain()->variableMng();
  m_trace_mng = itemGroup().mesh()->traceMng();
  m_compute_function = NULL;
  m_used = true;
  
  Trace::Setter setter(traceMng(),_className()); 
  traceMng()->debug(Trace::Medium) << "Create " << this << " on " << itemGroup().name() << " (group size=" << itemGroup().size() << ")";
  if (Trace::High <= traceMng()->configDbgLevel()) {
    IStackTraceService* stack_service = platform::getStackTraceService();
    if (stack_service) {
      traceMng()->debug(Trace::High) << stack_service->stackTrace().toString();
    }
  }
}

ItemGroupMapAbstractVariable::
~ItemGroupMapAbstractVariable()
{
  Trace::Setter setter(traceMng(),_className());
  traceMng()->debug(Trace::Medium) << "Destroy " << this;
  if (m_compute_function) delete m_compute_function;
}

ISubDomain* 
ItemGroupMapAbstractVariable::
subDomain() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

Real 
ItemGroupMapAbstractVariable::
allocatedMemory() const 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

const String& 
ItemGroupMapAbstractVariable::
name() const 
{
  return m_name;
}

const String& 
ItemGroupMapAbstractVariable::
fullName() const 
{
  return m_name;
}

eDataType 
ItemGroupMapAbstractVariable::
dataType() const 
{
  return DT_Unknown;
}

eItemKind 
ItemGroupMapAbstractVariable::
itemKind() const 
{
  return itemGroup().itemKind();
}
 
Integer 
ItemGroupMapAbstractVariable::
dimension() const 
{
  return 0;
}

Integer 
ItemGroupMapAbstractVariable::
multiTag() const 
{
  return 0;
}

Integer 
ItemGroupMapAbstractVariable::
nbElement() const 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

int 
ItemGroupMapAbstractVariable::
property() const 
{
  return 
    // IVariable::PTemporary |
    IVariable::PNoDump    | 
    IVariable::PNoRestore |
    IVariable::PPrivate
    ;
}

void 
ItemGroupMapAbstractVariable::
notifyReferencePropertyChanged() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
addVariableRef(VariableRef* var_ref) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
removeVariableRef(VariableRef* var_ref) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

Integer 
ItemGroupMapAbstractVariable::
nbReference() const 
{
  return 1; // Uniquement son ItemGroupMapAbstract propriétaire
}

Expression ItemGroupMapAbstractVariable::
expression() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void ItemGroupMapAbstractVariable::
setTraceInfo(Integer id,eTraceType tt)
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void ItemGroupMapAbstractVariable::
resize(Integer new_size) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void ItemGroupMapAbstractVariable::
resizeFromGroup() 
{
  Trace::Setter setter(traceMng(),_className());
  traceMng()->debug(Trace::Medium) << "Resize " << this << " from " << m_map->_size() << " to " << itemGroup().size() << " on " << itemGroup().name();
  m_map->_resizeFromGroup();
  traceMng()->debug(Trace::High) << "End of resize on " << itemGroup().name();
}

bool 
ItemGroupMapAbstractVariable::
initialize(const ItemGroup& group,const String& value) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

Integer 
ItemGroupMapAbstractVariable::
checkIfSync(int max_print) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

Integer 
ItemGroupMapAbstractVariable::
checkIfSame(IDataReader* reader,int max_print,bool compare_ghost)
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
synchronize() 
{
  // No synchronization
}

IMesh* 
ItemGroupMapAbstractVariable::
mesh() const 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}
  
ItemGroup 
ItemGroupMapAbstractVariable::
itemGroup() const 
{
  if (m_map)
    return m_map->group();
  else
    return ItemGroup();
}

const String& 
ItemGroupMapAbstractVariable::
itemGroupName() const 
{
  return itemGroup().name();
}

IItemFamily* 
ItemGroupMapAbstractVariable::
itemFamily() const 
{
  return itemGroup().itemFamily();
}

const String& 
ItemGroupMapAbstractVariable::
itemFamilyName() const 
{
  return itemGroup().itemFamily()->name();
}

void 
ItemGroupMapAbstractVariable::
syncReferences() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
setUsed(bool v) 
{
  if (m_used == v)
    return;
  m_used = v;
  
  Trace::Setter setter(traceMng(),_className());
  if (m_used)
    traceMng()->fatal() << "setUsed(true) not implemented";

  ARCANE_ASSERT((m_map != NULL),("Inconsitent state: m_map not defined!"));
  itemGroup().internal()->detachObserver(this);
  m_map = NULL;
}

bool 
ItemGroupMapAbstractVariable::
isUsed() const 
{
  return m_used;
}

bool 
ItemGroupMapAbstractVariable::
isPartial() const 
{
  return !itemGroup().isAllItems();
}
  
void 
ItemGroupMapAbstractVariable::
copyItemsValues(Int32ConstArrayView source,Int32ConstArrayView destination) 
{
  Trace::Setter setter(traceMng(),_className());
  ARCANE_ASSERT((source.size()==destination.size()),("Inconsistent size arguments"));
  m_map->_copyItemsValues(source,destination);
}

void 
ItemGroupMapAbstractVariable::
copyItemsMeanValues(Int32ConstArrayView first_source,
                                 Int32ConstArrayView second_source,
                                 Int32ConstArrayView destination)
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void
ItemGroupMapAbstractVariable::
compact(Int32ConstArrayView new_to_old_ids) 
{
  Trace::Setter setter(traceMng(),_className());
  traceMng()->debug(Trace::Medium) << "Compact " << this << " on " << itemGroup().name() << " (group size=" << itemGroup().size() << ")";
  for(Integer i=0;i<new_to_old_ids.size();++i)
    traceMng()->debug(Trace::Highest) << " Matching : " << i << " " << new_to_old_ids[i];
  m_map->_compact2(new_to_old_ids);
  traceMng()->debug(Trace::High) << "End of compact on " << itemGroup().name();
}


void
ItemGroupMapAbstractVariable::
changeGroupIds(Int32ConstArrayView old_to_new_ids) 
{
  Trace::Setter setter(traceMng(),_className());
  traceMng()->debug(Trace::Medium) << "ChangeGroupIds " << this << " on " << itemGroup().name() << " (group size=" << itemGroup().size() << ")";
  for(Integer i=0;i<old_to_new_ids.size();++i)
    traceMng()->debug(Trace::Highest) << " Matching : " << i << " " << old_to_new_ids[i];
  m_map->_changeGroupIds(old_to_new_ids);
  traceMng()->debug(Trace::High) << "End of ChangeGroupIds on " << itemGroup().name();
}


IData*
ItemGroupMapAbstractVariable::
data()
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
serialize(ISerializer* sbuffer,IDataOperation* operation) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
serialize(ISerializer* sbuffer,Int32ConstArrayView ids,IDataOperation* operation) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
write(IDataWriter *) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
read(IDataReader*) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

IObservable* 
ItemGroupMapAbstractVariable::
writeObservable() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

IObservable* 
ItemGroupMapAbstractVariable::
readObservable() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

IObservable* 
ItemGroupMapAbstractVariable::
onSizeChangedObservable()
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
addTag(const String& tagname,const String& tagvalue) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
removeTag(const String& tagname) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

bool 
ItemGroupMapAbstractVariable::
hasTag(const String& tagname) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

String 
ItemGroupMapAbstractVariable::
tagValue(const String& tagname) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
print(ostream& o) const 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}
 
void 
ItemGroupMapAbstractVariable::
update() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void
ItemGroupMapAbstractVariable::
update(Real wanted_time) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
setUpToDate() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

Int64 
ItemGroupMapAbstractVariable::
modifiedTime() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
addDepend(IVariable* var,eDependType dt) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
addDepend(IVariable* var,eDependType dt,const TraceInfo& tinfo)
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
removeDepend(IVariable* var) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
dependInfos(Array<VariableDependInfo>& infos)
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
setComputeFunction(IVariableComputeFunction* v) 
{
  if (m_compute_function == v)
    return;
  delete m_compute_function; // fonctionne meme si NULL
  m_compute_function = v;
}

IVariableComputeFunction* 
ItemGroupMapAbstractVariable::
computeFunction() 
{
  return m_compute_function;
}

IMemoryAccessTrace* 
ItemGroupMapAbstractVariable::
memoryAccessTrace() const 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
setIsSynchronized() 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

void 
ItemGroupMapAbstractVariable::
setIsSynchronized(const ItemGroup& item_group) 
{
  throw FatalErrorException(A_FUNCINFO,"called");
}

VariableRef* ItemGroupMapAbstractVariable::firstReference() const
{
  throw NotImplementedException(A_FUNCINFO);
}

const String& ItemGroupMapAbstractVariable::
meshName() const
{
  return itemGroup().mesh()->name();
}

VariableMetaData* ItemGroupMapAbstractVariable::
createMetaData() const
{
  throw NotImplementedException(A_FUNCINFO);
}

void ItemGroupMapAbstractVariable::
notifyEndRead()
{
}

void ItemGroupMapAbstractVariable::
notifyBeginWrite()
{
}

Integer ItemGroupMapAbstractVariable::
checkIfSameOnAllReplica(Integer max_print)
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemGroupItemGroupMapObserver : public IItemGroupObserver
{
public:
  ItemGroupItemGroupMapObserver(IVariable * var) : m_var(var) { }
  virtual ~ItemGroupItemGroupMapObserver() { }

  void executeExtend(const Int32ConstArrayView * info) {
    if (info)
      m_var->changeGroupIds(*info);
    else
      m_var->resizeFromGroup();
  }
  void executeReduce(const Int32ConstArrayView * info) {
    if (info)
      m_var->changeGroupIds(*info);
    else
      m_var->resizeFromGroup();
  }
  void executeCompact(const Int32ConstArrayView * info) {
    if (info)
      m_var->compact(*info);
    else
      m_var->resizeFromGroup();
  }
  void executeReorder(const Int32ConstArrayView * info) {
    if (info)
      m_var->changeGroupIds(*info);
    else
      m_var->resizeFromGroup();
  }
  void executeInvalidate() {
    m_var->resizeFromGroup();
  }
  bool needInfo() const { return m_var->isPartial(); }

private:
    IVariable * m_var;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupMapAbstract::
~ItemGroupMapAbstract()
{
  ARCANE_ASSERT(((m_variable==NULL)==(m_group->null())),("Inconsistent state: ItemGroupMap with group but without Variable"));
  if (!m_group->null()){
    //GG: supprime car fait planter lors du removeVariable() et n'est pas utile
    //m_variable->setUsed(false);
    m_group->detachObserver(m_variable);
    dynamic_cast<ItemGroupMapAbstractVariable*>(m_variable)->variableMng()->removeVariable(m_variable);
  }
}

/*---------------------------------------------------------------------------*/

void
ItemGroupMapAbstract::
_init(const ItemGroup & group) 
{
//   if (group.isAllItems())
//     throw Arcane::FatalErrorException(A_FUNCINFO,"AllItems group are not allowed in this release of ItemGroupMap");

  if (!m_group->null())
    {
      if (m_group != group.internal()) {
        m_group->detachObserver(m_variable);
        dynamic_cast<ItemGroupMapAbstractVariable*>(m_variable)->variableMng()->removeVariable(m_variable);
        m_group = group.internal();
        m_variable = new ItemGroupMapAbstractVariable(this);
        dynamic_cast<ItemGroupMapAbstractVariable*>(m_variable)->variableMng()->addVariable(m_variable);        
        m_group->attachObserver(m_variable,new ItemGroupItemGroupMapObserver(m_variable));
      }
    }
  else
    {
      m_group = group.internal();
      m_variable = new ItemGroupMapAbstractVariable(this);
      dynamic_cast<ItemGroupMapAbstractVariable*>(m_variable)->variableMng()->addVariable(m_variable);
      m_group->attachObserver(m_variable,new ItemGroupItemGroupMapObserver(m_variable));
    }

  if (group.isAllItems())
    traceMng()->warning() << "ItemGroupMap on 'AllItems' groups is unstable";
}

/*---------------------------------------------------------------------------*/

ITraceMng *
ItemGroupMapAbstract::
traceMng() const 
{
  if (m_variable)
    return dynamic_cast<ItemGroupMapAbstractVariable*>(m_variable)->traceMng();
  return NULL;
}

/*---------------------------------------------------------------------------*/

bool
ItemGroupMapAbstract::
checkSameGroup(const ItemGroup & group) const
{
  _checkGroupIntegrity();
  if (group.internal() == m_group)
    return true;
  if (group.size()       != m_group->size()     ||
      group.itemKind()   != m_group->itemKind() ||
      group.itemFamily() != m_group->itemFamily())
    return false;
  // Ici test sur les items mais meme taille, meme type, meme famille
  // On se fie aux ItemInternal's (meme les localIds peuvent suffir)
  const ItemInternalArrayView orgItems = m_group->itemsInternal();
  const ItemInternalArrayView testItems = group.internal()->itemsInternal();
  const Integer size = orgItems.size(); 
  for(Integer i=0;i<size;++i)
    if (orgItems[i] != testItems[i])
      return false;
  return true;
}

/*---------------------------------------------------------------------------*/

bool
ItemGroupMapAbstract::
checkSameGroup(const ItemVectorView & group) const
{
  _checkGroupIntegrity();
  if (group.size() != m_group->size()) {
    traceMng()->error() << "Bad sizes";
    return false;
  }
  // Ici test sur les items mais meme taille
  // On se fie donc aux ItemInternal's
  const ItemInternalArrayView orgItems = m_group->itemsInternal();
  const ItemInternalArrayView testItems = group.items();
  const Integer size = orgItems.size(); 
  for(Integer i=0;i<size;++i)
    if (orgItems[i] != testItems[i]) {
      traceMng()->error() << "Item " << i << " don't match : " << orgItems[i] << " vs " << testItems[i];   
      return false;
    }
  return true;
}
#endif
